import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

from tqdm import tqdm
from torch import optim
from baseline.datasets import FGSBIR_Dataset
from baseline.utils import loss_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloader(args):
    dataset_train = FGSBIR_Dataset(args, mode='train')
    dataloader_train = data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=int(args.threads))

    dataset_test = FGSBIR_Dataset(args, mode='test')
    dataloader_test = data.DataLoader(
        dataset_test, batch_size=args.test_batch_size, shuffle=False, num_workers=int(args.threads))

    return dataloader_train, dataloader_test
    
def evaluate_model(model, dataloader_test, args):
    with torch.no_grad():
        model.eval()
        sketch_array_tests = []
        sketch_names = []
        image_array_tests = torch.FloatTensor().to(device)
        image_names = []

        for idx, batch in enumerate(tqdm(dataloader_test)):
            sketch_features_all = torch.FloatTensor().to(device)
            # print(batch['sketch_imgs'].shape) # (1, 25, 3, 299, 299)
            
            for data_sketch in batch['sketch_imgs']:
                sketch_feature = model.encode_sketch(data_sketch.to(device))
                
                sketch_features_all = torch.cat((sketch_features_all, sketch_feature.detach()))

            # print("sketch_feature_ALL.shape: ", sketch_features_all.shape) # (25, 2048)
            sketch_array_tests.append(sketch_features_all.cpu())
            sketch_names.extend(batch['sketch_path'])

            if batch['positive_path'][0] not in image_names:
                positive_feature = model.encode_sample(batch['positive_img'].to(device))
                
                image_array_tests = torch.cat((image_array_tests, positive_feature))
                image_names.extend(batch['positive_path'])
                
        # print("image_array_tests shape", image_array_tests.shape)
        # print("image_array_tests shape", image_array_tests.shape)
        # print("sketch_array_tests shape", sketch_array_tests[0].shape) #(25, 2048)
        
        num_steps = len(sketch_array_tests[0])
        sketch_range = []
        
        rank_all = torch.zeros(len(sketch_array_tests), num_steps)
        rank_all_percentile = torch.zeros(len(sketch_array_tests), num_steps)
        sketch_range = torch.Tensor(sketch_range)
        
        for i_batch, sampled_batch in enumerate(sketch_array_tests):
            sketch_name = sketch_names[i_batch]

            sketch_query_name = '_'.join(sketch_name.split('/')[-1].split('_')[:-1])
            position_query = image_names.index(sketch_query_name)
            sketch_features = sampled_batch

            for i_sketch in range(sampled_batch.shape[0]):
                sketch_feature = sketch_features[i_sketch]
                target_distance = F.pairwise_distance(sketch_feature.to(device), image_array_tests[position_query].to(device))
                distance = F.pairwise_distance(sketch_feature.unsqueeze(0).to(device), image_array_tests.to(device))
                
                rank_all[i_batch, i_sketch] = distance.le(target_distance).sum()
                rank_all_percentile[i_batch, i_sketch] = (len(distance) - rank_all[i_batch, i_sketch]) / (len(distance) - 1)
                
        top1_accuracy = rank_all[:, -1].le(1).sum().numpy() / rank_all.shape[0]
        top5_accuracy = rank_all[:, -1].le(5).sum().numpy() / rank_all.shape[0]
        top10_accuracy = rank_all[:, -1].le(10).sum().numpy() / rank_all.shape[0]

        return top1_accuracy, top5_accuracy, top10_accuracy

def get_unique_filename(save_dir, base_name="results_log.txt"):
    filename = base_name
    name, ext = os.path.splitext(base_name)
    counter = 0
    while os.path.exists(os.path.join(save_dir, filename)):
        counter += 1
        filename = f"{name}_{counter}{ext}"
    return filename


def save_component_states(model, save_dir, dataset_name, rank_name, update_stage2_alias=False):
    backbone_state = {
        'sample_embedding_network': model.sample_embedding_network.state_dict(),
        'sketch_embedding_network': model.sketch_embedding_network.state_dict(),
    }
    attention_state = {
        'attention': model.attention.state_dict(),
        'sketch_attention': model.sketch_attention.state_dict(),
    }
    linear_state = {
        'linear': model.linear.state_dict(),
        'sketch_linear': model.sketch_linear.state_dict(),
    }

    torch.save(backbone_state, os.path.join(save_dir, f"{dataset_name}_{rank_name}_backbone.pth"))
    torch.save(attention_state, os.path.join(save_dir, f"{dataset_name}_{rank_name}_attention.pth"))
    torch.save(linear_state, os.path.join(save_dir, f"{dataset_name}_{rank_name}_linear.pth"))

    if update_stage2_alias:
        torch.save(backbone_state, os.path.join(save_dir, f"{dataset_name}_backbone.pth"))
        torch.save(attention_state, os.path.join(save_dir, f"{dataset_name}_attention.pth"))
        torch.save(linear_state, os.path.join(save_dir, f"{dataset_name}_linear.pth"))


def get_optimizer(model, args):
    if getattr(model, "is_vit", False):
        backbone_lr = args.backbone_lr if args.backbone_lr > 0 else args.lr * 0.1
        return optim.AdamW([
            {'params': model.sample_embedding_network.parameters(), 'lr': backbone_lr},
            {'params': model.sketch_embedding_network.parameters(), 'lr': backbone_lr},
            {'params': model.linear.parameters(), 'lr': args.lr},
            {'params': model.sketch_linear.parameters(), 'lr': args.lr},
            {'params': model.attention.parameters(), 'lr': args.lr},
            {'params': model.sketch_attention.parameters(), 'lr': args.lr},
        ], weight_decay=args.weight_decay)

    return optim.Adam(params=model.parameters(), lr=args.lr)


def train_model(model, args):
    model = model.to(device)
    dataloader_train, dataloader_test = get_dataloader(args)
    if args.load_pretrained:
        model.load_state_dict(torch.load(args.pretrained_dir, map_location=device), strict=False)
    os.makedirs(args.save_dir, exist_ok=True)
    filename = get_unique_filename(args.save_dir, "results_log.txt")
    
    lr = args.lr
    # loss_fn = nn.TripletMarginLoss(margin=args.margin)
    optimizer = get_optimizer(model, args)
    # optimizer = optim.AdamW([
    #     {'params': model.sample_embedding_network.parameters(), 'lr': lr},
    #     {'params': model.sketch_embedding_network.parameters(), 'lr': lr},
    # ])
    # scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    top5, top10, avg_loss = 0, 0, 0
    for i_epoch in range(args.epochs):
        print(f"Epoch: {i_epoch+1} / {args.epochs}")
                
        losses = []
        for _, batch_data in enumerate(tqdm(dataloader_train, dynamic_ncols=False)):
            model.train()
            optimizer.zero_grad()

            features = model(batch_data)
            loss = loss_fn(args, features)
            loss.backward()
            if getattr(model, "is_vit", False) and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)
        
        top1_eval, top5_eval, top10_eval = evaluate_model(
            model, dataloader_test, args)
            
        if top5_eval > top5:
            top5 = top5_eval
            torch.save(model.state_dict(), os.path.join(args.save_dir, args.dataset_name + '_best_top5.pth'))
            save_component_states(model, args.save_dir, args.dataset_name, 'top5')

        if top10_eval > top10:
            top10 = top10_eval
            torch.save(model.state_dict(), os.path.join(args.save_dir, args.dataset_name + '_best_top10.pth'))
            save_component_states(model, args.save_dir, args.dataset_name, 'top10', update_stage2_alias=True)
            
        torch.save(model.state_dict(), os.path.join(args.save_dir, args.dataset_name + "_last_model.pth"))
        torch.save(model.state_dict(), os.path.join(args.save_dir, "last_model.pth"))
        
        
        print('Top 1 accuracy : {:.5f}'.format(top1_eval))
        print('Top 5 accuracy : {:.5f}'.format(top5_eval))
        print('Top 10 accuracy: {:.5f}'.format(top10_eval))
        print('Loss:            {:.5f}'.format(avg_loss))
        
        with open(os.path.join(args.save_dir, filename), "a") as f:
            f.write("Epoch {:d} | Top1: {:.5f} | Top5: {:.5f} | Top10: {:.5f} | Loss: {:.5f}\n".format(
                i_epoch+1, top1_eval, top5_eval, top10_eval, avg_loss))
