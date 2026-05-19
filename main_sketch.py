import torch
import argparse
import os
from phase2.model import Siamese_SBIR
from phase2.train import train_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def load_checkpoint(path):
    return torch.load(path, map_location=device, weights_only=True)


def load_projector_from_prefix(projector, state_dict, prefix):
    projector_state = {
        key.replace(prefix, "", 1): value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if projector_state:
        projector.load_state_dict(projector_state, strict=False)


def load_stage2_from_baseline(model, args):
    pretrained_path = args.pretrained_dir
    if os.path.isfile(pretrained_path):
        baseline_state = load_checkpoint(pretrained_path)
        model.load_state_dict(baseline_state, strict=False)
        load_projector_from_prefix(model.attn.proj, baseline_state, "sketch_linear.")
        print(f"Loaded ViTS baseline checkpoint: {pretrained_path}")
        return

    if not os.path.isdir(pretrained_path):
        raise FileNotFoundError(f"Pretrained path not found: {pretrained_path}")

    backbones_state = load_checkpoint(os.path.join(pretrained_path, args.dataset_name + "_backbone.pth"))
    attention_state = load_checkpoint(os.path.join(pretrained_path, args.dataset_name + "_attention.pth"))
    linear_state = load_checkpoint(os.path.join(pretrained_path, args.dataset_name + "_linear.pth"))

    model.sample_embedding_network.load_state_dict(backbones_state['sample_embedding_network'], strict=False)
    model.sketch_embedding_network.load_state_dict(backbones_state['sketch_embedding_network'], strict=False)
    model.attention.load_state_dict(attention_state['attention'], strict=False)
    model.sketch_attention.load_state_dict(attention_state['sketch_attention'], strict=False)
    model.linear.load_state_dict(linear_state['linear'])
    model.sketch_linear.load_state_dict(linear_state['sketch_linear'])
    model.attn.proj.load_state_dict(linear_state['sketch_linear'], strict=False)
    print(f"Loaded ViTS split checkpoint directory: {pretrained_path}")


if __name__ == "__main__":
    parsers = argparse.ArgumentParser(description='Baseline Fine-Grained SBIR model')
    parsers.add_argument('--backbone', type=str, default='ViTS', help="InceptionV3/ViT/ViTS/ResNet50")
    parsers.add_argument('--vit_variant', type=str, default='b16', help="b16/b32 for torchvision ViT")
    parsers.add_argument('--vits_model_name', type=str, default='vit_small_patch16_224.augreg_in1k',
                         help="timm ViT-S ImageNet checkpoint")
    parsers.add_argument('--dataset_name', type=str, default='ShoeV2')
    parsers.add_argument('--output_size', type=int, default=64)
    parsers.add_argument('--num_heads', type=int, default=8)
    parsers.add_argument('--root_dir', type=str, default='/kaggle/input/fg-sbir-dataset')
    parsers.add_argument('--pretrained_dir', type=str, default='/kaggle/input/base_ae_model/pytorch/default/1/best_model.pth')
    parsers.add_argument('--save_dir', type=str, default='/kaggle/working/')
    
    parsers.add_argument('--use_kaiming_init', type=str2bool, nargs='?', const=True, default=True)
    parsers.add_argument('--load_pretrained', type=str2bool, nargs='?', const=True, default=False)
    parsers.add_argument('--state', type=int, default=0, help="0: full residual, 1: additive residual, 2: multiplicative residual")
    
    
    parsers.add_argument('--batch_size', type=int, default=48)
    parsers.add_argument('--test_batch_size', type=int, default=1)
    parsers.add_argument('--steps', type=int, default=20)
    parsers.add_argument('--gamma', type=float, default=0.5)
    parsers.add_argument('--margin', type=float, default=0.3)
    parsers.add_argument('--alpha', type=float, default=0.9)
    parsers.add_argument('--temperature', default=0.07, type=float, help='softmax temperature (default: 0.07)')
    parsers.add_argument('--threads', type=int, default=4)
    parsers.add_argument('--lr', type=float, default=0.0001)
    parsers.add_argument('--epochs', type=int, default=100)
    
    args = parsers.parse_args()
    model = Siamese_SBIR(args).to(device)

    if args.load_pretrained is False:
        load_stage2_from_baseline(model, args)
    
    train_model(model, args)
