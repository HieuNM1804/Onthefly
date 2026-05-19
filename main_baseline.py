import torch
import argparse
from baseline.model import Siamese_SBIR
from baseline.train import train_model

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


if __name__ == "__main__":
    parsers = argparse.ArgumentParser(description='Baseline Fine-Grained SBIR model')
    parsers.add_argument('--backbone', type=str, default='ViT', help="InceptionV3/ViT/ResNet50")
    parsers.add_argument('--vit_variant', type=str, default='b16',
                         help="b16 or b32 for torchvision ViT-B ImageNet-1K")
    parsers.add_argument('--vit_pool', type=str, default='cls_mean', help="cls/mean/cls_mean")
    parsers.add_argument('--dataset_name', type=str, default='ShoeV2')
    parsers.add_argument('--output_size', type=int, default=64)
    parsers.add_argument('--num_heads', type=int, default=8)
    parsers.add_argument('--root_dir', type=str, default='/kaggle/input/fg-sbir-dataset')
    parsers.add_argument('--pretrained_dir', type=str, default='/kaggle/input/base_ae_model/pytorch/default/1/best_model.pth')
    parsers.add_argument('--save_dir', type=str, default='/kaggle/working/')
    
    parsers.add_argument('--use_kaiming_init', type=str2bool, nargs='?', const=True, default=True)
    parsers.add_argument('--load_pretrained', type=str2bool, nargs='?', const=True, default=False)
    parsers.add_argument('--use_info', type=str2bool, nargs='?', const=True, default=False)
    
    parsers.add_argument('--batch_size', type=int, default=64)
    parsers.add_argument('--test_batch_size', type=int, default=1)
    parsers.add_argument('--step_size', type=int, default=100)
    parsers.add_argument('--gamma', type=float, default=0.5)
    parsers.add_argument('--use_scheduler', type=str2bool, nargs='?', const=True, default=True)
    parsers.add_argument('--scheduler_metric', type=str, default='top10', help="top1/top5/top10/loss")
    parsers.add_argument('--scheduler_patience', type=int, default=10)
    parsers.add_argument('--scheduler_min_lr', type=float, default=1e-7)
    parsers.add_argument('--margin', type=float, default=0.3)
    parsers.add_argument('--alpha', type=float, default=1)
    parsers.add_argument('--num_views', type=int, default=2, help='view 1 - 5')
    parsers.add_argument('--temperature', default=0.07, type=float, help='softmax temperature (default: 0.07)')
    parsers.add_argument('--threads', type=int, default=4)
    parsers.add_argument('--lr', type=float, default=0.0001)
    parsers.add_argument('--backbone_lr', type=float, default=0.0,
                         help="ViT backbone lr. If 0, use lr * 0.1")
    parsers.add_argument('--weight_decay', type=float, default=0.05)
    parsers.add_argument('--grad_clip', type=float, default=1.0)
    parsers.add_argument('--epochs', type=int, default=200)
    
    args = parsers.parse_args()
    model = Siamese_SBIR(args).to(device)
    train_model(model, args)
