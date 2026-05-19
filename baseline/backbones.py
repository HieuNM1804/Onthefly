import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from torchvision.models import (
    Inception_V3_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
    ViT_B_32_Weights,
    resnet50,
    vit_b_16,
    vit_b_32,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_vit_backbone(backbone_name):
    return str(backbone_name).lower() in {"vit", "vit_b_16", "vit-b-16", "vit_b_32", "vit-b-32"}


def get_backbone_feature_dim(backbone_name):
    return 768 if is_vit_backbone(backbone_name) else 2048


def build_backbone(args):
    backbone_name = str(args.backbone)
    normalized_name = backbone_name.lower()
    if normalized_name == "inceptionv3":
        return InceptionV3(args)
    if normalized_name == "resnet50":
        return ResNet50(args)
    if normalized_name in {"vit", "vit_b_16", "vit-b-16", "vit_b_32", "vit-b-32"}:
        return ViT(args)
    raise ValueError(f"Unsupported backbone: {backbone_name}")

class InceptionV3(nn.Module):
    def __init__(self, args):
        super(InceptionV3, self).__init__()
        backbone = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
        # backbone = models.inception_v3()
        
        ## Extract Inception Layers ##
        self.Conv2d_1a_3x3 = backbone.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = backbone.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = backbone.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = backbone.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = backbone.Conv2d_4a_3x3
        self.Mixed_5b = backbone.Mixed_5b
        self.Mixed_5c = backbone.Mixed_5c
        self.Mixed_5d = backbone.Mixed_5d
        self.Mixed_6a = backbone.Mixed_6a
        self.Mixed_6b = backbone.Mixed_6b
        self.Mixed_6c = backbone.Mixed_6c
        self.Mixed_6d = backbone.Mixed_6d
        self.Mixed_6e = backbone.Mixed_6e

        self.Mixed_7a = backbone.Mixed_7a
        self.Mixed_7b = backbone.Mixed_7b
        self.Mixed_7c = backbone.Mixed_7c
        self.pool_method =  nn.AdaptiveMaxPool2d(1) # as default
        self.args = args

    def forward(self, x):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        feature_maps_6b = x
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        
        return x
        
    def fix_weights(self):
        for x in self.parameters():
            x.requires_grad = False
            
class ViT(nn.Module):
    def __init__(self, args):
        super(ViT, self).__init__()
        self.args = args

        vit_variant = str(getattr(args, "vit_variant", "b16")).lower()
        if vit_variant in {"b32", "vit_b_32", "vit-b-32"}:
            self.model_name = "vit_b_32"
            backbone = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
        elif vit_variant in {"b16", "vit_b_16", "vit-b-16"}:
            self.model_name = "vit_b_16"
            backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported vit_variant: {vit_variant}. Use b16 or b32.")

        # Giữ lại các layer của ViT (bỏ classification head)
        self.patch_embedding  = backbone.conv_proj       # Conv2d patch projection
        self.class_token      = backbone.class_token     # CLS token
        self.pos_embedding    = backbone.encoder.pos_embedding
        self.dropout          = backbone.encoder.dropout
        self.encoder_layers   = backbone.encoder.layers  # 12 transformer blocks
        self.ln               = backbone.encoder.ln      # LayerNorm cuối

        self.hidden_dim = 768

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)

        cls_token = self.class_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        x = self.dropout(x + self.pos_embedding)

        x = self.encoder_layers(x)
        x = self.ln(x)

        cls_token = x[:, 0, :]

        return F.normalize(cls_token, dim=-1)

    def fix_weights(self):
        for param in self.parameters():
            param.requires_grad = False


class ResNet50(nn.Module):
    def __init__(self, args):
        super(ResNet50, self).__init__()
        self.args = args

        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Giữ lại các layer, bỏ AvgPool + FC ở cuối
        self.conv1   = backbone.conv1    # (N, 64, 112, 112)
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool  # (N, 64, 56, 56)

        self.layer1  = backbone.layer1  # (N, 256,  56, 56)
        self.layer2  = backbone.layer2  # (N, 512,  28, 28)
        self.layer3  = backbone.layer3  # (N, 1024, 14, 14)
        self.layer4  = backbone.layer4  # (N, 2048,  7,  7)

    def forward(self, x):
        # x: (N, 3, 224, 224)
        x = self.conv1(x)    # (N, 64, 112, 112)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # (N, 64, 56, 56)

        x = self.layer1(x)   # (N, 256,  56, 56)
        x = self.layer2(x)   # (N, 512,  28, 28)
        x = self.layer3(x)   # (N, 1024, 14, 14)
        x = self.layer4(x)   # (N, 2048,  7,  7)

        return x  # Tương đương (N, 2048, 8, 8) của InceptionV3

    def fix_weights(self):
        for param in self.parameters():
            param.requires_grad = False
