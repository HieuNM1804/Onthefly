import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from torchvision.models import Inception_V3_Weights, vit_b_32, ViT_B_32_Weights, resnet50, ResNet50_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        backbone = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)

        # Giữ lại các layer của ViT (bỏ classification head)
        self.patch_embedding  = backbone.conv_proj       # Conv2d patch projection
        self.class_token      = backbone.class_token     # CLS token
        self.pos_embedding    = backbone.encoder.pos_embedding
        self.dropout          = backbone.encoder.dropout
        self.encoder_layers   = backbone.encoder.layers  # 12 transformer blocks
        self.ln               = backbone.encoder.ln      # LayerNorm cuối

        # ViT-B/32 với input 224x224 → 7x7 = 49 patches, hidden_dim = 768
        self.num_patches = 49   # 7x7
        self.grid_size   = 7
        self.hidden_dim  = 768

    def forward(self, x):
        # x: (N, 3, 224, 224)
        B = x.shape[0]

        # Patch embedding: (N, 768, 7, 7) → flatten → (N, 49, 768)
        x = self.patch_embedding(x)                        # (N, 768, 7, 7)
        x = x.flatten(2).transpose(1, 2)                  # (N, 49, 768)

        # Thêm CLS token: (N, 50, 768)
        cls_token = self.class_token.expand(B, -1, -1)    # (N, 1, 768)
        x = torch.cat([cls_token, x], dim=1)              # (N, 50, 768)

        # Positional embedding + dropout
        x = self.dropout(x + self.pos_embedding)          # (N, 50, 768)

        # Qua các transformer block
        x = self.encoder_layers(x)                        # (N, 50, 768)
        x = self.ln(x)                                    # (N, 50, 768)

        cls_token = x[:, 0, :]                            # (N, 768)

        return cls_token
        # # Bỏ CLS token, lấy 49 patch tokens
        # patch_tokens = x[:, 1:, :]                        # (N, 49, 768)

        # # Reshape về spatial feature map giống InceptionV3
        # x = patch_tokens.transpose(1, 2)                  # (N, 768, 49)
        # x = x.reshape(B, self.hidden_dim,
        #                self.grid_size, self.grid_size)    # (N, 768, 7, 7)

        # return x   # Tương đương (N, 2048, 8, 8) của InceptionV3

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