import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline.backbones import build_backbone, get_backbone_feature_dim, is_vit_backbone
from baseline.attention import IdentityAttention, Linear_global, SelfAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SketchAttention(nn.Module):
    def __init__(self, args, in_features):
        super(SketchAttention, self).__init__()
        self.args = args
        self.in_features = in_features
        self.norm = nn.LayerNorm(in_features)
        self.mha = nn.MultiheadAttention(in_features, num_heads=args.num_heads, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        
        self.proj = Linear_global(feature_num=args.output_size, in_features=in_features)
        
    def forward(self, x):
        squeeze_batch = x.dim() == 2
        if squeeze_batch:
            x = x.unsqueeze(0)

        identify = x
        x_att = self.norm(x)
        att_out, _  = self.mha(x_att, x_att, x_att)
        att_out = self.dropout(att_out)
        
        if self.args.state == 0:
            attn = identify * att_out + identify
        elif self.args.state == 1:
            attn = identify + att_out
        elif self.args.state == 2:
            attn = identify * att_out
        else:
            raise ValueError(f"Unsupported residual state: {self.args.state}")

        attn = F.normalize(attn, dim=-1)
        output = self.proj(attn)

        if squeeze_batch:
            output = output.squeeze(0)
        
        return output

class Siamese_SBIR(nn.Module):
    def __init__(self, args):
        super(Siamese_SBIR, self).__init__()
        self.args = args
        self.is_vit = is_vit_backbone(args.backbone)

        self.sample_embedding_network = build_backbone(args)
        self.in_feature = getattr(
            self.sample_embedding_network,
            "hidden_dim",
            get_backbone_feature_dim(args.backbone),
        )
        attention_cls = IdentityAttention if self.is_vit else SelfAttention
        self.attention = attention_cls(args, in_feature=self.in_feature)
        self.linear = Linear_global(feature_num=args.output_size, in_features=self.in_feature)

        self.sketch_embedding_network = build_backbone(args)
        self.sketch_attention = attention_cls(args, in_feature=self.in_feature)
        self.sketch_linear = Linear_global(feature_num=args.output_size, in_features=self.in_feature)
        
        self.sample_embedding_network.fix_weights()
        self.sketch_embedding_network.fix_weights()
        self.attention.fix_weights()
        self.sketch_attention.fix_weights()
        self.linear.fix_weights()
        self.sketch_linear.fix_weights()
        
        self.attn = SketchAttention(args, in_features=self.in_feature)

    def encode_sample(self, image):
        feature = self.sample_embedding_network(image)
        feature = self.attention(feature)
        return self.linear(feature)

    def encode_sketch_base(self, image):
        feature = self.sketch_embedding_network(image)
        return self.sketch_attention(feature)

if __name__ == "__main__":
    dim = 2048
    dt_rank = 4
    dim_inner = 32
    d_state = 8
