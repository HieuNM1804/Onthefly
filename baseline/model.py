import torch
import torch.nn as nn

from baseline.backbones import build_backbone, get_backbone_feature_dim, is_vit_backbone
from baseline.attention import IdentityAttention, Linear_global, SelfAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Siamese_SBIR(nn.Module):
    def __init__(self, args):
        super(Siamese_SBIR, self).__init__()
        self.args = args
        self.is_vit = is_vit_backbone(args.backbone)

        self.sample_embedding_network = build_backbone(args)
        self.sketch_embedding_network = build_backbone(args)
        self.in_feature = getattr(
            self.sample_embedding_network,
            "hidden_dim",
            get_backbone_feature_dim(args.backbone),
        )

        attention_cls = IdentityAttention if self.is_vit else SelfAttention
        self.attention = attention_cls(args, in_feature=self.in_feature)
        self.sketch_attention = attention_cls(args, in_feature=self.in_feature)

        self.linear = Linear_global(feature_num=self.args.output_size, in_features=self.in_feature)
        self.sketch_linear = Linear_global(feature_num=self.args.output_size, in_features=self.in_feature)

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.kaiming_normal_(m.weight)
        
        if self.args.use_kaiming_init:
            self.attention.apply(init_weights)
            self.sketch_attention.apply(init_weights)
            
            self.linear.apply(init_weights)
            self.sketch_linear.apply(init_weights)

    def encode_sample(self, image):
        feature = self.sample_embedding_network(image)
        feature = self.attention(feature)
        return self.linear(feature)

    def encode_sketch(self, image):
        feature = self.sketch_embedding_network(image)
        feature = self.sketch_attention(feature)
        return self.sketch_linear(feature)
            
    def extract_feature(self, batch, num):
        sketch_img = batch[f'sketch_img_{num}'].to(device)
        positive_img = batch[f'positive_img_{num}'].to(device)
        negative_img = batch[f'negative_img_{num}'].to(device)
        
        positive_feature = self.encode_sample(positive_img)
        negative_feature = self.encode_sample(negative_img)
        sketch_feature = self.encode_sketch(sketch_img)
        
        return sketch_feature, positive_feature, negative_feature
    
    def forward(self, batch):
        outputs = {}
        for i in range(1, self.args.num_views+1):
            sketch_feature, positive_feature, negative_feature = self.extract_feature(batch=batch, num=i)
            outputs[f'sketch_feature_{i}']   = sketch_feature
            outputs[f'positive_feature_{i}'] = positive_feature
            outputs[f'negative_feature_{i}'] = negative_feature
            
        return outputs
