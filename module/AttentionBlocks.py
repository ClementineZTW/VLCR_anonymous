import torch
from torch import nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self,embed_dim,num_heads,supervised,mlp_ratio=4,dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.supervised = supervised
        self.dropout = dropout
        self.attn = nn.MultiheadAttention(embed_dim,num_heads,dropout,batch_first=True)
        self.first_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim,mlp_ratio*embed_dim),
            nn.ReLU(),
            nn.Linear(mlp_ratio*embed_dim,embed_dim),
            nn.Dropout(dropout)
        )
        self.second_norm = nn.LayerNorm(embed_dim)


    def forward(self,q,k,v):
        res_q = q
        (q,weight) = self.attn(q,k,v)
        q = res_q + q
        q = self.first_norm(q)

        res_q = q
        q = self.ffn(q)
        q = res_q + q
        q = self.second_norm(q)
        return q,weight


class VisualAttentionBlock(AttentionBlock):
    def forward(self,q,feat):
        return super().forward(q,feat,feat)

class EyeAttentionBlock(nn.Module):
    def __init__(self,embed_dim,num_heads,supervised,mlp_ratio=4,dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.supervised = supervised
        self.dropout = dropout
        self.attn = nn.MultiheadAttention(embed_dim,num_heads,dropout,batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self,q,feat):
        (q,weight) = self.attn(q,feat,feat)
        q = self.norm(q)
        return q,weight    

class LinguisticAttentionBlock(AttentionBlock):
    def forward(self,q):
        return super().forward(q,q,q)


class Transformer3Layers(nn.Module):
    def __init__(self,embed_dim,num_heads,supervised,mlp_ratio=4,dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.supervised = supervised
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.blocks = nn.Sequential(
            LinguisticAttentionBlock(embed_dim,num_heads,supervised,mlp_ratio=4,dropout=0.1),
            LinguisticAttentionBlock(embed_dim,num_heads,supervised,mlp_ratio=4,dropout=0.1),
            LinguisticAttentionBlock(embed_dim,num_heads,supervised,mlp_ratio=4,dropout=0.1)
        )
    def forward(self,q):
        for block in self.blocks:
            (q,weight) = block(q)
        return q,weight


class TokenLearner(nn.Module):
    
    def __init__(self, input_embed_dim, num_heads, supervised, out_token=27):
        super().__init__()
        self.supervised = supervised

        self.token_norm = nn.LayerNorm(input_embed_dim)
        self.tokenLearner = nn.Sequential(nn.Conv2d(input_embed_dim, input_embed_dim, kernel_size = (1,1), stride=1, groups=8, bias=False),
                                          nn.Conv2d(input_embed_dim, out_token, kernel_size = (1,1), stride=1, bias=False))
        self.feat = nn.Conv2d(input_embed_dim, input_embed_dim, kernel_size = (1,1), stride=1, groups=8, bias=False)
        self.norm = nn.LayerNorm(input_embed_dim)

    def forward(self,q,feat):
        x = feat
        B = x.shape[0]
        x = self.token_norm(x) # [bs, 257, 768]
        x = x.transpose(1, 2).unsqueeze(-1) # [bs, 768, 257, 1]
        selected = self.tokenLearner(x) # [bs, 27, 257, 1].
        selected = selected.flatten(2)  # [bs, 27, 257].
        selected = F.softmax(selected, dim=-1) 
        feat = self.feat(x) #  [bs, 768, 257, 1].
        feat = feat.flatten(2).transpose(1,2)  # [bs, 257, 768]
        x = torch.einsum('...si,...id->...sd', selected, feat) # [bs, 27, 768]
        
        x = self.norm(x)
        return (x,selected)


if __name__ == '__main__':
    q = torch.randn((10,27,192))
    feat = torch.randn((10,256,192))
    model = Comparer(192,3,27,False)
    res = model(q)
    print(res[0].shape)#,res[1].shape)
