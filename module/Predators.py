import torch
from torch import nn
from timm.models.layers import PatchEmbed

class AttentionBlock(nn.Module):
    def __init__(self,input_dim,output_dim,num_heads,mlp_ratio=4,dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim =  output_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.attn = nn.MultiheadAttention(output_dim,num_heads,dropout,batch_first=True)
        self.first_norm = nn.LayerNorm(output_dim)
        self.ffn = nn.Sequential(
            nn.Linear(output_dim,mlp_ratio*output_dim),
            nn.ReLU(),
            nn.Linear(mlp_ratio*output_dim,output_dim),
            nn.Dropout(dropout)
        )
        self.second_norm = nn.LayerNorm(output_dim)


    def forward(self,q,k,v,mask=None):
        res_q = q
        (q,weight) = self.attn(q,k,v,attn_mask=mask)
        q = res_q + q
        q = self.first_norm(q)

        res_q = q
        q = self.ffn(q)
        q = res_q + q
        q = self.second_norm(q)
        return q,weight


class Predator(nn.Module):
    def __init__(self,embed_dim,num_heads,depth,num_class=38,batch_max_length=27,mlp_ratio=4,dropout=0.1,patch_size=(4,4),img_size=(32,128)):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.num_class = num_class
        self.batch_max_length = batch_max_length
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.patch_size = patch_size
        self.img_size = img_size

        self.patch_embed = PatchEmbed(img_size,patch_size,embed_dim=self.embed_dim)
        num_patches = img_size[0]*img_size[1]//patch_size[0]//patch_size[1]
        
        self.char_tokens = nn.Parameter(torch.zeros((1,self.batch_max_length,self.embed_dim)))
        self.pos_embed = nn.Parameter(torch.zeros(1,num_patches+self.batch_max_length,self.embed_dim))

        self.num_all_tokens = num_patches + self.batch_max_length


        attn_blocks = []
        for _ in range(depth):
            attn_blocks.append(AttentionBlock(embed_dim,embed_dim,num_heads,mlp_ratio,dropout))  
             
        self.attn_blocks = nn.Sequential(*attn_blocks)
        self.char_head = nn.Linear(self.embed_dim,num_class)

    def forward(self,x,is_eval=False):
        x = self.patch_embed(x)        
        x = torch.cat((self.char_tokens.repeat(x.shape[0],1,1),x),dim=1)
        x += self.pos_embed
        for attn in self.attn_blocks:
            x,weight = attn(x,x,x)
        chars = x[:,:self.batch_max_length,:]
        chars = self.char_head(chars)
        return chars

def PredatorTiny():
    config = {
        'embed_dim':768,
        'num_heads':12,
        'depth':12,
    }
    return Predator(**config)


if __name__  == '__main__':
    model = PredatorTiny()
    x = torch.randn((10,3,32,128))
    res = model(x)
    print(res.shape)        

