import torch
from torch import nn
from module.PatchEmbeds import SVTRPatchEmbed
from module.AttentionBlocks import AttentionBlock
from module.SEAttentions import SqueezeExcitation



class Finder(nn.Module):
    def __init__(self,embed_dim,num_heads,depth,decoder_dim,num_class=38,batch_max_length=27,mlp_ratio=4,dropout=0.1,patch_size=(4,4),img_size=(32,128)):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.decoder_dim = decoder_dim
        self.num_class = num_class
        self.batch_max_length = batch_max_length
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.patch_size = patch_size
        self.img_size = img_size

        self.patch_embed = SVTRPatchEmbed(3,self.embed_dim)
        num_patch_h = img_size[0]//patch_size[0]
        num_patch_w = img_size[1]//patch_size[1]
        num_patches = num_patch_h * num_patch_w
        
        self.char_tokens = nn.Parameter(torch.zeros((1,self.batch_max_length,self.decoder_dim)))
        self.pos_embed = nn.Parameter(torch.zeros(1,num_patches,self.embed_dim))


        attn_blocks = []
        for _ in range(depth):
            attn_blocks.append(AttentionBlock(embed_dim,num_heads,False,mlp_ratio,dropout))          
        
            
        self.attn_blocks = nn.Sequential(*attn_blocks)

        self.se = SqueezeExcitation(num_patches)

        self.feat_proj = nn.Linear(self.embed_dim,self.decoder_dim)

        self.character_finder = AttentionBlock(self.decoder_dim,num_heads,False,mlp_ratio,dropout)        

        self.char_head = nn.Linear(self.decoder_dim,self.num_class)



    def forward(self,x):
        x = self.patch_embed(x)        

        x += self.pos_embed
        for attn in self.attn_blocks:   
            x,weight = attn(x,x,x)

        x = self.se(x)

        x = self.feat_proj(x)

        x,_ = self.character_finder(self.char_tokens.repeat(x.shape[0],1,1),x,x)
        
        x = self.char_head(x)

        return x

def FinderTiny():
    config = {
        'embed_dim':192,
        'num_heads':3,
        'depth':12,
        'decoder_dim':192*4,

    }
    return Finder(**config)


if __name__  == '__main__':
    model = FinderTiny()
    x = torch.randn((10,3,32,128))
    res = model(x)
    print(res.shape)        

