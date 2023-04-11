import torch
from torch import nn
from module.AttentionBlocks import VisualAttentionBlock as Visual
from module.AttentionBlocks import LinguisticAttentionBlock as Lingual
from module.AttentionBlocks import EyeAttentionBlock as Eye
from module.AttentionBlocks import TokenLearner, Transformer3Layers

class Decoder(nn.Module):
    def __init__(self,embed_dim,num_heads,num_classes,definition_string,supervised_blocks,random_mask=False,share_parameters=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.definition_string = definition_string
        self.num_blocks = len(definition_string)
        self.random_mask = random_mask
        self.share_parameters = share_parameters
        self.pre_norm = nn.LayerNorm(embed_dim)

        blocks = []
        shared_char_block = {}
        for (i,char) in enumerate(definition_string):
            assert char in ['V','L','E','T','3'], f'char is {char} instead of V, L, E, T or 3.'
            char_block_map = {
                'V':Visual,
                'L':Lingual,
                'E':Eye,
                'T':TokenLearner,
                '3':Transformer3Layers,
            }            
            if self.share_parameters:
                if char in shared_char_block.keys():
                    blocks.append(shared_char_block[char])
                else:
                    new_block = char_block_map[char](embed_dim,num_heads,True if i in supervised_blocks else False)
                    blocks.append(new_block)
                    shared_char_block[char] = new_block
            else:
                blocks.append(char_block_map[char](embed_dim,num_heads,True if i in supervised_blocks else False))

        self.blocks = nn.Sequential(*blocks)

        pe = nn.Parameter(torch.randn((1,27,embed_dim)))
        self.register_parameter('pe',pe)

        if self.share_parameters:
            self.v2l_proj = nn.Linear(embed_dim,embed_dim)
            self.l2v_proj = nn.Linear(embed_dim,embed_dim)

        self.visual_head = nn.Linear(embed_dim,num_classes)
        self.lingual_head = nn.Linear(embed_dim,num_classes)

    def forward(self,q,feat):
        res = torch.randn(0,device=q.device)
        for block in self.blocks:
            if self.random_mask:
                rand_int = torch.randint(q.shape[1],(1,))
                q[:,rand_int,:] = 0
            q = q + self.pe
            if type(block) in [Visual,TokenLearner]:
                (q,weight) = block(q,feat)
                head = self.visual_head
                if self.share_parameters:
                    proj = self.v2l_proj
            elif type(block) in [Lingual,Transformer3Layers]:
                (q,weight) = block(q)
                head = self.lingual_head
                if self.share_parameters:
                    proj = self.l2v_proj
            if block.supervised:
                res = torch.cat((res,head(q)),dim=1)
            if self.share_parameters:
                q = proj(q)
                #print('Here we share, there we cross.')

        return res
    def forward_with_weight(self,q,feat):
        res = torch.randn(0,device=q.device)
        for block in self.blocks:
            if self.random_mask:
                rand_int = torch.randint(q.shape[1],(1,))
                q[:,rand_int,:] = 0
            q = q + self.pe
            if type(block) in [Visual,TokenLearner]:
                (q,weight) = block(q,feat)
                head = self.visual_head
                if self.share_parameters:
                    proj = self.v2l_proj
            elif type(block) in [Lingual,Transformer3Layers]:
                (q,weight) = block(q)
                head = self.lingual_head
                if self.share_parameters:
                    proj = self.l2v_proj
            if block.supervised:
                res = torch.cat((res,head(q)),dim=1)
            if self.share_parameters:
                q = proj(q)
                #print('Here we share, there we cross.')

        return res,weight

if __name__ == '__main__':
    model = Decoder(192,3,38,'V3V3V',[0,1,2,3,4],random_mask=False,share_parameters=True)
    q = torch.randn((10,27,192))
    feat = torch.randn((10,256,192))
    res = model.forward_with_weight(q,feat)
    print(res[0].shape,res[1].shape)