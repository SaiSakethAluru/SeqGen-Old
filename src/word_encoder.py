import torch
import torch.nn as nn
from encoder_transformer_block import EncoderTransformerBlock

class WordEncoder(nn.Module):
    def __init__(
        self, 
        src_vocab_size, 
        embed_size, 
        num_layers, 
        heads, 
        device, 
        forward_expansion, 
        dropout, 
        max_length,
        labels
    ):
        super(WordEncoder,self).__init__()
        self.embed_size = embed_size
        self.device = device
        # Include glove here. 
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length,embed_size)
        self.layers = nn.ModuleList(
            [
                EncoderTransformerBlock(
                    embed_size,heads,dropout,forward_expansion,labels    
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N,seq_len = x.shape
        positions = torch.arange(0,seq_len).expand(N,seq_len).to(self.device)
        out = self.dropout(
            (self.word_embedding(x)+self.position_embedding(positions))
        )
        for layer in self.layers:
            out = layer(out,out,out,mask)
        
        return out
    