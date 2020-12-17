import torch
import torch.nn as nn
from encoder_transformer_block import EncoderTransformerBlock
from word_encoder import WordEncoder
class SentenceEncoder(nn.Module):
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
        super(SentenceEncoder,self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_level_encoder = WordEncoder(
            src_vocab_size, 
            embed_size, 
            num_layers, 
            heads, 
            device, 
            forward_expansion, 
            dropout, 
            max_length,
            labels
        )
        self.position_embedding = nn.Embedding(max_length,embed_size)
        self.layers = nn.ModuleList(
            [
                EncoderTransformerBlock(
                     embed_size, heads, dropout, forward_expansion, labels
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,mask):
        N,par_len,seq_len = x.shape
        positions = torch.arange(0,par_len).expand(N,par_len)


