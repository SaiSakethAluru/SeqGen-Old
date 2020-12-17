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
        ## TODO: Make this pretrained from glove
        self.word_embedding = nn.Embedding(max_length,embed_size)   # Needed to get the label embeddings
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
        positions = torch.arange(0,par_len).expand(N,par_len).to(self.device)
        word_level_outputs = []
        for i,sent in enumerate(x):
            word_level_outputs.append(
                self.word_level_encoder(
                    sent.reshape(N,seq_len), mask[i].reshape(N,seq_len)
                )
            )
        # NOTE: shape of each output tensor here should be N,embed_size for each entry of the above list
        # NOTE: After stacking it should be N,par_len, embed_size if everything works fine. Else adjust it.
        word_level_outputs = torch.stack(word_level_outputs,dim=1)
        out = self.dropout(
            (word_level_outputs + self.position_embedding(positions))
        )
        label_embed = [
            self.word_embedding(label) for label in self.labels
        ]
        # NOTE: Each entry in the above list should be 1,embed_size. If not adjust to this size
        label_embed = torch.cat(label_embed,dim=0)
        for layer in self.layers:
            out = layer(out,out,out,mask)


