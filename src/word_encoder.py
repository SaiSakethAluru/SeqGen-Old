import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import csv
from src.encoder_transformer_block import EncoderTransformerBlock

class WordEncoder(nn.Module):
    def __init__(
        self, 
        label_list,
        embed_size, 
        num_layers, 
        heads, 
        device, 
        forward_expansion, 
        dropout, 
        max_length,
        embed_path = '../data/glove.6B.100d.txt'
    ):
        super(WordEncoder,self).__init__()
        self.device = device
        self.labels = label_list
        ## TODO: Make this pretrained from glove
        embeds = pd.read_csv(filepath_or_buffer=embed_path,header=None,sep=' ',quoting=csv.QUOTE_NONE).values[:,1:]
        src_vocab_size,embed_size = embeds.shape
        self.embed_size = embed_size
        src_vocab_size += 3
        unknown_word = np.zeros((1, embed_size))
        pad_word = np.zeros((1,embed_size))
        sos_word = np.zeros((1,embed_size))
        embeds = torch.from_numpy(np.concatenate([unknown_word,pad_word,sos_word,embeds], axis=0).astype(np.float)).to(device)

        self.word_embedding = nn.Embedding(src_vocab_size, embed_size).from_pretrained(embeds)
        self.position_embedding = nn.Embedding(max_length,embed_size)
        self.layers = nn.ModuleList(
            [
                EncoderTransformerBlock(
                    embed_size,heads,dropout,forward_expansion,label_list    
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # print('word x.shape',x.shape)
        # print('word mask.shape',mask.shape)
        N,seq_len = x.shape
        positions = torch.arange(0,seq_len).expand(N,seq_len).to(self.device)
        # print('word positions.shape',positions.shape)
        # out - N,seq_len,embed_size
        out = self.dropout(
            (self.word_embedding(x)+self.position_embedding(positions))
        )
        # print('word out.shape',out.shape)
        label_embed = [
            self.word_embedding(label) for label in self.labels
        ]
        # print('word label_embed[0].shape',label_embed[0].shape)
        # NOTE: Each entry in the above list should be 1,embed_size. If not adjust to this size
        # label_embed - N,num_labels,embed_size
        # label_embed = torch.cat(label_embed,dim=0)
        label_embed = torch.stack(label_embed,dim=0)
        label_embed = label_embed.repeat(N,1,1)
        # print('word label_embed.shape',label_embed.shape)
        for layer in self.layers:
            out = layer(out,out,out,label_embed,mask)
        # out - N, seq_len, embed_size
        # print('word out.shape',out.shape)
        return out
    