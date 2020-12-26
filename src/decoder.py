import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import csv
from decoder_block import DecoderBlock


class Decoder(nn.Module):
    def __init__(
            self,
            label_list,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_par_len,
            max_seq_len,
            embed_path = '../data/glove.6B.100d.txt'
    ):
        super(Decoder, self).__init__()
        self.device = device
        # TODO: add glove here
        embeds = pd.read_csv(filepath_or_buffer=embed_path,header=None,sep=' ',quoting=csv.QUOTE_NONE).values[:,1:]
        src_vocab_size,embed_size = embeds.shape
        self.embed_size = embed_size
        src_vocab_size += 3
        unknown_word = np.zeros((1, embed_size))
        pad_word = np.zeros((1,embed_size))
        sos_word = np.zeros((1,embed_size))
        embeds = torch.from_numpy(np.concatenate([unknown_word,pad_word,sos_word,embeds], axis=0).astype(np.float))
        # words = [word[0] for word in words]
        # label_embeds = 
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size).from_pretrained(embeds)
        self.position_embedding = nn.Embedding(max_par_len, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, len(label_list))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, enc_word_out, src_mask, word_level_mask, trg_mask):
        print("decoder x.shape",x.shape)
        print('decoder enc_out.shape',enc_out.shape)
        print('decoder enc_word_out.shape',enc_word_out.shape)
        print('decoder src_mask.shape',src_mask.shape)
        print('decoder trg_mask.shape',trg_mask.shape)
        N, par_len = x.shape
        # x - N,par_len
        positions = torch.arange(0, par_len).expand(N, par_len).to(self.device)
        print('decoder positions.shape',positions.shape)
        # positions - N,par_len
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))
        print('decoder x.shape',x.shape)
        # x - N,par_len, embed_size
        for layer in self.layers:
            # enc_out - N,par_len,embed_size  -> is this right? 
            # enc_word_out - N,par_len,embed_size
            x = layer(x, enc_out, enc_out, enc_word_out, src_mask, word_level_mask, trg_mask)
            print('decoder x.shape',x.shape)

        out = self.fc_out(x)
        print('decoder out.shape',out.shape)
        # Expected shape of out 
        return out
