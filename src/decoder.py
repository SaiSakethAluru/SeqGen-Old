import torch
import torch.nn as nn
from decoder_block import DecoderBlock


class Decoder(nn.Module):
    def __init__(
            self,
            tgt_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
    ):
        super(Decoder, self).__init__()
        self.device = device
        # TODO: add glove here
        self.word_embedding = nn.Embedding(tgt_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, enc_word_out, src_mask, tgt_mask):
        N, par_len = x.shape
        positions = torch.arange(0, par_len).expand(N, par_len).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, enc_word_out, src_mask, tgt_mask)

        out = self.fc_out(x)

        return out
