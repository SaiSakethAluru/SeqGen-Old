import torch
import torch.nn as nn
from selfatt import SelfAttention


class DecoderTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(DecoderTransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.enc_sent_attention = SelfAttention(embed_size, heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, encoder_word_level_output, mask):
        # encoder_word_level should be N,embed_size
        # value,key,query should be?
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        # include word level outputs of encoder here
        encoder_word_attentions = self.enc_sent_attention(encoder_word_level_output, encoder_word_level_output, x, mask)
        x = self.dropout(self.norm3(x + encoder_word_attentions))

        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
