import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(CrossAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        # self.head_dim = embed_size // heads
        self.head_dim = embed_size

        # assert (
        #     self.head_dim * heads == embed_size
        # ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        #Inputs - N,seq_len,embed_size
        N = query.shape[0]
        # print('crossatt values.shape',values.shape)
        # print('crossatt keys.shape',keys.shape)
        # print('crossatt query.shape',query.shape)
        # print('crossatt mask.shape',mask.shape)
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        value_seq_len, key_seq_len = values.shape[2], keys.shape[2]
        # Split the embedding into self.heads different pieces

        # values = values.reshape(N, value_len, self.heads, self.head_dim)
        # keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        # query = query.reshape(N, query_len, self.heads, self.head_dim)
        ## Instead of dividing into parts for each head, we repeat the same thing. 
        ## DOUBT: Is this needed though?
        values = values.unsqueeze(3).expand(N,value_len,value_seq_len, self.heads,self.head_dim)
        keys = keys.unsqueeze(3).expand(N,key_len,key_seq_len, self.heads,self.head_dim)
        query = query.unsqueeze(2).expand(N,query_len,self.heads,self.head_dim)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqhd,nkshd->nhqks", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)
        # print('crossatt energy.shape',energy.shape)
        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhqls,nlshd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.
        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)
        return out