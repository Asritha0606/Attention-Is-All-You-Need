import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, value)
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None):
        residual = query
        batch_size = query.size(0)

        # Linear projections
        q = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)  # (batch, heads, seq_len, d_k)
        k = self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        v = self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)

        if mask is not None:
            # mask shape (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
            mask = mask.unsqueeze(1)

        # Apply attention
        out, attn = self.attention(q, k, v, mask=mask)

        out = out.transpose(1,2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)  # concat heads
        out = self.linear_out(out)
        out = self.dropout(out)
        out = self.layer_norm(out + residual)
        return out, attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.layer_norm(out + residual)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(num_heads, d_model, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, src, src_mask=None):
        out, _ = self.self_attn(src, src, src, mask=src_mask)
        out = self.ffn(out)
        return out

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(num_heads, d_model, dropout)
        self.enc_dec_attn = MultiHeadAttention(num_heads, d_model, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        out, _ = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        out, _ = self.enc_dec_attn(out, memory, memory, mask=memory_mask)
        out = self.ffn(out)
        return out

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, dropout=0.1, max_seq_len=5000):
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        self.pos_decoder = PositionalEncoding(d_model, max_seq_len)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_encoder_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_decoder_layers)]
        )

        self.linear_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_src_mask(self, src):
        # src shape (batch, src_len)
        return (src != 0).unsqueeze(-2)  # (batch, 1, src_len), assumes 0 is PAD

    def generate_tgt_mask(self, tgt):
        # tgt shape (batch, tgt_len)
        tgt_len = tgt.size(1)
        subsequent_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_mask = (tgt != 0).unsqueeze(-2)
        combined_mask = tgt_mask & subsequent_mask
        return combined_mask  # (batch, tgt_len, tgt_len)

    def forward(self, src, tgt):
        src_mask = self.generate_src_mask(src)
        tgt_mask = self.generate_tgt_mask(tgt)

        # Embedding + Positional encoding
        src_emb = self.dropout(self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model)))
        tgt_emb = self.dropout(self.pos_decoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model)))

        memory = src_emb
        for layer in self.encoder_layers:
            memory = layer(memory, src_mask)

        out = tgt_emb
        for layer in self.decoder_layers:
            out = layer(out, memory, tgt_mask, src_mask)

        output = self.linear_out(out)
        return output

# Example usage:
if __name__ == "__main__":
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    batch_size = 2
    src_len = 20
    tgt_len = 20

    model = Transformer(src_vocab_size, tgt_vocab_size)
    src = torch.randint(1, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))
    output = model(src, tgt)
    print("Output shape:", output.shape)  # Expected: (batch_size, tgt_len, tgt_vocab_size)

    print("Output:", output)
    print("Output:", output.argmax(dim=-1))  # Print predicted token indices