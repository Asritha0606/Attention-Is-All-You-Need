import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Data preprocessing helpers
# -------------------------

class Vocab:
    def __init__(self, tokens):
        self.special_tokens = ['<pad>', '<sos>', '<eos>']
        self.tokens = self.special_tokens + sorted(set(tokens))
        self.stoi = {tok: i for i, tok in enumerate(self.tokens)}
        self.itos = {i: tok for i, tok in enumerate(self.tokens)}
    
    def __len__(self):
        return len(self.tokens)

    def encode(self, text):
        return [self.stoi['<sos>']] + [self.stoi[t] for t in text.split()] + [self.stoi['<eos>']]

    def decode(self, ids):
        # Remove special tokens for output clarity
        tokens = [self.itos[i] for i in ids if i not in [self.stoi['<pad>'], self.stoi['<sos>'], self.stoi['<eos>']]]
        return ' '.join(tokens)


# Toy English-French parallel sentences
eng_sentences = [
    "hello",
    "how are you",
    "good morning",
    "thank you",
    "yes",
    "no"
]

fr_sentences = [
    "bonjour",
    "comment ça va",
    "bon matin",
    "merci",
    "oui",
    "non"
]

# Build vocabularies
eng_vocab = Vocab(" ".join(eng_sentences).split())
fr_vocab = Vocab(" ".join(fr_sentences).split())

print(f"English vocab size: {len(eng_vocab)}")
print(f"French vocab size: {len(fr_vocab)}")

# Max sentence length for padding
MAX_LEN = 6

def pad_sequence(seq, max_len, pad_value):
    seq = seq[:max_len]
    seq += [pad_value] * (max_len - len(seq))
    return seq

# Dataset class
class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab, max_len=MAX_LEN):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        self.data = []
        for src, tgt in zip(src_texts, tgt_texts):
            src_ids = pad_sequence(src_vocab.encode(src), max_len, src_vocab.stoi['<pad>'])
            tgt_ids = pad_sequence(tgt_vocab.encode(tgt), max_len, tgt_vocab.stoi['<pad>'])
            # Input target is shifted right by one: decoder input
            tgt_input = [tgt_vocab.stoi['<sos>']] + tgt_ids[:-1]
            self.data.append((torch.tensor(src_ids), torch.tensor(tgt_input), torch.tensor(tgt_ids)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# -------------------------
# Transformer model code
# -------------------------

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    return attn @ V, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model):
        super().__init__()
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)


    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        Tq = q.size(1)
        Tk = k.size(1)
        Tv = v.size(1)

        Q = self.q_linear(q).view(B, Tq, self.heads, self.d_k).transpose(1, 2)  # (B, heads, Tq, d_k)
        K = self.k_linear(k).view(B, Tk, self.heads, self.d_k).transpose(1, 2)  # (B, heads, Tk, d_k)
        V = self.v_linear(v).view(B, Tv, self.heads, self.d_k).transpose(1, 2)  # (B, heads, Tv, d_k)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (B, heads, Tq, Tk)
        if mask is not None:
        # Expand mask to (B, heads, Tq, Tk) shape
            if mask.dim() == 4:
                scores = scores.masked_fill(mask == 0, float('-1e9'))
            elif mask.dim() == 3:
                scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-1e9'))

        
        attn = torch.softmax(scores, dim=-1)
        out = attn @ V  # (B, heads, Tq, d_k)
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.heads * self.d_k)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_out = self.attn(x, x, x, mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(heads, d_model)
        self.cross_attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x2 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + x2)
        x2 = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + x2)
        x2 = self.ff(x)
        x = self.norm3(x + x2)
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, d_ff) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.embed(x)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, d_ff) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.embed(x)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=64, N=2, heads=4, d_ff=256):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, N, heads, d_ff)
        self.decoder = Decoder(tgt_vocab_size, d_model, N, heads, d_ff)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return self.fc_out(dec_out)

    
    def generate(self, src, max_len=20, start_symbol=0):
        self.eval()
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # updated mask shape
        enc_out = self.encoder(src, src_mask)
        ys = torch.ones(src.size(0), 1).fill_(start_symbol).long().to(src.device)

        for _ in range(max_len - 1):
            tgt_mask = torch.tril(torch.ones((ys.size(1), ys.size(1)), device=src.device)).bool()
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1)

            out = self.decoder(ys, enc_out, src_mask, tgt_mask)
            out = self.fc_out(out[:, -1])
            next_word = out.argmax(dim=-1).unsqueeze(1)
            ys = torch.cat([ys, next_word], dim=1)

            if next_word.item() == fr_vocab.stoi['<eos>']:
                break

        return ys



# -------------------------
# Training setup
# -------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

dataset = TranslationDataset(eng_sentences, fr_sentences, eng_vocab, fr_vocab)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = Transformer(len(eng_vocab), len(fr_vocab)).to(device)


def create_src_mask(src, pad_idx):
    # (B, 1, 1, S) -> broadcastable for multi-head attention
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)  # shape: (B, 1, 1, S)

def create_tgt_mask(tgt):
    B, T = tgt.shape
    mask = torch.tril(torch.ones(T, T)).bool().to(tgt.device)  # shape: (T, T)
    return mask.unsqueeze(0).unsqueeze(1)  # shape: (1, 1, T, T)

# In training loop:
pad_idx_src = eng_vocab.stoi['<pad>']
pad_idx_tgt = fr_vocab.stoi['<pad>']

EPOCHS = 100
optimizer = optim.Adam(model.parameters(), lr=1e-4)
pad_idx = fr_vocab.stoi['<pad>']
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for src, tgt_inp, tgt_label in dataloader:
        src, tgt_inp, tgt_label = src.to(device), tgt_inp.to(device), tgt_label.to(device)
        src_mask = create_src_mask(src, pad_idx_src).to(device)
        tgt_mask = create_tgt_mask(tgt_inp).to(device)

        optimizer.zero_grad()
        output = model(src, tgt_inp, src_mask=src_mask, tgt_mask=tgt_mask)  # pass src_mask!

        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Output contains NaN or Inf, stopping")
            break

        loss = loss_fn(output.view(-1, output.size(-1)), tgt_label.view(-1))
        if torch.isnan(loss) or torch.isinf(loss):
            print("Loss is NaN or Inf, stopping")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")



# -------------------------
# Inference example
# -------------------------

model.eval()
with torch.no_grad():
    for i, (src, tgt_inp, tgt_label) in enumerate(dataset):
        src = src.unsqueeze(0).to(device)  # batch=1
        generated_ids = model.generate(src, max_len=MAX_LEN, start_symbol=fr_vocab.stoi['<sos>'])
        src_text = eng_vocab.decode(src[0].cpu().tolist())
        pred_text = fr_vocab.decode(generated_ids[0].cpu().tolist())
        print(f"Source (EN): {src_text}")
        print(f"Predicted (FR): {pred_text}")
        if i == 5:  # show all 6
            break
