import math
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import torch.optim as optim
import kagglehub

# Download dataset from Kaggle
path = kagglehub.dataset_download("devicharith/language-translation-englishfrench")
print("Path to dataset files:", path)

# Check if the dataset is a directory and load the CSV file
dataset_dir = path  # Use the path directly as it should point to the extracted files
csv_file = os.path.join(dataset_dir, "eng_-french.csv")


if not os.path.isfile(csv_file):
    raise FileNotFoundError(f"{csv_file} not found in extracted dataset directory.")

data = pd.read_csv(csv_file)

# We'll use only a subset for quick demo training
data = data[['English words/sentences', 'French words/sentences']].dropna().head(1000)

class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tokenizer, max_len=50):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        src_encoding = self.tokenizer.encode_plus(
            src_text, max_length=self.max_len, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        tgt_encoding = self.tokenizer.encode_plus(
            tgt_text, max_length=self.max_len, padding='max_length',
            truncation=True, return_tensors='pt'
        )

        src_ids = src_encoding['input_ids'].squeeze(0)  # remove batch dim
        tgt_ids = tgt_encoding['input_ids'].squeeze(0)

        return src_ids, tgt_ids

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)
    return src_batch, tgt_batch

# Transformer components (ScaledDotProductAttention, MultiHeadAttention, PositionwiseFeedForward, PositionalEncoding,
# EncoderLayer, DecoderLayer, Transformer) - as previously defined

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

        q = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        k = self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        v = self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        out, attn = self.attention(q, k, v, mask=mask)

        out = out.transpose(1,2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
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
        pe = pe.unsqueeze(0)
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
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, dropout=0.1, max_seq_len=100):
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
        return (src != 0).unsqueeze(-2)

    def generate_tgt_mask(self, tgt):
        tgt_len = tgt.size(1)
        subsequent_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_mask = (tgt != 0).unsqueeze(-2)
        combined_mask = tgt_mask & subsequent_mask
        return combined_mask

    def forward(self, src, tgt):
        src_mask = self.generate_src_mask(src)
        tgt_mask = self.generate_tgt_mask(tgt)

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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    src_texts = data['English words/sentences'].tolist()
    tgt_texts = data['French words/sentences'].tolist()

    dataset = TranslationDataset(src_texts, tgt_texts, tokenizer, max_len=50)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    src_vocab_size = tokenizer.vocab_size
    tgt_vocab_size = tokenizer.vocab_size

    model = Transformer(src_vocab_size, tgt_vocab_size, max_seq_len=50).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    epochs = 10
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for src_batch, tgt_batch in dataloader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]

            optimizer.zero_grad()
            output = model(src_batch, tgt_input)

            output = output.contiguous().view(-1, tgt_vocab_size)
            tgt_output = tgt_output.contiguous().view(-1)

            loss = criterion(output, tgt_output)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

    # Inference example
    model.eval()
    test_sentence = "hello"
    test_tokens = torch.tensor(tokenizer.encode(test_sentence, add_special_tokens=True)).unsqueeze(0).to(device)
    generated = [tokenizer.cls_token_id]  # Start with [CLS] token

    max_len = 20
    for _ in range(max_len):
        tgt_input = torch.tensor(generated).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(test_tokens, tgt_input)
            next_token = output[:, -1, :].argmax(dim=-1).item()
            generated.append(next_token)
            if next_token == tokenizer.sep_token_id:
                break

    print("Generated token IDs:", generated)
    translated_text = tokenizer.decode(generated, skip_special_tokens=True)
    print("Translated text:", translated_text)

if __name__ == "__main__":
    main()
