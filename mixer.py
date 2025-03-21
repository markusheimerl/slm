#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import requests
import os
import numpy as np
import time
import math
import random

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim) / math.sqrt(embedding_dim))
    
    def forward(self, x):
        return F.embedding(x, self.weight)

def download_gutenberg(url, save_path):
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    
    response = requests.get(url)
    text = response.text
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    return text

def get_combined_corpus():
    if not os.path.exists('gutenberg_texts'):
        os.makedirs('gutenberg_texts')
    
    books = [
        ("https://www.gutenberg.org/files/1342/1342-0.txt", "pride_and_prejudice.txt"),
        ("https://www.gutenberg.org/files/84/84-0.txt", "frankenstein.txt"),
        ("https://www.gutenberg.org/files/1661/1661-0.txt", "sherlock_holmes.txt"),
        ("https://www.gutenberg.org/files/2701/2701-0.txt", "moby_dick.txt"),
        ("https://www.gutenberg.org/files/98/98-0.txt", "tale_of_two_cities.txt"),
        ("https://www.gutenberg.org/files/1400/1400-0.txt", "great_expectations.txt"),
        ("https://www.gutenberg.org/files/345/345-0.txt", "dracula.txt"),
        ("https://www.gutenberg.org/files/174/174-0.txt", "dorian_gray.txt"),
        ("https://www.gutenberg.org/files/16/16-0.txt", "peter_pan.txt"),
        ("https://www.gutenberg.org/files/768/768-0.txt", "wuthering_heights.txt"),
        ("https://www.gutenberg.org/files/45/45-0.txt", "anne_of_green_gables.txt"),
        ("https://www.gutenberg.org/files/1260/1260-0.txt", "jane_eyre.txt"),
    ]
    
    combined_text = ""
    for url, filename in books:
        print(f"Processing {filename}...")
        filepath = os.path.join('gutenberg_texts', filename)
        text = download_gutenberg(url, filepath)
        combined_text += text + "\n\n"
    
    print(f"Combined corpus size: {len(combined_text)} characters")
    with open('gutenberg_texts/combined_corpus.txt', 'w', encoding='utf-8') as f:
        f.write(combined_text)
    
    return combined_text

class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256
        print(f"Using byte-level tokenization with vocab size: {self.vocab_size}")
    
    def encode(self, text):
        return [b for b in text.encode('utf-8')]
    
    def decode(self, indices):
        try:
            return bytes(indices).decode('utf-8', errors='replace')
        except:
            return ''.join([chr(min(i, 127)) for i in indices])

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_length=64):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.data = self.tokenizer.encode(text)
        self.text = text
    
    def __len__(self):
        return max(0, len(self.data) - self.seq_length - 1)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_length], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.seq_length+1], dtype=torch.long)
        return x, y
    
    def get_random_text_snippet(self, char_length):
        if len(self.text) <= char_length:
            return self.text
        start_idx = random.randint(0, len(self.text) - char_length - 1)
        return self.text[start_idx:start_idx + char_length]

class CausalTokenMixingMLP(nn.Module):
    def __init__(self, seq_length):
        super().__init__()
        self.weight1 = nn.Parameter(torch.randn(seq_length, seq_length) / math.sqrt(seq_length))
        self.weight2 = nn.Parameter(torch.randn(seq_length, seq_length) / math.sqrt(seq_length))
        mask = torch.tril(torch.ones(seq_length, seq_length))
        self.register_buffer('mask', mask)
    
    def forward(self, x):
        masked_weight1 = self.weight1 * self.mask
        x = F.linear(x, masked_weight1)
        x = x * torch.sigmoid(x)
        masked_weight2 = self.weight2 * self.mask
        x = F.linear(x, masked_weight2)
        return x

class ChannelMixingMLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.weight1 = nn.Parameter(torch.randn(hidden_dim, embed_dim) / math.sqrt(embed_dim))
        self.weight2 = nn.Parameter(torch.randn(embed_dim, hidden_dim) / math.sqrt(hidden_dim))
    
    def forward(self, x):
        x = F.linear(x, self.weight1)
        x = x * torch.sigmoid(x)
        x = F.linear(x, self.weight2)
        return x

class MixerBlock(nn.Module):
    def __init__(self, embed_dim, seq_length, hidden_dim_channel):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        
        self.token_mixing = CausalTokenMixingMLP(seq_length)
        self.channel_mixing = ChannelMixingMLP(embed_dim, hidden_dim_channel)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Token mixing
        residual = x
        x = x.transpose(1, 2)
        x_flat = x.reshape(batch_size * self.embed_dim, self.seq_length)
        x_mixed = self.token_mixing(x_flat)
        x = x_mixed.reshape(batch_size, self.embed_dim, self.seq_length)
        x = x.transpose(1, 2)
        x = x + residual
        
        # Channel mixing
        residual = x
        x_flat = x.reshape(batch_size * self.seq_length, self.embed_dim)
        x_mixed = self.channel_mixing(x_flat)
        x = x_mixed.reshape(batch_size, self.seq_length, self.embed_dim)
        x = x + residual
        
        return x

class MixerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=4, seq_length=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        
        self.embedding = Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([
            MixerBlock(embed_dim=embed_dim, seq_length=seq_length, hidden_dim_channel=hidden_dim)
            for _ in range(num_layers)
        ])
        self.out_proj_weight = nn.Parameter(torch.randn(vocab_size, embed_dim) / math.sqrt(embed_dim))
    
    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return F.linear(x, self.out_proj_weight)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    return total_params

def generate_text(model, tokenizer, seed_text, max_length=100, device='cpu', temperature=0.7):
    model.eval()
    tokens = tokenizer.encode(seed_text)[-model.seq_length:]
    if len(tokens) < model.seq_length:
        tokens = [0] * (model.seq_length - len(tokens)) + tokens
    generated = list(tokens)
    
    with torch.no_grad():
        for _ in range(max_length):
            x = torch.tensor([tokens], dtype=torch.long).to(device)
            output = model(x)
            next_token_logits = output[0, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=0)
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)
            tokens = tokens[1:] + [next_token]
    
    return tokenizer.decode(generated)

def train(model, dataloader, optimizer, tokenizer, device, epochs=5, generate_every=200, seed_texts=None):
    if seed_texts is None:
        seed_texts = ["it is a truth universally acknowledged",
                      "in the beginning",
                      "once upon a time"]
    
    model.train()
    start_time = time.time()
    total_batches = 0
    
    for epoch in range(epochs):
        total_loss = 0
        epoch_batches = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            output = output.view(-1, output.size(-1))
            target = target.view(-1)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
            epoch_batches += 1
            
            if batch_idx % 50 == 0:
                elapsed = time.time() - start_time
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}, Time: {elapsed:.2f}s')
            
            if total_batches % generate_every == 0:
                print(f"\n--- GENERATING TEXT (batch {total_batches}) ---")
                model.eval()
                for seed in seed_texts:
                    generated = generate_text(model, tokenizer, seed, max_length=100, device=device)
                    print(f"\nSeed: \"{seed}\"")
                    print(f"Generated: \"{generated}\"\n")
                model.train()
                print(f"--- RESUMING TRAINING ---\n")
        
        avg_loss = total_loss / epoch_batches
        print(f'Epoch: {epoch+1}, Average Loss: {avg_loss:.4f}')
    
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    text = get_combined_corpus()
    tokenizer = ByteTokenizer()
    
    seq_length = 1024
    dataset = TextDataset(text, tokenizer, seq_length=seq_length)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = MixerModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=16,
        hidden_dim=512,
        num_layers=4,
        seq_length=seq_length
    ).to(device)
    
    count_parameters(model)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    
    seed_texts = [dataset.get_random_text_snippet(seq_length) for _ in range(3)]
    print("Random seed texts for generation:")
    for i, seed in enumerate(seed_texts):
        print(f"{i+1}. {seed}")
    
    train(
        model,
        dataloader,
        optimizer,
        tokenizer,
        device,
        epochs=3,
        generate_every=200,
        seed_texts=seed_texts
    )
    
if __name__ == "__main__":
    main()