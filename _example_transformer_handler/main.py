import torch
import torch.nn as nn
import torch.nn.functional as F

with open('truyen_kieu_data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Tổng số ký tự: {len(text)}")
print(f"200 ký tự đầu:\n{text[:200]}")

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size}")
print(f"Các ký tự: {''.join(chars)}")

# Ánh xạ ký tự <-> số nguyên
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [char_to_idx[c] for c in s]
decode = lambda l: ''.join([idx_to_char[i] for i in l])

# Test thử
print(encode("Trăm năm"))
print(decode(encode("Trăm năm")))

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Train: {len(train_data)} ký tự, Val: {len(val_data)} ký tự")

torch.manual_seed(1337)
batch_size = 16
block_size = 32  # độ dài ngữ cảnh


def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x, y


xb, yb = get_batch('train')
print("Input (x):", decode(xb[0].tolist()))
print("Target (y):", decode(yb[0].tolist()))

n_embd = 32
n_head = 2
head_size = n_embd // n_head  # 16
dropout = 0.1


class Head(nn.Module):
    """Một đầu Self-Attention — tính Q, K, V và trả về Context Vector"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)  # W_K
        self.query = nn.Linear(n_embd, head_size, bias=False)  # W_Q
        self.value = nn.Linear(n_embd, head_size, bias=False)  # W_V
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # Batch, Time (độ dài chuỗi), Channels (n_embd)

        k = self.key(x)  # (B, T, head_size)  <- đúng bước "K_tạo = E[tạo] × W_K" đã học
        q = self.query(x)  # (B, T, head_size)  <- đúng bước "Q_tạo = E[tạo] × W_Q"
        v = self.value(x)  # (B, T, head_size)  <- đúng bước "V_tạo = E[tạo] × W_V"

        # Bước "Dot Product" (bước 2 lý thuyết) + Scale (bước 3)
        wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)  # (B, T, T)

        # Che các từ TƯƠNG LAI (không cho nhìn trước) — vì đây là mô hình sinh văn bản
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # Bước Softmax (bước 3 lý thuyết)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Bước Weighted Sum (bước 4 lý thuyết)
        out = wei @ v  # (B, T, head_size)
        return out, wei  # trả thêm wei để sau này visualize


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        outs_and_weis = [h(x) for h in self.heads]
        outs = [o for o, w in outs_and_weis]
        weis = [w for o, w in outs_and_weis]
        out = torch.cat(outs, dim=-1)
        out = self.dropout(self.proj(out))
        return out, weis


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        sa_out, weis = self.sa(self.ln1(x))
        x = x + sa_out          # residual connection
        x = x + self.ffwd(self.ln2(x))
        return x, weis

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)                          # (B,T,n_embd)
        pos_emb = self.position_embedding(torch.arange(T))            # (T,n_embd)
        x = tok_emb + pos_emb

        all_weis = []
        for block in self.blocks:
            x, weis = block(x)
            all_weis.append(weis)

        x = self.ln_f(x)
        logits = self.lm_head(x)   # (B,T,vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

        return logits, loss, all_weis