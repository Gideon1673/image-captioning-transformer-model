import unicodedata
import torch


text = "Học máy là gì"
text = unicodedata.normalize("NFC", text)  # ép về dạng composed
print(text)

tokens = ["Học", "máy", "là", "gì"]

chars = sorted(list(set(text)))  # thực tế: dùng vocab đã build từ toàn bộ corpus, không chỉ 1 câu
# chars.append('<bos>')
# chars.append('<eos>')
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)

def encode(s, stoi):
    return [stoi.get(c, stoi.get('<unk>', 0)) for c in s]

ids = encode(text, stoi)
# ids = [stoi['<bos>']] + ids + [stoi['<eos>']]
print(ids)

idx = torch.tensor(ids, dtype=torch.long)      # shape: (T,)
idx = idx.unsqueeze(0)                         # shape: (1, T) — batch_size=1

n_embd = 64  # kích thước embedding, giống config bạn dùng cho mini-GPT
token_embedding_table = torch.nn.Embedding(vocab_size, n_embd)

tok_emb = token_embedding_table(idx)  # shape: (1, T, n_embd)
T = idx.shape[1]
position_embedding_table = torch.nn.Embedding(T, n_embd)  # learned positional

pos = torch.arange(T, device=idx.device)  # shape: (T,)
pos_emb = position_embedding_table(pos)   # shape: (T, n_embd)

x = tok_emb + pos_emb  # broadcasting: (1,T,n_embd) + (T,n_embd) -> (1,T,n_embd)
print(x.shape)  # torch.Size([1, 13, 64])