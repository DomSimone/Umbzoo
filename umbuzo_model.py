"""
Umbuzo LLM: A GPT-Style Transformer Architecture built from scratch.
Configured for ~2 Billion parameters with 120 layers as requested.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class UmbuzoAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.resid_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(torch.tril(torch.ones(T, T, device=x.device)) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        return y

class UmbuzoMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class UmbuzoBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size)
        self.attn = UmbuzoAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size)
        self.mlp = UmbuzoMLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class UmbuzoLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.hidden_size),
            wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size),
            drop = nn.Dropout(config.hidden_dropout_prob),
            h = nn.ModuleList([UmbuzoBlock(config) for _ in range(config.num_hidden_layers)]),
            ln_f = nn.LayerNorm(config.hidden_size),
        ))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Init weights
        self.apply(self._init_weights)
        print(f"Umbuzo Model Initialized with {self.get_num_params():,} parameters.")

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, attention_mask=None):
        device = input_ids.device
        b, t = input_ids.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(input_ids)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            # Gradient checkpointing can be added here during training loop
            x = block(x)
            
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        return logits, loss

if __name__ == "__main__":
    from config import model_cfg
    # Update to your specific request for parameter count verification
    model_cfg.num_hidden_layers = 120
    model_cfg.hidden_size = 1024 # Scaled down for local CPU summary if needed, will use config for GPU
    
    model = UmbuzoLLM(model_cfg)
    print("Model Summary:")
    print(model)
