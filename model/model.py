import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class GPT2Config:
    def __init__(self):
        self.vocab_size = 50257
        self.n_positions = 1024
        self.n_ctx = 1024
        self.n_embd = 768
        self.n_layer = 12
        self.n_head = 12
        self.resid_pdrop = 0.1
        self.embd_pdrop = 0.1
        self.attn_pdrop = 0.1
        self.layer_norm_epsilon = 1e-5
        self.initializer_range = 0.
        self.output_attentions = False 
class GPT2Model(nn.Module):
    def __init__(self, config):
        super(GPT2Model, self).__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
    def forward(self, input_ids, position_ids=None, past=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        position_ids = position_ids.view(-1, input_shape[-1])
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if inputs_embeds.size() != position_embeds.size():
            raise ValueError("the embeddings of inputs and position are not the same size")
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        presents = ()
        for block, past in zip(self.h, past):
            hidden_states, present = block(hidden_states, past=past)
            presents = presents + (present,)
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents
        
class GPT2Block(nn.Module):
    def __init__(self, config):
        super(GPT2Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)
    def forward(self, x, past):
        a, present = self.attn(self.ln_1(x), past=past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present
        
class GPT2Attention(nn.Module):
    def __init__(self, config):
        super(GPT2Attention, self).__init__()
        self.output_attentions = config.output_attentions
        self.n_head = config.n_head
        self.split_size = config.n_embd
        self.scale = self.split_size ** -0.5
        self.c_attn = Conv1D(3 * self.split_size, self.split_size)
        self.c_proj = Conv1D(self.split_size, self.split_size)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.bias = nn.Parameter(torch.zeros(1, 1, 1024, 1024))
    def _attn(self, q, k, v): 
        w = torch.matmul(q, k) 
        if self.scale: 
            w = w / math.sqrt(v.size(-1)) 
        w = w.softmax(dim=-1) 
        w = self.attn_dropout(w) 
        return torch.matmul(w, v) 
    def merge_heads(self, x): 
        x = x.permute(0, 2, 1, 3).contiguous() 
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),) 
        return x.view(*new_x_shape) 
    def split_heads(self, x, k=False): 
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head) 
        x = x.view(*new_x_shape) 
        if k: 
            return x.permute(0, 2, 3, 1) 
        else: 
            return x.permute(0, 2, 1, 3)
    def forward(self, x, past):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if past is not None:
            past_key, past_value = past
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-1)
        present = (key, value)
        attn_output = self._attn(query, key, value)
        attn_output = self.merge_heads(attn_output)
        attn_output = self.c_proj(attn_output)
        attn_output += self.bias
        attn_output = self.resid_dropout(attn_output)
        return attn_output, present 
class GPT2MLP(nn.Module):
    def __init__(self, config):
        super(GPT2MLP, self).__init__()
        self.c_fc = Conv1D(4 * config.n_embd, config.n_embd)
        self.c_proj = Conv1D(config.n_embd, 4 * config.n_embd)
        self.act = F.gelu
        self.dropout = nn.Dropout(config.resid_pdrop)
    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)
        
class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))
    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x
        
