import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

class SimpleLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
    def forward(self, input_ids, attention_mask=None):
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask)
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        return lm_logits
    
    def count_non_embedding_params(self):
        core_params = self.transformer.h.parameters()
        lm_head_params = self.lm_head.parameters()    
        
        total_core_params = sum(p.numel() for p in core_params)
        total_lm_head_params = sum(p.numel() for p in lm_head_params)

        return total_core_params + total_lm_head_params