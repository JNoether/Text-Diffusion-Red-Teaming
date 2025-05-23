from transformers import AutoTokenizer, AutoModel, T5EncoderModel
from torch import nn
import torch.nn.functional as F
import torch
from torch.distributions.normal import Normal
import math

class DiffusionModel(nn.Module):
    def __init__(self, model, out_size, cache_dir=None) -> None:
        super().__init__()
        self.out_size = out_size
        self.transformer = T5EncoderModel.from_pretrained(model, cache_dir=cache_dir)
        # forward pass to get size of 
        self.d_model = self.transformer.config.d_model + 768
        self.mean_head = nn.Sequential(
            nn.Linear(self.d_model , self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model , out_size)
        )
        # self.std_head = nn.Sequential(
        #     nn.Linear(d_model , d_model),
        #     nn.ReLU(),
        #     nn.Linear(d_model, out_size),
        #     # nn.Softplus()
        # )
        self.std_head = nn.Parameter(torch.ones(1, out_size), requires_grad=False)
        self.value_head = nn.Sequential(
            nn.Linear(self.transformer.config.d_model, self.transformer.config.d_model),
            nn.ReLU(),
            nn.Linear(self.transformer.config.d_model, 1)
        )

    def forward(self, input_ids, attention_mask, embedding):
        batch_size = input_ids.shape[0]
        hidden_embs = self.transformer(input_ids, attention_mask).last_hidden_state
        hidden_embs = hidden_embs.view(batch_size, -1, self.d_model-768)[:, -1, :]
        hidden_embs = torch.concat((hidden_embs, embedding), dim=-1)
        mean = self.mean_head(hidden_embs)
        mean = mean / mean.shape[-1]**0.5
        std = self.std_head
        std = torch.exp(std + 1e-6)

        if std.isnan().any():
            std = torch.zeros(std.shape)
        std = torch.max(std, torch.ones(std.shape, device=mean.device) * 1e-2)
        return mean, std
    
    def predict_noise(self, input_ids, attention_mask, embedding, eps=None, return_logprob=True, deterministic=False):
        mean, std = self.forward(input_ids, attention_mask, embedding)
        # save for logging
        self.mean = mean
        self.std = std
        if deterministic:
            return mean, torch.ones(mean.shape)
        else:
            noise = torch.normal(mean, std)
            if eps:
                noise = (noise.T / noise.norm(dim=1) * eps).T
            if return_logprob:
                logprob = self.get_logprobs(noise, mean, std)
                return noise, logprob
            return noise

    def get_logprobs(self, noise, mean, std):
        probs = 0.5 * (1 + torch.erf((noise - mean) * std.reciprocal() / math.sqrt(2)))
        log_probs = torch.log(probs + 1e-8)
        if mean.ndim > 0:
            return log_probs
        return log_probs
    
    def value_function(self, input_ids, attention_masks):
        batch_size = input_ids.shape[0]
        hidden_embs = self.transformer(input_ids, attention_masks).last_hidden_state
        hidden_embs = hidden_embs.view(batch_size, -1, self.transformer.config.d_model)[:, -1, :]
        return (self.value_head(hidden_embs))