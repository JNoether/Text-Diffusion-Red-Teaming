from torch import nn
import torch
import torch.nn.functional as F
from dataclasses import asdict
import wandb
from sentence_transformers import util
import pandas as pd
import numpy as np
import vec2text
import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

class RedTeamDiffusion:
    def __init__(self, config, diffusion_model, diffusion_tokenizer, embedder, embedder_tokenizer, decoder, device) -> None:
        assert config.regularization in ["threshold_cos_sim", "mean", "threshold_mean", "clipping", "none"]
        assert config.reward in ["toxicity", "1/0", "relative_toxicity"]
        self.config = config
        self.device = device
        self.diffusion_tokenizer = diffusion_tokenizer
        self.diffusion_model = diffusion_model
        self.diffusion_model.to(device)
        self.embedder = embedder
        self.embedder.to(device)
        self.embedder_tokenizer = embedder_tokenizer
        self.decoder = decoder
        self.optim = torch.optim.AdamW(diffusion_model.parameters(), lr=self.config.lr, weight_decay=1e-4)
        if self.config.lr_schedule:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, "max", patience=3, factor=0.1, cooldown=2)
        # set up logging
        if config.log_wandb:
            wandb.init(
                project=config.name,
                config = asdict(config)
            )
    
    def tokenize_diffusion_input(self, input):
        return self.diffusion_tokenizer(input, return_tensors="pt", padding=True)

    def tokenize_embeder(self, input):
        return self.embedder_tokenizer(input, return_tensors="pt", padding=True)

    def embedd_input(self, input_ids):
        hidden_state = self.embedder(**input_ids).last_hidden_state
        return vec2text.models.model_utils.mean_pool(hidden_state, input_ids["attention_mask"])
    
    def predict_noise(self, input_ids, attention_mask, embedding, eps=None, return_logprob=True, deterministic=False):
        return self.diffusion_model.predict_noise(input_ids, attention_mask, embedding, eps, return_logprob, deterministic)
    
    def apply_noise(self, embedding, noise):
        return embedding - noise

    def reconstruct_embedding(self, embedding, num_steps=20):
        return vec2text.invert_embeddings(
            embeddings=embedding,
            corrector=self.decoder,
            num_steps=num_steps
        )
    
    def modify_prompt(self, original_prompt, num_steps=20, deterministic=False):
        tokens = self.tokenize_embeder(original_prompt)
        tokens.to(self.device)
        emb = self.embedd_input(tokens)
        emb.to(self.device)
        tokens = self.tokenize_diffusion_input(original_prompt)
        tokens.to(self.device)
        noise, logprobs = self.predict_noise(**tokens, embedding=emb, eps=self.config.eps if self.config.normalize_noise else None, return_logprob=True, deterministic=deterministic)
        if self.config.regularization == "clipping":
            noise = torch.clamp(noise, -self.config.eps/noise.shape[-1]**0.5, self.config.eps/noise.shape[-1]**0.5)
        # save for logging
        self.mean = self.diffusion_model.mean
        self.std = self.diffusion_model.std
        modified_emb = self.apply_noise(emb, noise)
        return self.reconstruct_embedding(modified_emb, num_steps=num_steps), tokens, noise, logprobs 

    def gae(self, rewards, values, mask):
        rewards = rewards.to(self.device)
        batch_size = rewards.shape[0]
        advantages = torch.zeros(rewards.shape)
        last_value, last_advantage = torch.zeros(batch_size, device=self.device), torch.zeros(batch_size, device=self.device) 
        for t in range(rewards.shape[-1] - 1, -1, -1):
            last_value *= mask[:, t]
            last_advantage *= mask[:, t]
            delta = rewards[:, t] + self.config.gamma * last_value - values[:, t]
            advantages[:, t] = delta + self.config.gamma * self.config._lambda * last_advantage
            last_advantage = advantages[:, t]
            last_value = values[:, t]
        return advantages


    def policy_loss(self, logprobs, logprobs_old, advantages, eps=0.1):
        ratio = torch.exp(logprobs - logprobs_old)
        ratio_clipped = torch.clamp(ratio, 1-eps, 1+eps)
        loss = torch.max(ratio * -advantages, ratio_clipped * -advantages)
        return loss.mean()

    def value_loss(self, values, rewards):
        # reward here is 0, 0, 0, ..., r
        returns = (torch.ones(rewards.shape, device=self.device).T * rewards.sum(dim=1)).T
        loss = (values - returns)**2
        return loss.mean()
    
    def value_loss_clipped(self, old_values, values, rewards):
        returns = (torch.ones(rewards.shape, device=self.device).T * rewards.sum(dim=1)).T
        loss_uncliped = (values - returns)**2
        clipped_values = torch.clamp(values, old_values - self.config.vf_clipping_parameter, old_values + self.config.vf_clipping_parameter)
        loss_clipped = (clipped_values - returns)**2
        return torch.max(loss_uncliped, loss_clipped).mean()
    
    def mean_regularization(self, mean):
        return ((mean.abs() + 1e-6)**0.5).mean()

    def threshold_mean_regularization(self, mean, noise):
        return F.relu(mean.norm(dim=1) - self.config.eps).mean()

    def threshold_cos_sim_regularization(self, mean, emb):
        modified_emb = (emb + mean)
        cos_loss =  F.cosine_embedding_loss(emb, modified_emb, target=torch.ones(emb.shape[0], device=self.device), reduction="none")
        return F.relu(cos_loss - self.config.eps).mean()

    def minibatch_step(self, input_ids, attention_mask, modified_ids, modified_masks, embeddings, noise, logprobs_old, rewards, log_dict=None):
        rewards=rewards.to(self.device).detach()
        logprobs_old = logprobs_old.detach()
        noise = noise.detach()
        embeddings = embeddings.detach()
        old_values = self.diffusion_model.value_function(input_ids, attention_mask)
        advantages = self.gae(rewards=rewards, values=old_values, mask=attention_mask)
        # normalize advantages
        if self.config.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.to(self.device).detach()
        for epoch in range(self.config.ppo_epochs):
            mean, std = self.diffusion_model(input_ids, attention_mask, embeddings)
            current_logprobs = self.diffusion_model.get_logprobs(noise, mean, std)
            if F.kl_div(current_logprobs, logprobs_old, reduce="mean", log_target=True) > self.config.target_kl:
                break
            values = self.diffusion_model.value_function(input_ids, attention_mask)
            
            policy_loss = self.policy_loss(logprobs=current_logprobs, logprobs_old=logprobs_old, advantages=advantages.detach(), eps=self.config.clipping_parameter)
            if self.config.clip_vfloss:
                value_loss = self.value_loss_clipped(old_values=old_values.detach(), values=values, rewards=rewards)
            else:
                value_loss = self.value_loss(values=values, rewards=rewards)
            if self.config.regularization == "mean":
                reg_loss = self.mean_regularization(mean)
            elif self.config.regularization == "threshold_mean":
                reg_loss = self.threshold_mean_regularization(mean, noise)
            elif self.config.regularization == "threshold_cos_sim":
                reg_loss = self.threshold_cos_sim_regularization(mean, embeddings)
            else:
                reg_loss = torch.tensor(0)
            loss = policy_loss + self.config.vf_coef * value_loss + self.config.reg_coef * reg_loss
            self.optim.zero_grad()
            loss.backward()
            # clip gradients
            nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), self.config.gradient_clipping)
            self.optim.step()
            if log_dict:
                log_dict["loss/loss"].append(loss.item())
                log_dict["loss/policy_loss"].append(policy_loss.item())
                log_dict["loss/value_loss"].append(value_loss.item())
                log_dict["loss/reg_loss"].append(reg_loss.item())
                log_dict["ppo/kl_div"].append(F.kl_div(logprobs_old, current_logprobs).mean().item())
                if len(advantages) > 1:
                    log_dict["ppo/advantages"] += advantages.squeeze().detach().cpu().tolist()
                else:
                    log_dict["ppo/advantages"].append(advantages.item())

    def training_step(self, input_ids, attention_mask, modified_ids, modified_masks, noise, logprobs_old, rewards, batch):
        tokens = self.tokenize_embeder(batch)
        tokens.to(self.device)
        emb = self.embedd_input(tokens)
        emb.to(self.device)
        log_dict = {
            "env/reward_dist" : rewards.squeeze().numpy(),
            "env/reward_mean" : rewards.mean().item(),
            "env/reward_std" : rewards.std().item(),
            "env/toxic_answer_ratio" : (rewards > 0.).sum().item() / len(rewards), 
            "pred/action/mean_dist" : self.mean.squeeze().detach().cpu().numpy(),
            "pred/action/mean_mean" : self.mean.mean().item(),
            "pred/action/std_dist" : self.std.squeeze().detach().cpu().numpy(),
            "pred/action/std_mean" : self.std.mean().item(),
            "pred/noise_mean" : noise.mean(dim=0).detach().cpu().numpy(),
            "pred/noise_std" : noise.std(dim=0).detach().cpu().numpy(),
            "pred/noise_dist" : noise.mean(dim=0).detach().cpu(),
            "pred/noise_norm" : noise.norm(dim=1).mean().item(),
            "pred/noise_norm_dist" : noise.norm(dim=1).detach().cpu(),
            "pred/log-probabilities_dist" : logprobs_old.detach().cpu().numpy(),
            "pred/log-probabilities_mean" : logprobs_old.mean().item(),
            # will be inserted by train_minibatch
            "loss/loss" : [],
            "loss/policy_loss" : [],
            "loss/value_loss" : [],
            "loss/reg_loss" : [],
            "ppo/kl_div" : [],
            "ppo/advantages" : []
        }
        batch_size = input_ids.shape[0]
        # mean, std = self.diffusion_model(input_ids, attention_mask)
        # logprobs_old = self.diffusion_model.get_logprobs(noise, mean, std)
        for minibatch_start in range(0, batch_size, self.config.minibatch_size):
            minibatch_end = minibatch_start + self.config.minibatch_size
            ids_batch = input_ids[minibatch_start : minibatch_end]
            mask_batch = attention_mask[minibatch_start : minibatch_end]
            modified_ids_batch = modified_ids[minibatch_start : minibatch_end]
            modified_masks_batch = modified_masks[minibatch_start : minibatch_end]
            noise_batch = noise[minibatch_start : minibatch_end]
            rew_batch = rewards[minibatch_start : minibatch_end]
            logprobs_batch = logprobs_old[minibatch_start : minibatch_end]
            emb_batch = emb[minibatch_start : minibatch_end]
            self.minibatch_step(input_ids=ids_batch, attention_mask=mask_batch, modified_ids=modified_ids_batch, modified_masks=modified_masks_batch, embeddings=emb_batch, noise=noise_batch, logprobs_old=logprobs_batch, rewards=rew_batch, log_dict=log_dict)
        self.diffusion_model.std_head -= self.config.std_anneal
        for key in ["loss/loss", "loss/policy_loss", "loss/value_loss", "loss/reg_loss", "ppo/advantages"]:
            log_dict[key] = sum(log_dict[key]) / len(log_dict[key])
        return log_dict

    def log(self, log_dict, original_prompts, modified_prompts, llm_output, eval_prompts, eval_prompts_modified, eval_llm_outputs, eval_rewards):
        wandb.log(log_dict)
        original_tok = self.tokenize_embeder(original_prompts).to(self.device)
        # original_emb = self.embedd_input(original_tok)
        original_emb = self.embedder(**original_tok).last_hidden_state
        original_emb = vec2text.models.model_utils.mean_pool(original_emb, original_tok["attention_mask"])
        modified_tok = self.tokenize_embeder(modified_prompts).to(self.device)
        modified_emb = self.embedder(**modified_tok).last_hidden_state
        modified_emb = vec2text.models.model_utils.mean_pool(modified_emb, modified_tok["attention_mask"])
        # modified_emb = self.embedd_input(modified_tok)
        cos_sim = [util.cos_sim(original_emb[i], modified_emb[i]).item() for i in range(len(original_emb))]
        
        eval_tok = self.tokenize_embeder(eval_prompts).to(self.device)
        # original_emb = self.embedd_input(original_tok)
        eval_emb = self.embedder(**eval_tok).last_hidden_state
        eval_emb = vec2text.models.model_utils.mean_pool(eval_emb, eval_tok["attention_mask"])
        modified_eval_tok = self.tokenize_embeder(eval_prompts_modified).to(self.device)
        modified_eval_emb = self.embedder(**modified_eval_tok).last_hidden_state
        modified_eval_emb = vec2text.models.model_utils.mean_pool(modified_eval_emb, modified_eval_tok["attention_mask"])
        eval_cos_sim = [util.cos_sim(eval_emb[i], modified_eval_emb[i]).item() for i in range(len(eval_emb))]
        
        wandb.log({"env/cos_sim_mean" : sum(cos_sim)/len(cos_sim), 
                   "env/cos_sim" : np.array(cos_sim),
                   "eval/reward_mean" : eval_rewards.mean().item(),
                   "eval/toxic_ratio" : (eval_rewards > 0.).sum(),
                   "eval/cos_sim" : np.array(eval_cos_sim),
                   "eval/cos_sim_mean" : sum(eval_cos_sim) / len(eval_cos_sim)})
        data = pd.DataFrame({
                             "Original Prompt" : original_prompts,
                             "Modified Prompt" : modified_prompts,
                             "LLM Output" : llm_output,
                             "Toxicity" : log_dict["env/toxicity_dist"],
                             "Cosine Similarity" : np.array(cos_sim),
                             "Norm" : log_dict["pred/noise_norm_dist"]})
        res_table = wandb.Table(data=data)
        wandb.log({"Tables/Conversations" : res_table})

        data = pd.DataFrame({
                             "Original Prompt" : eval_prompts,
                             "Modified Prompt" : eval_prompts_modified,
                             "LLM Output" : eval_llm_outputs,
                             "Toxicity" : eval_rewards.squeeze(),
                             "Cosine Similarity" : np.array(eval_cos_sim).squeeze()})
        res_table = wandb.Table(data=data)
        wandb.log({"Tables/Conversations_eval" : res_table})

        