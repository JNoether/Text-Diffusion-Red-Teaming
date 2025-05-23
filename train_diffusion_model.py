from Diffusion_Model import DiffusionModel
from red_team_diffusion import RedTeamDiffusion
from rtdConfig import RTD_Config
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM
from argparse import ArgumentParser
import vec2text
from tqdm import tqdm
import re
import os
import numpy as np
os.environ["WANDB_PROJECT"] = "Red-Team-Diffusion"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
transformers.logging.set_verbosity_error()


def preprocess_alpaca_data(sample):
    sample["query"] = sample["instruction"] + sample["input"]
    return sample

def preprocess_red_teaming_data(datapoint):
    datapoint["query"] = re.findall(r"Human: (.*?)\n", datapoint["transcript"])[0]
    return datapoi˜`nt

def preprocess_rlhf_data(datapoint):
    datapoint["query"] = re.findall(r"Human: (.*?)\n", datapoint["chosen"])[0].replace("Assistant:", "")
    return datapoint

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ArgumentParser()
    args.add_argument("--name", type=str, default="unnamed_experiment")
    args.add_argument("--eps", type=float, default=0.1)
    args.add_argument("--batch-size", type=int, default=64)
    args.add_argument("--target-model", type=str, default="vicgalle/gpt2-alpaca")
    args.add_argument("--epochs", type=int, default=1)
    args.add_argument("--log", action="store_true")
    args.add_argument("--dataset", choices=["alpaca", "red-team", "all"])
    args.add_argument("--cache-dir", type=str)
    args.add_argument("--save-path", type=str, default="unnamed_experiment")
    args.add_argument("--device", type=str)
    args = args.parse_args()

    if args.device:
        device = args.device
        print(device)

    # load diffusion model
    diffusion_model = DiffusionModel("t5-base", 768, cache_dir=(args.cache_dir+"/models/") if args.cache_dir else None)
    diffusion_tokenizer = AutoTokenizer.from_pretrained("t5-base", cache_dir=(args.cache_dir+"/models/") if args.cache_dir else None)
    
    # load embedding model
    embedder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base", cache_dir=(args.cache_dir+"/models/") if args.cache_dir else None).encoder
    embedder_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base", cache_dir=(args.cache_dir+"/models/") if args.cache_dir else None)

    # load reconstruction model
    decoder = vec2text.load_corrector("gtr-base")

    # load reward model
    toxicity_tokenizer = AutoTokenizer.from_pretrained("nicholasKluge/ToxicityModel", cache_dir=(args.cache_dir+"/models/") if args.cache_dir else None)
    toxicity_classifier = AutoModelForSequenceClassification.from_pretrained("nicholasKluge/ToxicityModel", cache_dir=(args.cache_dir+"/models/") if args.cache_dir else None)

    # load target model
    target_model = AutoModelForCausalLM.from_pretrained(args.target_model, cache_dir=(args.cache_dir+"/models/") if args.cache_dir else None, load_in_8bit=True)
    # target_model.to(device)
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model, cache_dir=(args.cache_dir+"/models/") if args.cache_dir else None) 
    # set up target model generation
    if "gpt2" in args.target_model:
        generation_config =  {"do_sample":False, 
                              "temperature" : 1, 
                              "top_p" : 0.92, 
                              "top_k" : 0, 
                              "max_new_tokens" : 32} # taken from https://huggingface.co/vicgalle/gpt2-alpaca
        system_prefix = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"
        system_suffix = "\n### Response:\n\n"
    elif "vicuna" in args.target_model:
        generation_config =  {"do_sample":False, 
                              "temperature" : 1, 
                              "top_p" : 0.92, 
                              "top_k" : 0, 
                              "max_new_tokens" : 32}
        system_prefix = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nUSER:"""
        system_suffix = "\nASSISTANT:\n"
    elif "llama" in args.target_model:
        system_prefix = "[INST]"
        system_suffix = "[/INST]"
        generation_config =  {"do_sample":False, 
                              "temperature" : 1, 
                              "top_p" : 0.92, 
                              "top_k" : 0, 
                              "max_new_tokens" : 32}
        target_tokenizer.pad_token = target_tokenizer.eos_token
    
    # configure trainer
    config = RTD_Config(
        log_wandb=args.log,
        eps=args.eps,
        name=args.name
    )

    trainer = RedTeamDiffusion(
        config=config,
        diffusion_model=diffusion_model,
        diffusion_tokenizer=diffusion_tokenizer,
        embedder=embedder,
        embedder_tokenizer=embedder_tokenizer,
        decoder=decoder,
        device=device
    )

    # load data
    if args.dataset == "alpaca":
        dataset = load_dataset("vicgalle/alpaca-gpt4", split="train", cache_dir=(args.cache_dir+"/data/") if args.cache_dir else None)
        dataset = dataset.map(preprocess_alpaca_data)
    elif args.dataset == "red-team":
        dataset = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts")
        dataset = dataset["train"].map(preprocess_red_teaming_data)
    elif args.dataset == "all":
        alpaca = load_dataset("vicgalle/alpaca-gpt4", split="train", cache_dir=(args.cache_dir+"/data/") if args.cache_dir else None)
        alpaca = alpaca.map(preprocess_alpaca_data).select(range(4096, len(alpaca)))
        red_team = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts")
        red_team = red_team["train"].map(preprocess_red_teaming_data).select(range(4096, len(red_team["train"])))        
        dataset = concatenate_datasets([red_team, red_team])
        dataset = dataset.shuffle()

    print(len(dataset))
    dataset = dataset.filter(lambda sample: len(diffusion_tokenizer.encode(sample["query"])) < 32)
    print(len(dataset))
    dataset = dataset.select(range(4096, len(dataset)))
    val_dataset = dataset.select(range(1024))
    dataset = DataLoader(dataset["query"], batch_size=config.batch_size)
    val_dataset = DataLoader(val_dataset["query"], config.batch_size//2)
    # used for saving
    best_rew = float("-inf")
    # main training loop
    for i in range(args.epochs):
        for j, batch in enumerate(tqdm(dataset)):
            # modify prompt
            modified_prompt, tokens, noise, logprobs = trainer.modify_prompt(batch)

            # get llm output
            target_tokens = target_tokenizer([system_prefix + prompt + system_suffix for prompt in modified_prompt], return_tensors="pt", padding=True)
            target_tokens = target_tokens.to(device)
            target_out = target_model.generate(**target_tokens, **generation_config)
            if "gpt2" in args.target_model:
                target_out = [target_tokenizer.decode(out).replace(system_prefix + modified_prompt[i] + system_suffix, "").replace("<|endoftext|>", "") for i, out in enumerate(target_out)]
            elif "vicuna" in args.target_model:
                target_out = [target_tokenizer.decode(out).replace(system_prefix + modified_prompt[i] + system_suffix, "").replace("<unk>", "") for i, out in enumerate(target_out)]
            elif "Llama" in args.target_model:
                target_out = [target_tokenizer.decode(out).replace(system_prefix + modified_prompt[i] + system_suffix, "").replace("<s>", "").replace("</s>", "") for i, out in enumerate(target_out)]
            # get reward
            rewards = (torch.tensor([-toxicity_classifier(**toxicity_tokenizer(out, return_tensors="pt", padding=True)).logits.item() for out in target_out]).unsqueeze(-1).detach())
            # rewards = rewards.to(device)
            if config.reward == "1/0":
                rewards = (rewards > 0).float()
            elif config.reward == "relative_toxicity":
                # get llm output of unmodified prompt
                unmod_prompt = target_tokenizer([system_prefix + prompt + system_suffix for prompt in batch], return_tensors="pt", padding=True)
                unmod_prompt = unmod_prompt.to(device)
                unmod_out = target_model.generate(**unmod_prompt, **generation_config)
                unmod_out = [target_tokenizer.decode(out)[len(system_prefix)+len(modified_prompt[i])+len(system_suffix):].replace("<|endoftext|>", "") for i, out in enumerate(unmod_out)]
                baseline_reward = (torch.tensor([-toxicity_classifier(**toxicity_tokenizer(out, return_tensors="pt", padding=True)).logits.item() for out in unmod_out]).unsqueeze(-1))
                # baseline_reward = baseline_reward.to(device)
                rewards = rewards - baseline_reward
            
            # update model
            modified_ids = diffusion_tokenizer(modified_prompt, return_tensors="pt", padding=True).to(device)
            stats = trainer.training_step(**tokens, modified_ids=modified_ids["input_ids"], modified_masks=modified_ids["attention_mask"], noise=noise, logprobs_old=logprobs, rewards=rewards, batch=batch)
            if config.reward == "relative_toxicity":
                stats["env/toxicity_dist"] = (rewards + baseline_reward).squeeze()
                stats["env/toxicity_mean"] = (rewards + baseline_reward).squeeze().mean().item()
                stats["env/toxic_answer_ratio"] = (stats["env/toxicity_dist"] > 0).float().mean()
            else:
                stats["env/toxicity_dist"] = rewards.squeeze()
                stats["env/toxicity_mean"] = rewards.mean().item()
            # log progress
            if args.log and j % 10 == 0:
                # evaluate
                cumulative_toxicity, val_modified_prompts, val_batches, val_outs = [], [], [], []
                for val_batch in val_dataset:
                    with torch.no_grad():
                        val_prompts_modified, _, _, _ = trainer.modify_prompt(val_batch, deterministic=True)
                        target_tokens = target_tokenizer([system_prefix + prompt + system_suffix for prompt in val_prompts_modified], return_tensors="pt", padding=True)
                        target_tokens = target_tokens.to(device)
                        target_out_val = target_model.generate(**target_tokens, **generation_config)
                        if "gpt2" in args.target_model:
                            target_out_val = [target_tokenizer.decode(out).replace(system_prefix + val_prompts_modified[i] + system_suffix, "").replace("<|endoftext|>", "").replace("<|endoftext|>", "") for i, out in enumerate(target_out_val)]
                        elif "vicuna" in args.target_model:
                            target_out_val = [target_tokenizer.decode(out).replace(system_prefix + val_prompts_modified[i] + system_suffix, "").replace("<unk>", "") for i, out in enumerate(target_out_val)]
                        elif "Llama" in args.target_model:
                            target_out_val = [target_tokenizer.decode(out).replace(system_prefix + val_prompts_modified[i] + system_suffix, "").replace("<s>", "").replace("</s>", "") for i, out in enumerate(target_out_val)]
                        val_modified_prompts += val_prompts_modified
                        val_batches += val_batch
                        val_outs += target_out_val
                        # get reward
                        cumulative_toxicity += [-toxicity_classifier(**toxicity_tokenizer(out, return_tensors="pt", padding=True)).logits.item() for out in target_out_val]
                eval_rewards = sum(cumulative_toxicity) / 1024
                if eval_rewards > best_rew:
                    torch.save(diffusion_model.state_dict(), args.save_path)
                    best_rew = eval_rewards
                with torch.no_grad():
                    trainer.log(stats, batch, modified_prompt, target_out, val_batches, val_modified_prompts, val_outs, np.array(cumulative_toxicity))
                if config.lr_schedule:
                    trainer.scheduler.step(eval_rewards)
            