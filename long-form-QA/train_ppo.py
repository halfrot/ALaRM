import argparse
import json
import logging
import numpy as np
import random
import trl
import torch
import torch.nn.functional as F
import accelerate
import wandb
from peft import LoraConfig
from tqdm import tqdm
from typing import List, Optional
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    LlamaConfig,
    LlamaModel,
    LlamaTokenizer
)
from my_reward import FineGrainedReward
from trl import PPOConfig
from pathlib import Path


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


logging.basicConfig(level=logging.ERROR)

# prepare accelerator and logger
accelerator = accelerate.Accelerator()
device = accelerator.device
log = accelerate.logging.get_logger(__name__, log_level='INFO')


def log_info(s):
    if accelerator.is_main_process:
        log.info(s)


# load parameters
parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", default=False)
parser.add_argument("--test_do_sample", action="store_true", default=False)
parser.add_argument("--test_on_train", action="store_true", default=False)
parser.add_argument("--policy_ckpt", type=str, help="path to the initial policy checkpoint",
                    default="./long-form-QA/model_ckpts/t5-large-1k-train")
parser.add_argument("--relevance_model_ckpt", type=str, default="./long-form-QA/model_ckpts/rel_rm")
parser.add_argument("--factuality_model_ckpt", type=str, default="./long-form-QA/model_ckpts/fact_rm")
parser.add_argument("--completeness_model_ckpt", type=str, default="./long-form-QA/model_ckpts/comp_rm")
parser.add_argument("--relevance_positive_reward", type=float, default=0.3)
parser.add_argument("--relevance_negative_reward", type=float, default=-0.3)
parser.add_argument("--factuality_positive_reward", type=float, default=0.5)
parser.add_argument("--factuality_negative_reward", type=float, default=-0.5)
parser.add_argument("--completeness_mean", type=float, default=-0.44677690555995353)
parser.add_argument("--completeness_std", type=float, default=8.301160619054132)
parser.add_argument("--completeness_bias", type=float, default=0.0)
parser.add_argument("--completeness_scale", type=float, default=0.3)
parser.add_argument("--ultra_mean", type=float, default=-4.896130742253484)
parser.add_argument("--ultra_std", type=float, default=3.0595760500526468)
parser.add_argument("--length_mean", type=float, default=96.78820166320166)

parser.add_argument("--reward_type", required=True, type=str,
                    choices=["hierarchical", "holistic_only", "aspect_only", "weighted_sum"])
parser.add_argument("--sigmoid_shaping", action="store_true", default=False)
parser.add_argument("--length_reward", action="store_true", default=False)
parser.add_argument("--w_rel", type=float, default=1)
parser.add_argument("--w_fact", type=float, default=1)
parser.add_argument("--w_comp", type=float, default=1)
parser.add_argument("--hierarchical_threshold", type=float, default=0.6)
parser.add_argument("--lr", type=float, default=0.0002)
parser.add_argument("--holistic_reward_weight", type=float, default=5)
parser.add_argument("--max_input_len", type=int, default=1024)
parser.add_argument("--max_generated_len", type=int, default=200)
parser.add_argument("--outer_epoch", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--mini_batch_size", type=int, default=4)
parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
parser.add_argument("--kl_threshold", type=float, default=0.1)
parser.add_argument("--n_ppo_epoch_per_rollout", type=int, default=4)
parser.add_argument("--eval_interval", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save_dir", type=str, default="./long-form-QA/model_ckpts/alarm")
parser.add_argument("--run_name", type=str, help="wandb run name",
                    default="no-name-assigned")
args = parser.parse_args()


# prepare data
class TextGenDataset(Dataset):
    def __init__(self, split, tokenizer, accelerator=None, length_limit=None):
        super().__init__()

        self.split = split
        self.dataset_fns = {
            "train": "./long-form-QA/qa_feedback/train.json",
            "dev": "./long-form-QA/qa_feedback/dev.json",
            "test": "./long-form-QA/qa_feedback/test.json"
        }

        self.n_card = 1
        if accelerator is not None:
            self.n_card = accelerator.num_processes

        self.tokenizer = tokenizer

        self.instances = self.load_datasets()

        if length_limit is not None:
            self.instances = self.instances[:length_limit]

        if split == 'train':
            random.shuffle(self.instances)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]

    def load_datasets(self):
        instances = []

        with open(self.dataset_fns[self.split]) as f:
            task_data = json.load(f)

        for task_instance in task_data:
            instances.append({
                "prompt": task_instance['text'],
                "metadata": {
                    "prompt": task_instance['text'],
                    "references": task_instance['answer'],
                    "passages": task_instance['passages'],
                    "question": task_instance['question'],
                }
            })

        log_info(f'Loaded split {self.split} with {len(instances)} total instances')

        instances = instances[:len(instances) // self.n_card * self.n_card]  # or Trainer will stuck
        return instances

    # Make a collate function to fix dataloader weird list batching
    def collate_fn(self, batch):

        # process input prompts
        prompts = [item['prompt'] for item in batch]
        # trl PPO Trainer does the padding part itself, thus needs special input_ids
        trl_input_ids = []
        for item in batch:
            trl_input_ids.append(
                self.tokenizer(item['prompt'], truncation=True, max_length=self.tokenizer.max_input_len,
                               return_tensors='pt').input_ids.squeeze(0))

        prompts_tok = self.tokenizer.batch_encode_plus(
            prompts,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.tokenizer.max_input_len,
        )

        prompts_input_ids = prompts_tok.input_ids
        prompts_attention_mask = prompts_tok.attention_mask

        # process metadata
        metadata = [item['metadata'] for item in batch]

        result = {
            "trl_input_ids": trl_input_ids,
            'prompts_input_ids': prompts_input_ids,
            'prompts_attention_mask': prompts_attention_mask,
            'metadata': metadata,
            "query": prompts
        }
        return result


class LlamaRewardModel(PreTrainedModel):
    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.regression_head = torch.nn.Linear(self.config.hidden_size, 1, bias=False)

    def forward(  # args are the same as LlamaForCausalLM
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]
        rewards = self.regression_head(hidden_states).squeeze(-1)

        ends = attention_mask.cumsum(dim=1).argmax(dim=1).view(-1, 1)
        rewards = torch.gather(rewards, 1, ends)

        return rewards


def main():
    # set seed
    set_seed(args.seed)

    # set saving directories
    log_info(f"Write to output directory: {args.save_dir}")

    # initialize policy and value model tokenizers
    tokenizer = AutoTokenizer.from_pretrained(args.policy_ckpt,
                                              model_max_length=args.max_input_len)
    tokenizer.padding_side = "right"
    tokenizer.max_input_len = args.max_input_len
    tokenizer.max_generated_len = args.max_generated_len

    # Load data
    log_info(f'Loading data ...')
    train_dataset = TextGenDataset('train', tokenizer, accelerator=accelerator)
    # train ds is shuffled in its constructor
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True,
                                  collate_fn=train_dataset.collate_fn)

    eval_dataset = TextGenDataset('dev', tokenizer, accelerator=accelerator)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=eval_dataset.collate_fn)

    train_dataloader, eval_dataloader = accelerator.prepare(train_dataloader, eval_dataloader)

    # Initialize models and optimizer
    log_info(f'Initializing models ...')

    current_device = accelerator.local_process_index
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )

    policy = trl.AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
        args.policy_ckpt,
        torch_dtype=torch.bfloat16,
        device_map={
            "": current_device
        },
        peft_config=lora_config,
        is_trainable=True
    )
    config = PPOConfig(
        model_name=args.policy_ckpt,
        learning_rate=args.lr,
        log_with="wandb",
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        remove_unused_columns=False,
        early_stopping=True,
        target_kl=args.kl_threshold,
        ppo_epochs=args.n_ppo_epoch_per_rollout,
        seed=args.seed,
        tracker_project_name="Hierarchical-RLHF",
        tracker_kwargs={
            "wandb": {
                "name": args.run_name
            }
        }
    )

    ppo_trainer = trl.PPOTrainer(config, policy, tokenizer=tokenizer, dataset=train_dataset,
                                 data_collator=train_dataset.collate_fn)

    reward = FineGrainedReward(
        tokenizer=tokenizer,
        non_factual_model_ckpt=args.relevance_model_ckpt,
        factual_model_ckpt=args.factuality_model_ckpt,
        completeness_model_ckpt=args.completeness_model_ckpt,
        verbosity_positive_reward=args.relevance_positive_reward,
        verbosity_negative_reward=args.relevance_negative_reward,
        factuality_positive_reward=args.factuality_positive_reward,
        factuality_negative_reward=args.factuality_negative_reward,
        completeness_reward_mean=args.completeness_mean,
        completeness_reward_std=args.completeness_std,
        completeness_reward_bias=args.completeness_bias,
        completeness_reward_scale=args.completeness_scale,
    )

    ultra_tokenizer = LlamaTokenizer.from_pretrained("openbmb/UltraRM-13b")
    ultra_tokenizer.pad_token_id = ultra_tokenizer.eos_token_id
    ultra_rm = LlamaRewardModel.from_pretrained("openbmb/UltraRM-13b", torch_dtype=torch.bfloat16,
                                                device_map={
                                                    "": current_device
                                                })
    ultra_template = "Human: {instruction}\nAssistant: {completion}"

    # prepare reward models
    reward.verbosity_reward.nf_reward_model = accelerator.prepare(reward.verbosity_reward.nf_reward_model)
    reward.factuality_reward.f_reward_model = accelerator.prepare(reward.factuality_reward.f_reward_model)
    reward.completeness_reward.model = accelerator.prepare(reward.completeness_reward.model)

    generation_kwargs = {
        "do_sample": True,
        "top_k": 0,
        "top_p": 1.0
    }
    # set eos_token_id to -1 for manual truncation, and thus assign low score to too short generations
    generation_kwargs.update({
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": -1,
        "max_new_tokens": args.max_generated_len,
    })
    for epoch in tqdm(range(args.outer_epoch)):
        if accelerator.is_main_process:
            step_bar = tqdm(range(ppo_trainer.dataloader.__len__()),
                            desc='Train step of epoch %d' % epoch)
        for step, batch in enumerate(ppo_trainer.dataloader):
            trl_query_ids = [t for t in batch["trl_input_ids"]]

            response_tensors = ppo_trainer.generate(
                trl_query_ids,
                # trl PPO Trainer doesn't support attention_mask, default is ones
                # attention_mask=query_mask,
                return_prompt=False,
                **generation_kwargs,
            )

            too_short_idx = []
            no_begin_response_tensors = []
            for i, r_t in enumerate(response_tensors):
                if len(r_t) <= 2:
                    too_short_idx.append(i)
                    r_t = response_tensors[i] = ppo_trainer.generate(
                        trl_query_ids[i],
                        # trl PPO Trainer doesn't support attention_mask, default is ones
                        # attention_mask=query_mask,
                        return_prompt=False,
                        remove_padding=False,
                        **generation_kwargs,
                    )[0]
                no_begin_response_tensors.append(
                    F.pad(
                        r_t[1:],
                        (0, args.max_generated_len - r_t.size(0)),
                        value=tokenizer.pad_token_id
                    )
                )
            no_begin_response_tensors = torch.stack(no_begin_response_tensors)
            generated_attention_mask = (no_begin_response_tensors != tokenizer.pad_token_id).long()

            batch["response"] = tokenizer.batch_decode(no_begin_response_tensors, skip_special_tokens=True,
                                                       clean_up_tokenzization_spaces=True)
            fg_rewards = reward.get_finegrained_reward(
                prompts_input_ids=batch["prompts_input_ids"],
                prompts_attention_mask=batch['prompts_attention_mask'],
                generated_input_ids=no_begin_response_tensors,
                generated_attention_mask=generated_attention_mask,
                generated_texts=batch['response'],
                metadata=batch["metadata"],
                w_rel=args.w_rel,
                w_fact=args.w_fact,
                w_comp=args.w_comp,
                sigmoid_shaping=args.sigmoid_shaping
            )

            # choose the configured reward modeling strategy
            rewards = []
            holistic_rewards = []
            for i in range(len(batch["query"])):
                if i in too_short_idx:
                    holistic_rewards.append(np.float_(-10))
                    rewards.append(
                        [0] * (len(fg_rewards["rewards"][i]) - 1) + [args.holistic_reward_weight * np.float_(-10)])
                    continue
                # deal with too short responses
                with torch.no_grad():
                    inputs = ultra_tokenizer(
                        ultra_template.replace("{instruction}", batch["query"][i]).replace("{completion}",
                                                                                           batch["response"][i]),
                        return_tensors="pt", max_length=1230).to(current_device)
                    holistic_reward = np.float_(ultra_rm(**inputs).item())
                    # z_normalize
                    holistic_reward = (holistic_reward - args.ultra_mean) / args.ultra_std
                holistic_rewards.append(holistic_reward)
                if args.length_reward:
                    fg_rewards["rewards"][i] = [0] * (len(fg_rewards["rewards"][i]) - 1) + [
                        np.float_(len(fg_rewards["rewards"][i]) / args.length_mean)]
                if args.reward_type == "holistic_only":
                    rewards.append(
                        [0] * (len(fg_rewards["rewards"][i]) - 1) + [args.holistic_reward_weight * holistic_reward])
                elif args.reward_type == "aspect_only":
                    rewards.append(fg_rewards["rewards"][i])
                elif args.reward_type == "weighted_sum":
                    fg_rewards["rewards"][i][-1] += args.holistic_reward_weight * holistic_reward
                    rewards.append(fg_rewards["rewards"][i])
                elif args.reward_type == "hierarchical":
                    if holistic_reward < args.hierarchical_threshold:
                        # will be assigned to the last token
                        rewards.append(
                            [0] * (len(fg_rewards["rewards"][i]) - 1) + [args.holistic_reward_weight * holistic_reward])
                    else:
                        fg_rewards["rewards"][i][-1] += args.holistic_reward_weight * holistic_reward
                        rewards.append(fg_rewards["rewards"][i])
            logs = {}
            if accelerator.is_main_process:
                logs.update({
                    "fg_rewards/factuality_rate": np.mean(
                        [correct_n / fg_rewards["n_sentences"][i] if fg_rewards["n_sentences"][i] else 0.5 for
                         i, correct_n in enumerate(fg_rewards["n_factuality_correct"])]).item(),
                    "fg_rewards/verbosity_rate": np.mean(
                        [correct_n / fg_rewards["n_sub_sentences"][i] if fg_rewards["n_sub_sentences"][i] else 0.5 for
                         i, correct_n in enumerate(fg_rewards["n_verbosity_correct"])]).item(),
                    "fg_rewards/completeness_rewards": np.mean([reward[-1] for reward in fg_rewards[
                        "completeness_rewards"]]).item(),
                    "fg_rewards/holistic_rewards": np.mean([r.item() for r in holistic_rewards]),
                })
            stats = ppo_trainer.step(
                trl_query_ids,
                [t for t in response_tensors],
                [torch.tensor(r, dtype=torch.float) for r in rewards]
            )
            ppo_trainer.log_stats(
                stats, batch, [np.sum(reward) for reward in rewards]
            )
            if accelerator.is_main_process:
                step_bar.update()
            if (step + epoch * ppo_trainer.dataloader.__len__()) % args.eval_interval == 0:
                with torch.no_grad():
                    eval_fg_rewards = None
                    eval_holistic_rewards = None
                    eval_rewards = None
                    eval_length = []
                    eval_generation_kwargs = {
                        "do_sample": False,
                        "num_beams": 1
                    }
                    eval_generation_kwargs.update({
                        "pad_token_id": tokenizer.pad_token_id,
                        "eos_token_id": tokenizer.eos_token_id,
                        "max_new_tokens": args.max_generated_len
                    })
                    for i, eval_batch in tqdm(enumerate(eval_dataloader)):
                        trl_query_ids = [t for t in eval_batch["trl_input_ids"]]

                        response_tensors = ppo_trainer.generate(
                            trl_query_ids,
                            return_prompt=False,
                            **eval_generation_kwargs,
                        )
                        eval_length.extend([len(response) for response in response_tensors])

                        no_begin_response_tensors = []
                        for r_t in response_tensors:
                            no_begin_response_tensors.append(
                                F.pad(
                                    r_t[1:],
                                    (0, args.max_generated_len - r_t.size(0)),
                                    value=tokenizer.pad_token_id
                                )
                            )
                        no_begin_response_tensors = torch.stack(no_begin_response_tensors)
                        generated_attention_mask = (no_begin_response_tensors != tokenizer.pad_token_id).long()

                        eval_batch["response"] = tokenizer.batch_decode(no_begin_response_tensors,
                                                                        skip_special_tokens=True,
                                                                        clean_up_tokenzization_spaces=True)

                        fg_rewards = reward.get_finegrained_reward(
                            prompts_input_ids=eval_batch["prompts_input_ids"],
                            prompts_attention_mask=eval_batch['prompts_attention_mask'],
                            generated_input_ids=no_begin_response_tensors,
                            generated_attention_mask=generated_attention_mask,
                            generated_texts=eval_batch['response'],
                            metadata=eval_batch["metadata"],
                            w_rel=args.w_rel,
                            w_fact=args.w_fact,
                            w_comp=args.w_comp,
                            sigmoid_shaping=args.sigmoid_shaping
                        )

                        rewards = []
                        holistic_rewards = []
                        for i in range(len(eval_batch["query"])):
                            inputs = ultra_tokenizer(
                                ultra_template.replace("{instruction}", eval_batch["query"][i]).replace(
                                    "{completion}", eval_batch["response"][i]), return_tensors="pt",
                                max_length=1230).to(current_device)
                            holistic_reward = np.float_(ultra_rm(**inputs).item())
                            # z_normalize
                            holistic_reward = (holistic_reward - args.ultra_mean) / args.ultra_std
                            holistic_rewards.append(holistic_reward)
                            if args.length_reward:
                                fg_rewards["rewards"][i] = [0] * (len(fg_rewards["rewards"][i]) - 1) + [
                                    np.float_(len(fg_rewards["rewards"][i]) / args.length_mean)]
                            if args.reward_type == "holistic_only":
                                rewards.append(
                                    [0] * (len(fg_rewards["rewards"][i]) - 1) + [
                                        args.holistic_reward_weight * holistic_reward])
                            elif args.reward_type == "aspect_only":
                                rewards.append(fg_rewards["rewards"][i])
                            elif args.reward_type == "weighted_sum":
                                fg_rewards["rewards"][i][-1] += args.holistic_reward_weight * holistic_reward
                                rewards.append(fg_rewards["rewards"][i])
                            elif args.reward_type == "hierarchical":
                                if holistic_reward < args.hierarchical_threshold:
                                    # will be assigned to the last token
                                    rewards.append(
                                        [0] * (len(fg_rewards["rewards"][i]) - 1) + [
                                            args.holistic_reward_weight * holistic_reward])
                                else:
                                    fg_rewards["rewards"][i][-1] += args.holistic_reward_weight * holistic_reward
                                    rewards.append(fg_rewards["rewards"][i])
                        if eval_rewards:
                            eval_fg_rewards["n_factuality_correct"] += fg_rewards["n_factuality_correct"]
                            eval_fg_rewards["n_verbosity_correct"] += fg_rewards["n_verbosity_correct"]
                            eval_fg_rewards["n_sentences"] += fg_rewards["n_sentences"]
                            eval_fg_rewards["n_sub_sentences"] += fg_rewards["n_sub_sentences"]
                            eval_fg_rewards["completeness_rewards"] += fg_rewards["completeness_rewards"]
                            eval_holistic_rewards += holistic_rewards
                            eval_rewards += rewards
                        else:
                            eval_fg_rewards = fg_rewards
                            eval_holistic_rewards = holistic_rewards
                            eval_rewards = rewards
                    eval_factuality_rate = np.mean(
                        [correct_n / eval_fg_rewards["n_sentences"][i] if eval_fg_rewards["n_sentences"][i] else 0.5 for
                         i, correct_n in enumerate(eval_fg_rewards["n_factuality_correct"])]).item()
                    eval_verbosity_rate = np.mean([correct_n / eval_fg_rewards["n_sub_sentences"][i] if
                                                   eval_fg_rewards["n_sub_sentences"][i] else 0.5 for i, correct_n in
                                                   enumerate(eval_fg_rewards["n_verbosity_correct"])]).item()
                    eval_completeness_rewards = np.mean([reward[-1] for reward in eval_fg_rewards[
                        "completeness_rewards"]]).item()
                    eval_holistic_rewards = np.mean([r.item() for r in eval_holistic_rewards])
                    eval_rewards = np.mean([np.sum(reward) for reward in eval_rewards]).item()
                    eval_response_length = np.mean(eval_length)

                    eval_factuality_rate = torch.tensor(eval_factuality_rate).to(current_device)
                    eval_verbosity_rate = torch.tensor(eval_verbosity_rate).to(current_device)
                    eval_completeness_rewards = torch.tensor(eval_completeness_rewards).to(current_device)
                    eval_holistic_rewards = torch.tensor(eval_holistic_rewards).to(current_device)
                    eval_rewards = torch.tensor(eval_rewards).to(current_device)
                    eval_response_length = torch.tensor(eval_response_length).to(current_device)

                    if ppo_trainer.is_distributed:
                        import torch.distributed as dist

                        dist.barrier()
                        dist.all_reduce(eval_factuality_rate)
                        dist.all_reduce(eval_verbosity_rate)
                        dist.all_reduce(eval_completeness_rewards)
                        dist.all_reduce(eval_holistic_rewards)
                        dist.all_reduce(eval_rewards)
                        dist.all_reduce(eval_response_length)

                    if accelerator.is_main_process:
                        num_proc = accelerator.num_processes
                        logs.update({
                            "eval/factuality_rate": eval_factuality_rate / num_proc,
                            "eval/verbosity_rate": eval_verbosity_rate / num_proc,
                            "eval/completeness_rewards": eval_completeness_rewards / num_proc,
                            "eval/holistic_rewards": eval_holistic_rewards / num_proc,
                            "eval/rewards": eval_rewards / num_proc,
                            "eval/response_length": eval_response_length / num_proc
                        })
                ppo_trainer.save_pretrained(
                    Path(args.save_dir) / f"step_{step + 1 + epoch * ppo_trainer.dataloader.__len__()}")
            if accelerator.is_main_process:
                wandb.log(logs)


def test():
    if accelerator.is_main_process:
        run = wandb.init(
            project="Hierarchical-RLHF",
            name=args.run_name,
            config={
                "model_ckpt": args.policy_ckpt,
                "seed": args.seed,
            }
        )
    # set seed
    set_seed(args.seed)

    # initialize policy and value model tokenizers
    tokenizer = AutoTokenizer.from_pretrained(args.policy_ckpt, model_max_length=args.max_input_len)
    tokenizer.padding_side = "right"
    tokenizer.max_input_len = args.max_input_len
    tokenizer.max_generated_len = args.max_generated_len

    # Load data
    log_info(f'Loading data ...')

    if args.test_on_train:
        test_dataset = TextGenDataset('train', tokenizer, accelerator=accelerator)
    else:
        test_dataset = TextGenDataset('test', tokenizer, accelerator=accelerator)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=test_dataset.collate_fn)

    test_dataloader = accelerator.prepare(test_dataloader)

    # Initialize models and optimizer
    log_info(f'Initializing models ...')

    current_device = accelerator.local_process_index
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )

    policy = trl.AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
        args.policy_ckpt,
        torch_dtype=torch.bfloat16,
        device_map={
            "": current_device
        },
        peft_config=lora_config,
        is_trainable=False
    )

    reward = FineGrainedReward(
        tokenizer=tokenizer,
        non_factual_model_ckpt=args.relevance_model_ckpt,
        factual_model_ckpt=args.factuality_model_ckpt,
        completeness_model_ckpt=args.completeness_model_ckpt,
        verbosity_positive_reward=args.relevance_positive_reward,
        verbosity_negative_reward=args.relevance_negative_reward,
        factuality_positive_reward=args.factuality_positive_reward,
        factuality_negative_reward=args.factuality_negative_reward,
        completeness_reward_mean=args.completeness_mean,
        completeness_reward_std=args.completeness_std,
        completeness_reward_bias=args.completeness_bias,
        completeness_reward_scale=args.completeness_scale,
    )

    ultra_tokenizer = LlamaTokenizer.from_pretrained("openbmb/UltraRM-13b")
    ultra_tokenizer.pad_token_id = ultra_tokenizer.eos_token_id
    ultra_rm = LlamaRewardModel.from_pretrained("openbmb/UltraRM-13b", torch_dtype=torch.bfloat16, device_map={
        "": current_device
    })
    ultra_template = "Human: {instruction}\nAssistant: {completion}"

    # prepare reward models
    reward.verbosity_reward.nf_reward_model = accelerator.prepare(reward.verbosity_reward.nf_reward_model)
    reward.factuality_reward.f_reward_model = accelerator.prepare(reward.factuality_reward.f_reward_model)
    reward.completeness_reward.model = accelerator.prepare(reward.completeness_reward.model)

    test_fg_rewards = None
    test_holistic_rewards = None
    test_rewards = None
    test_length = []
    logs = {}
    if args.test_do_sample:
        test_generation_kwargs = {
            "do_sample": True,
            "top_k": 0,
            "top_p": 1.0
        }
    else:
        test_generation_kwargs = {
            "do_sample": False,
            "num_beams": 1
        }
    test_generation_kwargs.update({
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "max_new_tokens": args.max_generated_len,
    })
    test_gather_list = []
    with torch.no_grad():
        for i, test_batch in tqdm(enumerate(test_dataloader)):
            query_inputs = {
                "inputs": test_batch["prompts_input_ids"],
                "attention_mask": test_batch["prompts_attention_mask"]
            }

            response_tensors = policy.generate(
                **query_inputs,
                **test_generation_kwargs,
            )
            # default generate() doesn't remove pad token
            response_tensors_length = []
            for response in response_tensors:
                if tokenizer.eos_token_id in response:
                    response_tensors_length.append((response == tokenizer.eos_token_id).nonzero()[0].item())
                else:
                    response_tensors_length.append(response.size(0))
            test_length.extend(response_tensors_length)

            no_begin_response_tensors = []
            for r_t in response_tensors:
                no_begin_response_tensors.append(
                    F.pad(
                        r_t[1:],
                        (0, args.max_generated_len - r_t.size(0)),
                        value=tokenizer.pad_token_id
                    )
                )
            no_begin_response_tensors = torch.stack(no_begin_response_tensors)
            generated_attention_mask = (no_begin_response_tensors != tokenizer.pad_token_id).long()

            test_batch["response"] = tokenizer.batch_decode(no_begin_response_tensors, skip_special_tokens=True,
                                                            clean_up_tokenzization_spaces=True)

            fg_rewards = reward.get_finegrained_reward(
                prompts_input_ids=test_batch["prompts_input_ids"],
                prompts_attention_mask=test_batch['prompts_attention_mask'],
                generated_input_ids=no_begin_response_tensors,
                generated_attention_mask=generated_attention_mask,
                generated_texts=test_batch['response'],
                metadata=test_batch["metadata"],
                w_rel=args.w_rel,
                w_fact=args.w_fact,
                w_comp=args.w_comp,
                sigmoid_shaping=args.sigmoid_shaping
            )

            rewards = []
            holistic_rewards = []
            for i in range(len(test_batch["query"])):
                inputs = ultra_tokenizer(
                    ultra_template.replace("{instruction}", test_batch["query"][i]).replace("{completion}",
                                                                                            test_batch["response"][i]),
                    return_tensors="pt", max_length=1230).to(current_device)
                holistic_reward = np.float_(ultra_rm(**inputs).item())
                # z_normalize
                holistic_reward = (holistic_reward - args.ultra_mean) / args.ultra_std
                holistic_rewards.append(holistic_reward)
                if args.length_reward:
                    fg_rewards["rewards"][i] = [0] * (len(fg_rewards["rewards"][i]) - 1) + [
                        np.float_(len(fg_rewards["rewards"][i]) / args.length_mean)]
                if args.reward_type == "holistic_only":
                    rewards.append(
                        [0] * (len(fg_rewards["rewards"][i]) - 1) + [args.holistic_reward_weight * holistic_reward])
                elif args.reward_type == "aspect_only":
                    rewards.append(fg_rewards["rewards"][i])
                elif args.reward_type == "weighted_sum":
                    fg_rewards["rewards"][i][-1] += args.holistic_reward_weight * holistic_reward
                    rewards.append(fg_rewards["rewards"][i])
                elif args.reward_type == "hierarchical":
                    if holistic_reward < args.hierarchical_threshold:
                        # will be assigned to the last token
                        rewards.append(
                            [0] * (len(fg_rewards["rewards"][i]) - 1) + [
                                args.holistic_reward_weight * holistic_reward])
                    else:
                        fg_rewards["rewards"][i][-1] += args.holistic_reward_weight * holistic_reward
                        rewards.append(fg_rewards["rewards"][i])
                test_gather_list.append({
                    "query": test_batch["query"][i],
                    "generation": test_batch["response"][i],
                    "holistic_reward": holistic_reward,
                    "factuality_rate": fg_rewards["n_factuality_correct"][i] / fg_rewards["n_sentences"][i] if
                    fg_rewards["n_sentences"][i] else 0.5,
                    "verbosity_rate": fg_rewards["n_verbosity_correct"][i] / fg_rewards["n_sub_sentences"][i] if
                    fg_rewards["n_sub_sentences"][i] else 0.5,
                    "completeness_reward": fg_rewards["completeness_rewards"][i][-1],
                    "response_length": response_tensors_length[i]
                })
            if test_rewards:
                test_fg_rewards["n_factuality_correct"] += fg_rewards["n_factuality_correct"]
                test_fg_rewards["n_verbosity_correct"] += fg_rewards["n_verbosity_correct"]
                test_fg_rewards["n_sentences"] += fg_rewards["n_sentences"]
                test_fg_rewards["n_sub_sentences"] += fg_rewards["n_sub_sentences"]
                test_fg_rewards["completeness_rewards"] += fg_rewards["completeness_rewards"]
                test_holistic_rewards += holistic_rewards
                test_rewards += rewards
            else:
                test_fg_rewards = fg_rewards
                test_holistic_rewards = holistic_rewards
                test_rewards = rewards
    test_factuality_rate = np.mean(
        [correct_n / test_fg_rewards["n_sentences"][i] if test_fg_rewards["n_sentences"][i] else 0.5 for i, correct_n in
         enumerate(test_fg_rewards["n_factuality_correct"])]).item()
    test_verbosity_rate = np.mean(
        [correct_n / test_fg_rewards["n_sub_sentences"][i] if test_fg_rewards["n_sub_sentences"][i] else 0.5 for
         i, correct_n in enumerate(test_fg_rewards["n_verbosity_correct"])]).item()
    test_completeness_rewards = np.mean([reward[-1] for reward in test_fg_rewards[
        "completeness_rewards"]]).item()
    test_holistic_rewards = np.mean([r.item() for r in test_holistic_rewards])
    test_rewards = np.mean([np.mean(reward) for reward in test_rewards]).item()
    test_response_length = np.mean(test_length)

    test_factuality_rate = torch.tensor(test_factuality_rate).to(current_device)
    test_verbosity_rate = torch.tensor(test_verbosity_rate).to(current_device)
    test_completeness_rewards = torch.tensor(test_completeness_rewards).to(current_device)
    test_holistic_rewards = torch.tensor(test_holistic_rewards).to(current_device)
    test_rewards = torch.tensor(test_rewards).to(current_device)
    test_response_length = torch.tensor(test_response_length).to(current_device)

    if accelerator.use_distributed:
        import torch.distributed as dist

        dist.barrier()
        dist.all_reduce(test_factuality_rate)
        dist.all_reduce(test_verbosity_rate)
        dist.all_reduce(test_completeness_rewards)
        dist.all_reduce(test_holistic_rewards)
        dist.all_reduce(test_rewards)
        dist.all_reduce(test_response_length)
        gathered_list = accelerator.gather_for_metrics(test_gather_list)
    if accelerator.is_main_process:
        num_proc = accelerator.num_processes
        logs.update({
            "test/factuality_rate": test_factuality_rate / num_proc,
            "test/verbosity_rate": test_verbosity_rate / num_proc,
            "test/completeness_rewards": test_completeness_rewards / num_proc,
            "test/holistic_rewards": test_holistic_rewards / num_proc,
            "test/rewards": test_rewards / num_proc,
            "test/response_length": test_response_length / num_proc
        })
        wandb.log(logs)

        def remove_duplicate_dicts(lst):
            unique_list = []
            query_hash = []
            for item in lst:
                if item["query"] not in query_hash:
                    query_hash.append(item["query"])
                    unique_list.append(item)
            return unique_list

        gathered_list = remove_duplicate_dicts(gathered_list)
        sorted_list = sorted(gathered_list, key=lambda d: d['query'])
        with open(args.save_dir, "w") as f:
            json.dump(sorted_list, f, indent=4)


if __name__ == '__main__':
    if args.test:
        test()
    else:
        main()
