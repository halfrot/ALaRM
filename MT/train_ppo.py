import argparse
import json
import logging
import numpy as np
import random
import textstat
import trl
import accelerate
import wandb
import language_tool_python
import torch
import torch.nn.functional as F
from peft import LoraConfig
from trl import PPOConfig
from tqdm import tqdm
from typing import List, Optional
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    LlamaConfig,
    LlamaModel,
    LlamaTokenizer
)
from typing import Dict
from pathlib import Path
from lingua import Language, LanguageDetectorBuilder

tool = language_tool_python.LanguageTool("en-US", config={
    "maxCheckThreads": 16,
    'cacheSize': 1000,
    'pipelineCaching': True
})
languages = [Language.ENGLISH, Language.SPANISH]
detector = LanguageDetectorBuilder.from_languages(*languages).build()


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
parser.add_argument("--policy_ckpt", default="halfrot/sft-mt5-base", type=str)
parser.add_argument("--readability_mean", type=float, default=11.19907041598879)
parser.add_argument("--readability_std", type=float, default=6.04405504045225)
parser.add_argument("--ultra_mean", type=float, default=-2.34417221913525)
parser.add_argument("--ultra_std", type=float, default=2.946283030347152)
parser.add_argument("--length_mean", type=float, default=36.69312107831745)

parser.add_argument("--reward_type", required=True, type=str,
                    choices=["hierarchical", "holistic_only", "aspect_only", "weighted_sum"])
parser.add_argument("--sigmoid_shaping", action="store_true", default=False)
parser.add_argument("--length_reward", action="store_true", default=False)
parser.add_argument("--w_read", type=float, default=1)
parser.add_argument("--w_grammar", type=float, default=1)
parser.add_argument("--w_confidence", type=float, default=1)
parser.add_argument("--grammar_correct_reward", type=float, default=0)
parser.add_argument("--grammar_incorrect_reward", type=float, default=-1)
parser.add_argument("--hierarchical_threshold", type=float, default=0.5)
parser.add_argument("--lr", type=float, default=0.0005)
parser.add_argument("--holistic_reward_weight", type=float, default=3)
parser.add_argument("--max_len", type=int, default=128)
parser.add_argument("--episodes", type=int, default=50, help="total episodes to train")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--mini_batch_size", type=int, default=16)
parser.add_argument("--eval_batch_size", type=int, default=256)
parser.add_argument("--rm_max_batch_size", type=int, default=32)
parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
parser.add_argument("--kl_threshold", type=float, default=0.1)
parser.add_argument("--n_ppo_epoch_per_rollout", type=int, default=4)
parser.add_argument("--eval_interval", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save_dir", type=str, default="./MT/model_ckpts/alarm")
parser.add_argument("--run_name", type=str, help="wandb run name",
                    default="no-name-assigned")
args = parser.parse_args()


class MTDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        super().__init__()
        self.pairs = []
        self.tokenizer = tokenizer
        self.prefix = "translate Spanish to English: "
        for sample in dataset:
            self.pairs.append(sample["translation"])
        self.len = len(self.pairs)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        pair: Dict = self.pairs[idx]
        trl_input_ids = self.tokenizer(self.prefix + pair["es"], return_tensors="pt",
                                       truncation=True).input_ids.squeeze(0)
        return {
            "prompt": self.prefix + pair["es"],
            "input_ids": trl_input_ids,
        }

    def collator(self, samples):
        prompts = [s["prompt"] for s in samples]
        input_ids = [s['input_ids'] for s in samples]
        batch = {
            'query': prompts,
            'input_ids': input_ids,
        }
        return batch


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


def reward_shape(x, sigmoid_shaping=False, avg_clip=0):
    if sigmoid_shaping:
        return 1 / (1 + np.exp(-x)) - avg_clip
    return np.float_(x)


def get_finegrained_reward(batch_query, batch_response, batch_length, tokenizer):
    fg_rewards = {
        "readability": [],
        "language_confidence": [],
        "grammar_rewards": [],
        "grammar_errors": [],
        "error_rate": [],
        "rewards": []
    }

    textstat.set_lang("en")
    for i, query in enumerate(batch_query):
        response = batch_response[i]
        readability_score = textstat.flesch_kincaid_grade(response)
        # z-normalize
        fg_rewards["readability"].append((readability_score - args.readability_mean) / args.readability_std)

    for i, response in enumerate(batch_response):
        fg_rewards["language_confidence"].append(detector.compute_language_confidence(response, Language.ENGLISH))

        matches = tool.check(response)
        fg_rewards["grammar_errors"].append(len(matches))
        fg_rewards["error_rate"].append(len(matches) / batch_length[i])
        grammar_reward = [reward_shape(args.grammar_correct_reward, args.sigmoid_shaping)] * batch_length[i]
        for match in matches:
            offset_token_id = None
            with torch.no_grad():
                token_ids = tokenizer.encode(response)
                for offset in range(len(token_ids) - 1):
                    if len(tokenizer.decode(token_ids[:offset + 1], skip_special_tokens=True)) >= match.offset:
                        offset_token_id = offset
                        break
                error_p = 1
                while offset_token_id + error_p < len(token_ids) and len(
                        tokenizer.decode(token_ids[offset_token_id:offset_token_id + error_p],
                                         skip_special_tokens=True)) <= match.errorLength:
                    grammar_reward[offset_token_id + error_p - 1] = reward_shape(args.grammar_incorrect_reward,
                                                                                 args.sigmoid_shaping)
                    error_p += 1
        fg_rewards["grammar_rewards"].append(grammar_reward)

    for i in range(len(batch_query)):
        fg_rewards["rewards"].append(args.w_grammar * np.array(fg_rewards["grammar_rewards"][i]))
        fg_rewards["rewards"][i][-1] += (args.w_read * reward_shape(fg_rewards["readability"][i],
                                                                    args.sigmoid_shaping) + args.w_confidence *
                                         reward_shape(fg_rewards["language_confidence"][i], args.sigmoid_shaping))
    return fg_rewards


def main():
    # set seed
    set_seed(args.seed)

    # set saving directories
    log_info(f"Write to output directory: {args.save_dir}")

    # initialize policy and value model tokenizers
    tokenizer = AutoTokenizer.from_pretrained(args.policy_ckpt, model_max_length=args.max_len)
    tokenizer.padding_side = "right"
    tokenizer.max_input_len = args.max_len
    tokenizer.max_generated_len = args.max_len

    # Load data
    log_info(f'Loading data ...')
    raw_dataset = load_dataset("opus/europarl", name="en-es")["train"]
    raw_dataset = raw_dataset.select(range(100000))
    # split train_dataset into train and eval
    raw_dataset = raw_dataset.train_test_split(test_size=0.3, seed=42)
    train_dataset = raw_dataset["train"]
    train_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
    eval_dataset = train_dataset["test"].select(range(2000))
    train_dataset = train_dataset["train"]
    train_dataset = MTDataset(train_dataset, tokenizer)
    eval_dataset = MTDataset(eval_dataset, tokenizer)

    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                 collate_fn=eval_dataset.collator)

    eval_dataloader = accelerator.prepare(eval_dataloader)

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
        steps=args.episodes,
        tracker_project_name="Hierarchical-RLHF-MT",
        tracker_kwargs={
            "wandb": {
                "name": args.run_name
            }
        }
    )

    ppo_trainer = trl.PPOTrainer(config, policy, tokenizer=tokenizer, dataset=train_dataset,
                                 data_collator=train_dataset.collator)
    ultra_tokenizer = LlamaTokenizer.from_pretrained("openbmb/UltraRM-13b")
    ultra_tokenizer.pad_token_id = ultra_tokenizer.eos_token_id
    ultra_rm = LlamaRewardModel.from_pretrained("openbmb/UltraRM-13b", torch_dtype=torch.bfloat16,
                                                device_map={
                                                    "": current_device
                                                })
    ultra_template = "Human: {instruction}\nAssistant: {completion}"

    generation_kwargs = {
        "do_sample": True,
        "top_k": 0,
        "top_p": 1.0
    }
    # set eos_token_id to -1 for manual truncation, and thus assign low score to too short generations
    generation_kwargs.update({
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": -1,
        "max_new_tokens": args.max_len,
    })
    if accelerator.is_main_process:
        step_bar = tqdm(range(args.episodes), desc='Train episodes')
    current_step = 0
    while current_step < args.episodes:
        for step, batch in enumerate(ppo_trainer.dataloader):
            if current_step >= args.episodes:
                break
            current_step += 1
            query_ids = [t for t in batch["input_ids"]]

            response_tensors = ppo_trainer.generate(
                query_ids,
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
                        query_ids[i],
                        # trl PPO Trainer doesn't support attention_mask, default is ones
                        # attention_mask=query_mask,
                        return_prompt=False,
                        remove_padding=False,
                        **generation_kwargs,
                    )[0]
                no_begin_response_tensors.append(
                    F.pad(
                        r_t[1:],
                        (0, args.max_len - r_t.size(0)),
                        value=tokenizer.pad_token_id
                    )
                )
            no_begin_response_tensors = torch.stack(no_begin_response_tensors)
            generated_attention_mask = (no_begin_response_tensors != tokenizer.pad_token_id).long()

            batch["response"] = tokenizer.batch_decode(no_begin_response_tensors, skip_special_tokens=True,
                                                       clean_up_tokenzization_spaces=True)

            # choose the configured reward modeling strategy
            rewards = []
            response_tokens_length = []
            for i in range(len(batch["query"])):
                response_tokens_length.append(torch.sum(generated_attention_mask[i]).item())
            fg_rewards = get_finegrained_reward(batch["query"], batch["response"], response_tokens_length, tokenizer)

            holistic_rewards = []
            with torch.no_grad():
                for batch_start in range(0, len(batch["query"]), args.rm_max_batch_size):
                    batch_inputs_for_ultra = []
                    for mini_batch_start in range(batch_start,
                                                  min(len(batch["query"]), batch_start + args.rm_max_batch_size)):
                        batch_inputs_for_ultra.append(
                            ultra_template.replace("{instruction}",
                                                   batch["query"][mini_batch_start]).replace(
                                "{completion}", batch["response"][mini_batch_start]))
                    batch_inputs_for_ultra = ultra_tokenizer(
                        batch_inputs_for_ultra,
                        padding=True,
                        max_length=300,
                        truncation=True,
                        return_tensors="pt"
                    ).to(current_device)
                    outputs_ultra = ultra_rm(**batch_inputs_for_ultra)
                    for output_ultra in outputs_ultra:
                        holistic_reward = np.float_(output_ultra.item())
                        # z_normalize
                        holistic_reward = (holistic_reward - args.ultra_mean) / args.ultra_std
                        holistic_rewards.append(holistic_reward)

            for i in range(len(batch["query"])):
                if i in too_short_idx:
                    holistic_rewards[i] = (np.float_(-10))
                    rewards.append(
                        [0] * (response_tokens_length[i] - 1) + [args.holistic_reward_weight * np.float_(-10)])
                    continue
                # deal with too short responses
                holistic_reward = holistic_rewards[i]
                if args.length_reward:
                    fg_rewards["rewards"][i] = [0] * (response_tokens_length[i] - 1) + [
                        np.float_(response_tokens_length[i] / args.length_mean)]
                if args.reward_type == "holistic_only":
                    rewards.append(
                        [0] * (response_tokens_length[i] - 1) + [args.holistic_reward_weight * holistic_reward])
                elif args.reward_type == "aspect_only":
                    rewards.append(fg_rewards["rewards"][i])
                elif args.reward_type == "weighted_sum":
                    fg_rewards["rewards"][i][-1] += args.holistic_reward_weight * holistic_reward
                    rewards.append(fg_rewards["rewards"][i])
                elif args.reward_type == "hierarchical":
                    if holistic_reward < args.hierarchical_threshold:
                        # will be assigned to the last token
                        rewards.append(
                            [0] * (response_tokens_length[i] - 1) + [args.holistic_reward_weight * holistic_reward])
                    else:
                        fg_rewards["rewards"][i][-1] += args.holistic_reward_weight * holistic_reward
                        rewards.append(fg_rewards["rewards"][i])
            logs = {}
            if accelerator.is_main_process:
                logs.update({
                    "fg_rewards/language_confidence": np.mean([r for r in fg_rewards["language_confidence"]]).item(),
                    "fg_rewards/readability_score": np.mean([r for r in fg_rewards["readability"]]).item(),
                    "fg_rewards/grammar_errors": np.mean([r for r in fg_rewards["grammar_errors"]]).item(),
                    "fg_rewards/error_rate": np.mean([r for r in fg_rewards["error_rate"]]).item(),
                    "fg_rewards/holistic_rewards": np.mean([r.item() for r in holistic_rewards]),
                })
            stats = ppo_trainer.step(
                query_ids,
                [t for t in response_tensors],
                [torch.tensor(r, dtype=torch.float) for r in rewards]
            )
            ppo_trainer.log_stats(
                stats, batch, [np.sum(reward) for reward in rewards]
            )
            if accelerator.is_main_process:
                step_bar.update()
            if current_step % args.eval_interval == 0:
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
                        "max_new_tokens": args.max_len,
                    })
                    for i, eval_batch in tqdm(enumerate(eval_dataloader)):
                        query_ids = [t for t in eval_batch["input_ids"]]

                        response_tensors = ppo_trainer.generate(
                            query_ids,
                            return_prompt=False,
                            **eval_generation_kwargs,
                        )
                        eval_length.extend([len(response) for response in response_tensors])

                        no_begin_response_tensors = []
                        for r_t in response_tensors:
                            no_begin_response_tensors.append(
                                F.pad(
                                    r_t[1:],
                                    (0, args.max_len - r_t.size(0)),
                                    value=tokenizer.pad_token_id
                                )
                            )
                        no_begin_response_tensors = torch.stack(no_begin_response_tensors)
                        generated_attention_mask = (no_begin_response_tensors != tokenizer.pad_token_id).long()

                        eval_batch["response"] = tokenizer.batch_decode(no_begin_response_tensors,
                                                                        skip_special_tokens=True,
                                                                        clean_up_tokenzization_spaces=True)
                        rewards = []
                        response_tokens_length = []
                        for i in range(len(eval_batch["query"])):
                            response_tokens_length.append(torch.sum(generated_attention_mask[i]).item())
                        fg_rewards = get_finegrained_reward(eval_batch["query"], eval_batch["response"],
                                                            response_tokens_length, tokenizer)
                        holistic_rewards = []
                        for batch_start in range(0, len(eval_batch["query"]), args.rm_max_batch_size):
                            batch_inputs_for_ultra = []
                            for mini_batch_start in range(batch_start, min(len(eval_batch["query"]),
                                                                           batch_start + args.rm_max_batch_size)):
                                batch_inputs_for_ultra.append(
                                    ultra_template.replace("{instruction}",
                                                           eval_batch["query"][mini_batch_start]).replace(
                                        "{completion}", eval_batch["response"][mini_batch_start]))
                            batch_inputs_for_ultra = ultra_tokenizer(
                                batch_inputs_for_ultra,
                                padding=True,
                                max_length=300,
                                truncation=True,
                                return_tensors="pt"
                            ).to(current_device)
                            outputs_ultra = ultra_rm(**batch_inputs_for_ultra)
                            for output_ultra in outputs_ultra:
                                holistic_reward = np.float_(output_ultra.item())
                                # z_normalize
                                holistic_reward = (holistic_reward - args.ultra_mean) / args.ultra_std
                                holistic_rewards.append(holistic_reward)

                        for i in range(len(eval_batch["query"])):
                            holistic_reward = holistic_rewards[i]
                            if args.length_reward:
                                fg_rewards["rewards"][i] = [0] * (response_tokens_length[i] - 1) + [
                                    np.float_(response_tokens_length[i] / args.length_mean)]
                            if args.reward_type == "holistic_only":
                                rewards.append(
                                    [0] * (response_tokens_length[i] - 1) + [
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
                                        [0] * (response_tokens_length[i] - 1) + [
                                            args.holistic_reward_weight * holistic_reward])
                                else:
                                    fg_rewards["rewards"][i][-1] += args.holistic_reward_weight * holistic_reward
                                    rewards.append(fg_rewards["rewards"][i])
                        if eval_rewards:
                            eval_fg_rewards["grammar_errors"] += fg_rewards["grammar_errors"]
                            eval_fg_rewards["language_confidence"] += fg_rewards["language_confidence"]
                            eval_fg_rewards["readability"] += fg_rewards["readability"]
                            eval_fg_rewards["error_rate"] += fg_rewards["error_rate"]
                            eval_holistic_rewards += holistic_rewards
                            eval_rewards += rewards
                        else:
                            eval_fg_rewards = fg_rewards
                            eval_holistic_rewards = holistic_rewards
                            eval_rewards = rewards
                    eval_grammar_errors = np.mean([r for r in eval_fg_rewards["grammar_errors"]]).item()
                    eval_language_confidence = np.mean([r for r in eval_fg_rewards["language_confidence"]]).item()
                    eval_readability = np.mean([r for r in eval_fg_rewards["readability"]]).item()
                    eval_error_rate = np.mean([r for r in eval_fg_rewards["error_rate"]]).item()
                    eval_holistic_rewards = np.mean([r.item() for r in eval_holistic_rewards])
                    eval_rewards = np.mean([np.sum(reward) for reward in eval_rewards]).item()
                    eval_response_length = np.mean(eval_length)

                    eval_grammar_errors = torch.tensor(eval_grammar_errors).to(current_device)
                    eval_language_confidence = torch.tensor(eval_language_confidence).to(current_device)
                    eval_readability = torch.tensor(eval_readability).to(current_device)
                    eval_error_rate = torch.tensor(eval_error_rate).to(current_device)
                    eval_holistic_rewards = torch.tensor(eval_holistic_rewards).to(current_device)
                    eval_rewards = torch.tensor(eval_rewards).to(current_device)
                    eval_response_length = torch.tensor(eval_response_length).to(current_device)

                    if ppo_trainer.is_distributed:
                        import torch.distributed as dist

                        dist.barrier()
                        dist.all_reduce(eval_readability)
                        dist.all_reduce(eval_holistic_rewards)
                        dist.all_reduce(eval_rewards)
                        dist.all_reduce(eval_response_length)
                        dist.all_reduce(eval_grammar_errors)
                        dist.all_reduce(eval_language_confidence)
                        dist.all_reduce(eval_error_rate)

                    if accelerator.is_main_process:
                        num_proc = accelerator.num_processes
                        logs.update({
                            "eval/grammar_errors": eval_grammar_errors / num_proc,
                            "eval/language_confidence": eval_language_confidence / num_proc,
                            "eval/readability_score": eval_readability / num_proc,
                            "eval/error_rate": eval_error_rate / num_proc,
                            "eval/holistic_rewards": eval_holistic_rewards / num_proc,
                            "eval/rewards": eval_rewards / num_proc,
                            "eval/response_length": eval_response_length / num_proc
                        })
                ppo_trainer.save_pretrained(Path(args.save_dir) / f"step_{current_step}")
            if accelerator.is_main_process:
                wandb.log(logs)


def test():
    if accelerator.is_main_process:
        run = wandb.init(
            project="Hierarchical-RLHF-MT",
            name=args.run_name,
            config={
                "model_ckpt": args.policy_ckpt,
                "seed": args.seed,
                "generation_kwargs": {
                    "do_sample": False,
                    "num_beams": 1
                },
                "max_len": args.max_len,
            }
        )
    # set seed
    set_seed(args.seed)

    # initialize policy and value model tokenizers
    tokenizer = AutoTokenizer.from_pretrained(args.policy_ckpt,
                                              model_max_length=args.max_len)
    tokenizer.padding_side = "right"
    tokenizer.max_input_len = args.max_len
    tokenizer.max_generated_len = args.max_len

    # Load data
    log_info(f'Loading data ...')

    raw_dataset = load_dataset("opus/europarl", name="en-es")["train"]
    raw_dataset = raw_dataset.select(range(100000))
    # split train_dataset into train and eval
    raw_dataset = raw_dataset.train_test_split(test_size=0.3, seed=42)
    if args.test_on_train:
        test_dataset = raw_dataset["train"]
    else:
        test_dataset = raw_dataset["test"]
    test_dataset = MTDataset(test_dataset, tokenizer)

    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size,
                                 shuffle=False, drop_last=False, collate_fn=test_dataset.collator)

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

    ultra_tokenizer = LlamaTokenizer.from_pretrained("openbmb/UltraRM-13b")
    ultra_tokenizer.pad_token_id = ultra_tokenizer.eos_token_id
    ultra_rm = LlamaRewardModel.from_pretrained("openbmb/UltraRM-13b", torch_dtype=torch.bfloat16, device_map={
        "": current_device
    })
    ultra_template = "Human: {instruction}\nAssistant: {completion}"

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
        "max_new_tokens": args.max_len,
    })
    test_gather_list = []
    with torch.no_grad():
        for i, test_batch in tqdm(enumerate(test_dataloader)):
            query_inputs = {
                "input_ids": test_batch["input_ids"],
                "attention_mask": [torch.ones_like(element) for element in test_batch["input_ids"]]
            }
            padded_inputs = tokenizer.pad(
                query_inputs,
                padding=True,
                return_tensors="pt"
            ).to(current_device)

            response_tensors = policy.generate(
                **padded_inputs,
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
                        (0, args.max_len - r_t.size(0)),
                        value=tokenizer.pad_token_id
                    )
                )
            no_begin_response_tensors = torch.stack(no_begin_response_tensors)
            generated_attention_mask = (no_begin_response_tensors != tokenizer.pad_token_id).long()

            test_batch["response"] = tokenizer.batch_decode(no_begin_response_tensors, skip_special_tokens=True,
                                                            clean_up_tokenzization_spaces=True)
            rewards = []
            response_tokens_length = []
            for i in range(len(test_batch["query"])):
                response_tokens_length.append(torch.sum(generated_attention_mask[i]).item())
            fg_rewards = get_finegrained_reward(test_batch["query"], test_batch["response"],
                                                response_tokens_length, tokenizer)
            holistic_rewards = []
            for batch_start in range(0, len(test_batch["query"]), args.rm_max_batch_size):
                batch_inputs_for_ultra = []
                for mini_batch_start in range(batch_start,
                                              min(len(test_batch["query"]), batch_start + args.rm_max_batch_size)):
                    batch_inputs_for_ultra.append(
                        ultra_template.replace("{instruction}", test_batch["query"][mini_batch_start]).replace(
                            "{completion}", test_batch["response"][mini_batch_start]))
                batch_inputs_for_ultra = ultra_tokenizer(
                    batch_inputs_for_ultra,
                    padding=True,
                    max_length=300,
                    truncation=True,
                    return_tensors="pt"
                ).to(current_device)
                outputs_ultra = ultra_rm(**batch_inputs_for_ultra)
                for output_ultra in outputs_ultra:
                    holistic_reward = np.float_(output_ultra.item())
                    # z_normalize
                    holistic_reward = (holistic_reward - args.ultra_mean) / args.ultra_std
                    holistic_rewards.append(holistic_reward)

            for i in range(len(test_batch["query"])):
                holistic_reward = holistic_rewards[i]
                if args.length_reward:
                    fg_rewards["rewards"][i] = [0] * (response_tokens_length[i] - 1) + [
                        np.float_(response_tokens_length[i] / args.length_mean)]
                if args.reward_type == "holistic_only":
                    rewards.append(
                        [0] * (response_tokens_length[i] - 1) + [args.holistic_reward_weight * holistic_reward])
                elif args.reward_type == "aspect_only":
                    rewards.append(fg_rewards["rewards"][i])
                elif args.reward_type == "weighted_sum":
                    fg_rewards["rewards"][i][-1] += args.holistic_reward_weight * holistic_reward
                    rewards.append(fg_rewards["rewards"][i])
                elif args.reward_type == "hierarchical":
                    if holistic_reward < args.hierarchical_threshold:
                        # will be assigned to the last token
                        rewards.append(
                            [0] * (response_tokens_length[i] - 1) + [args.holistic_reward_weight * holistic_reward])
                    else:
                        fg_rewards["rewards"][i][-1] += args.holistic_reward_weight * holistic_reward
                        rewards.append(fg_rewards["rewards"][i])
                test_gather_list.append({
                    "query": test_batch["query"][i],
                    "generation": test_batch["response"][i],
                    "holistic_reward": holistic_reward,
                    "readability": fg_rewards["readability"][i],
                    "language_confidence": fg_rewards["language_confidence"][i],
                    "grammar_errors": fg_rewards["grammar_errors"][i],
                    "error_rate": fg_rewards["error_rate"][i],
                    "response_length": response_tensors_length[i]
                })
            if test_rewards:
                test_fg_rewards["grammar_errors"] += fg_rewards["grammar_errors"]
                test_fg_rewards["language_confidence"] += fg_rewards["language_confidence"]
                test_fg_rewards["readability"] += fg_rewards["readability"]
                test_fg_rewards["error_rate"] += fg_rewards["error_rate"]
                test_holistic_rewards += holistic_rewards
                test_rewards += rewards
            else:
                test_fg_rewards = fg_rewards
                test_holistic_rewards = holistic_rewards
                test_rewards = rewards
    test_grammar_errors = np.mean([r for r in test_fg_rewards["grammar_errors"]]).item()
    test_language_confidence = np.mean([r for r in test_fg_rewards["language_confidence"]]).item()
    test_readability = np.mean([r for r in test_fg_rewards["readability"]]).item()
    test_error_rate = np.mean([r for r in test_fg_rewards["error_rate"]]).item()
    test_holistic_rewards = np.mean([r.item() for r in test_holistic_rewards])
    test_rewards = np.mean([np.mean(reward) for reward in test_rewards]).item()
    test_response_length = np.mean(test_length)

    test_grammar_errors = torch.tensor(test_grammar_errors).to(current_device)
    test_language_confidence = torch.tensor(test_language_confidence).to(current_device)
    test_readability = torch.tensor(test_readability).to(current_device)
    test_error_rate = torch.tensor(test_error_rate).to(current_device)
    test_holistic_rewards = torch.tensor(test_holistic_rewards).to(current_device)
    test_rewards = torch.tensor(test_rewards).to(current_device)
    test_response_length = torch.tensor(test_response_length).to(current_device)

    if accelerator.use_distributed:
        import torch.distributed as dist

        dist.barrier()
        dist.all_reduce(test_readability)
        dist.all_reduce(test_holistic_rewards)
        dist.all_reduce(test_rewards)
        dist.all_reduce(test_response_length)
        dist.all_reduce(test_grammar_errors)
        dist.all_reduce(test_language_confidence)
        dist.all_reduce(test_error_rate)
        gathered_list = accelerator.gather_for_metrics(test_gather_list)
    if accelerator.is_main_process:
        num_proc = accelerator.num_processes
        logs.update({
            "test/grammar_errors": test_grammar_errors / num_proc,
            "test/language_confidence": test_language_confidence / num_proc,
            "test/readability_score": test_readability / num_proc,
            "test/error_rate": test_error_rate / num_proc,
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
