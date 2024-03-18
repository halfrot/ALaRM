import os
import torch
import argparse
import evaluate
import wandb
import textstat
from accelerate import Accelerator
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
from datasets import load_dataset
from tqdm import trange


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--run_name", default="eval-mt5-base-sft", type=str)

    return parser.parse_args()


def inference(model, tokenizer, inputs_list):
    num = len(inputs_list)
    generated_list = []
    for batch_start in trange(0, num, args.batch_size):
        batch_inputs = inputs_list[batch_start:min(num, batch_start + args.batch_size)]
        batch_inputs = tokenizer(
            batch_inputs,
            padding=True,
            max_length=args.max_len,
            truncation=True,
            return_tensors="pt").to("cuda")
        outputs = model.generate(
            **batch_inputs,
            max_length=args.max_len,
            num_return_sequences=1,
            do_sample=False,
            num_beams=1
        )
        generated_list.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return generated_list


def eval(args):
    current_device = Accelerator().process_index
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_path,
        device_map={
            "": current_device
        }
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, model_max_length=args.max_len)

    raw_dataset = load_dataset("opus/europarl", name="en-es")["train"]
    raw_dataset = raw_dataset.select(range(100000))
    # split train_dataset into train and eval
    dataset = raw_dataset.train_test_split(test_size=0.3, seed=42)["test"]
    sample_list = [sample for sample in dataset]
    prefix = "translate Spanish to English: "
    with Accelerator().split_between_processes(sample_list) as sub_sample_list:
        inputs_list = [prefix + sample["translation"]["es"] for sample in sub_sample_list]
        pred = inference(model, tokenizer, inputs_list)
        ref = [[sample["translation"]["en"]] for sample in sub_sample_list]
    bleu = evaluate.load("sacrebleu")
    bleu_score = torch.tensor(bleu.compute(predictions=pred, references=ref)["score"]).to(current_device)
    tot_DC = 0
    tot_FKG = 0
    tot_FRE = 0
    for single_pred in pred:
        tot_DC += textstat.dale_chall_readability_score(single_pred)
        tot_FKG += textstat.flesch_kincaid_grade(single_pred)
        tot_FRE += textstat.flesch_reading_ease(single_pred)
    DC_score = torch.tensor(tot_DC / len(pred)).to(current_device)
    FKG_score = torch.tensor(tot_FKG / len(pred)).to(current_device)
    FRE_score = torch.tensor(tot_FRE / len(pred)).to(current_device)

    import torch.distributed as dist

    dist.barrier()
    dist.all_reduce(bleu_score)
    dist.all_reduce(DC_score)
    dist.all_reduce(FKG_score)
    dist.all_reduce(FRE_score)
    if Accelerator().is_main_process:
        wandb.log({
            "bleu": bleu_score / Accelerator().num_processes,
            "DC": DC_score / Accelerator().num_processes,
            "FKG": FKG_score / Accelerator().num_processes,
            "FRE": FRE_score / Accelerator().num_processes
        })


if __name__ == "__main__":
    args = get_args()
    if Accelerator().is_main_process:
        run = wandb.init(
            project="SFT-MT",
            name=args.run_name,
            config={
                "model_path": args.model_path,
                "dataset_name": args.dataset_name,
                "max_len": args.max_len,
                "batch_size": args.batch_size,
                "fp16": args.fp16,
            }
        )
    eval(args)
