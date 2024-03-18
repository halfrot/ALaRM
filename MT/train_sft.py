import os
import evaluate
import argparse
import wandb
from typing import Dict
import numpy as np
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from accelerate import Accelerator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="google/mt5-base")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./MT/model_ckpts/sft-mt5-base")
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--save_freq", default=3000, type=int)
    parser.add_argument("--run_name", default="mt5-base-sft", type=str)

    return parser.parse_args()


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
        encoded_source_dict = self.tokenizer(
            self.prefix + pair["es"], padding="max_length", return_tensors="pt", truncation=True,
        )
        encoded_label_dict = self.tokenizer(
            pair["en"], padding="max_length", return_tensors="pt", truncation=True,
        )
        return {
            "input_ids": encoded_source_dict["input_ids"].squeeze(0),
            "attention_mask": encoded_source_dict["attention_mask"].squeeze(0),
            # ignore padding token loss in labels
            "labels": [
                (l if l != self.tokenizer.pad_token_id else -100) for l in encoded_label_dict["input_ids"].squeeze(0)
            ]
        }


def train(args):
    current_device = Accelerator().process_index
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_path,
        device_map={
            "": current_device
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, model_max_length=args.max_len)
    raw_dataset = load_dataset("opus/europarl", name="en-es")["train"]
    raw_dataset = raw_dataset.select(range(100000))
    # split train_dataset into train and eval
    raw_dataset = raw_dataset.train_test_split(test_size=0.3, seed=42)
    train_dataset = raw_dataset["train"]
    test_dataset = raw_dataset["test"]
    train_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
    eval_dataset = train_dataset["test"]
    train_dataset = train_dataset["train"]
    train_dataset = MTDataset(train_dataset, tokenizer)
    eval_dataset = MTDataset(eval_dataset, tokenizer)
    test_dataset = MTDataset(test_dataset, tokenizer)

    bleu = evaluate.load("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = [pred[:np.where(pred == tokenizer.eos_token_id)[0][0]] if np.where(pred == tokenizer.eos_token_id)[
                                                                              0].size > 0 else pred for pred in preds]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100s used for padding as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
        return result

    def preprocess_logits_for_metrics(logits, labels):
        """
        Original Trainer may have a memory leak.
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        pred_ids = logits[0].argmax(dim=-1)
        return pred_ids

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.save_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=True if args.bf16 else False,
        seed=args.seed,
        weight_decay=args.weight_decay,
        run_name=args.run_name,
        report_to=["wandb"],
        ddp_find_unused_parameters=False,
        load_best_model_at_end=True,  # there is a bug if load_in_8bit=True
        metric_for_best_model="eval_loss",  # Replace 'your_metric' with the metric you want to use for early stopping
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()

    print("Saving last checkpoint of the model")
    trainer.model.save_pretrained(os.path.join(args.save_dir, "final_checkpoint/"), safe_serialization=False)


if __name__ == "__main__":
    args = get_args()
    if Accelerator().is_main_process:
        run = wandb.init(
            # Set the project where this run will be logged
            project="SFT-MT",
            # Track hyperparameters and run metadata
            name=args.run_name,
            config={
                "lr": args.lr,
                "model_path": args.model_path,
                "output_dir": args.save_dir,
                "batch_size": args.batch_size,
                "eval_freq": args.eval_freq,
                "save_freq": args.save_freq,
                "max_len": args.max_len,
            }
        )
    train(args)
