import argparse
from tqdm import tqdm
import os
import random
import re

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
import wandb


class MyDataset(Dataset):
    def __init__(self, encoded_segments):
        self.encoded_segments = encoded_segments

    def __len__(self):
        return len(self.encoded_segments)

    def __getitem__(self, idx):
        return self.encoded_segments[idx]


def print_trainable_parameters(model):
    if accelerator.is_main_process:
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

def get_dataset(data_dir, mode, tokenizer, max_length):
    def clean_text(text):
        replacements = {
            "\n": " ", 
            "’": "'", 
            "“": "\"", 
            "”": "\"",
            "—": "-",
            "…": "...",
            "‘": "'",
        }
        regex = re.compile('|'.join(re.escape(key) for key in replacements.keys()))
        text = regex.sub(lambda match: replacements[match.group(0)], text.strip())
        text = re.sub(r"[_*\[\]]", "", text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def read_from_dir(data_dir):
        text = ""
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                    text += "\n\n" + f.read()
        return text
    
    print(f"Reading {mode} data...")
    
    raw_text = read_from_dir(os.path.join(data_dir, mode))
    cleaned = clean_text(raw_text)
    tokens = tokenizer.encode(cleaned)
    if mode == "train":
        tokens = tokens[:80000]
    if mode == "test":
        tokens = tokens[:20000]

    segments = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    encoded_segments = [tokenizer.prepare_for_model(seg, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt') for seg in segments]

    return MyDataset(encoded_segments)

def evaluate_model(model, test_dl):
    model.eval()
    total_loss = 0

    if accelerator.is_main_process:
        test_iter = tqdm(test_dl, desc="Evaluating")
    else:
        test_iter = test_dl

    for batch in test_iter:
        with torch.no_grad():
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item()
    
    avg_eval_loss = total_loss / len(test_dl)
    if accelerator.is_main_process:
        print(f"Average Evaluation Loss: {avg_eval_loss}")
        wandb.log({"evaluation_loss": avg_eval_loss})

def train_model(model, train_dl, test_dl, epochs, optimizer):
    model.train()
    print("Start training...")

    for epoch in tqdm(range(epochs)):
        if accelerator.is_main_process:
            train_iter = tqdm(train_dl, desc=f"Epoch {epoch + 1}")
        else:
            train_iter = train_dl

        total_loss = 0
        for batch in train_iter:
            optimizer.zero_grad()

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            accelerator.backward(loss)
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dl)
        if accelerator.is_main_process:
            print(f"Average Training Loss: {avg_train_loss}")
            wandb.log({"train_loss": avg_train_loss})

        evaluate_model(model, test_dl)
        # torch.cuda.empty_cache()

    print("Training finished...")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--author", type=str, default="Mark Twain")
    parser.add_argument("--model", type=str, default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--tokenizer", type=str, default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--data_dir", type=str, default="../../data/10TargetAuthors/")
    parser.add_argument("--cache_dir", type=str, default="../../cache/")
    parser.add_argument("--weights", type=str, default="../../weights/10TargetAuthors/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_token_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1006)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--percentage", type=float, default=1.0)
    args = parser.parse_args()

    # accelerator
    accelerator = Accelerator()
    args.device = accelerator.device
    print(f"Device: {args.device}")

    if accelerator.is_main_process:
        wandb.init(project="your-project", entity="your-name", config=args, name=f"{args.author}_{int(100*args.percentage)}")

    # seed
    seed_everything(args.seed)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # lora config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type='CAUSAL_LM',
    )

    # model
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        device_map=args.device,
        do_sample=True,
        use_cache=True,
        cache_dir=args.cache_dir
    )
    
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    # data
    train_dataset = get_dataset(os.path.join(args.data_dir, args.author), "train", tokenizer, args.max_token_length)

    # use percentage to control data size
    if args.percentage < 1.0:
        train_dataset = torch.utils.data.Subset(train_dataset, torch.randperm(len(train_dataset))[:int(args.percentage * len(train_dataset))])

    test_dataset = get_dataset(os.path.join(args.data_dir, args.author), "test", tokenizer, args.max_token_length)

    # save dataset for later
    torch.save(train_dataset, os.path.join(args.data_dir, f"{args.author}/train_{int(100*args.percentage)}.pt"))
    if args.percentage == 1.0:
        torch.save(test_dataset, os.path.join(args.data_dir, f"{args.author}/test.pt"))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # prepare through accelerator
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(model, optimizer, train_dataloader, test_dataloader)

    # training loop
    train_model(model, train_dataloader, test_dataloader, epochs=args.epochs, optimizer=optimizer)

    # unwrap and save
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            os.path.join(args.weights, f"{args.author}_{int(100*args.percentage)}"), 
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        wandb.finish()