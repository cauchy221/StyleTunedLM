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

from presidio_analyzer import AnalyzerEngine
import spacy
from spacy.pipeline import EntityRuler


analyzer = AnalyzerEngine()
analyzer.nlp_engine.nlp['en'].max_length = 3500000

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 3500000

ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})
# You could add other names or entities here
patterns = [{"label": "PERSON", "pattern": name} for name in ["Bingo", "Little", "Emsworth", "Bingley"]]
ruler.add_patterns(patterns)

# Process the text
doc = nlp("I met Bingo and Alice at the park with Bob.")
for ent in doc.ents:
    print(ent.text, ent.label_)


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
    
    def identify_names(text):
        results = analyzer.analyze(text=text, entities=["PERSON"], language="en")
        name_positions_1 = [(res.start, res.end) for res in results]

        doc = nlp(text)
        name_positions_2 = [(ent.start_char, ent.end_char) for ent in doc.ents if ent.label_ == "PERSON"]

        # merge the two lists
        name_positions = name_positions_1 + name_positions_2
        name_positions = list(set(name_positions))
        return name_positions
    
    def create_attention_mask(inputs, token_based_name_positions, start_index):
        mask = torch.ones_like(inputs['input_ids'][0], dtype=torch.long)
        for start_token, end_token in token_based_name_positions:
            adjusted_start = max(start_token - start_index, 0)
            adjusted_end = min(end_token - start_index, max_length - 1)
            if adjusted_start <= adjusted_end:
                mask[adjusted_start:adjusted_end + 1] = 0
        return mask
    
    def convert_char_spans_to_token_spans(text, char_spans, tokenizer):
        encoded = tokenizer(text, return_offsets_mapping=True)
        token_spans = []
        for start_char, end_char in tqdm(char_spans, total=len(char_spans)):
            start_token = next((i for i, (start, end) in enumerate(encoded['offset_mapping']) if start <= start_char < end), None)
            end_token = next((i for i, (start, end) in enumerate(encoded['offset_mapping']) if start < end_char <= end), None)
            if start_token is not None and end_token is not None:
                token_spans.append((start_token, end_token))
        return token_spans
    
    print(f"Reading {mode} data...")
    
    raw_text = read_from_dir(os.path.join(data_dir, mode))
    if mode == "train":
        m = 70000
    else:
        m = 10000
    cleaned_text = clean_text(raw_text).split()[:m]
    # print(f"Number of words: {len(cleaned_text)}")
    cleaned_text = " ".join(cleaned_text)
    tokens = tokenizer.encode(cleaned_text)
    # print(f"Number of tokens: {len(tokens)}")

    name_positions = identify_names(cleaned_text)
    token_based_name_positions = convert_char_spans_to_token_spans(cleaned_text, name_positions, tokenizer)

    if mode == "train":
        tokens = tokens[:80000]
    elif mode == "test":
        tokens = tokens[:20000]

    segments = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    encoded_segments = []

    start_index = 0
    for seg in tqdm(segments, total=len(segments)):
        seg = torch.tensor(seg)  # Convert the segment to a tensor
        seg_inputs = {'input_ids': seg.unsqueeze(0)}
        attention_mask = torch.ones_like(seg).unsqueeze(0)

        inputs = {
            'input_ids': seg_inputs['input_ids'],
            'attention_mask': attention_mask
        }

        custom_attention_mask = create_attention_mask(inputs, token_based_name_positions, start_index)
        attention_mask = inputs['attention_mask'] * custom_attention_mask

        encoded_segment = {
            'input_ids': inputs['input_ids'][0],
            'attention_mask': attention_mask[0],
        }
        encoded_segments.append(encoded_segment)
        start_index += seg.size(0)  # Increment by number of tokens in the segment

    print(f"Number of segments: {len(encoded_segments)}")
    return MyDataset(encoded_segments[:-1])

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

            # set masked tokens to -100
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            # reset attention mask to be all ones
            attention_mask = torch.ones_like(attention_mask)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
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
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            attention_mask = torch.ones_like(attention_mask)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
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
    parser.add_argument("--author", type=str, default="P. G. Wodehouse")
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

    # print(args)

    # accelerator
    accelerator = Accelerator()
    args.device = accelerator.device
    print(f"Device: {args.device}")

    if accelerator.is_main_process:
        wandb.init(project="your-project", entity="your-name", config=args, name=f"{args.author}_masking_{int(100*args.percentage)}")

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
    # print_trainable_parameters(model)

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
            os.path.join(args.weights, f"{args.author}_masking_{int(100*args.percentage)}"), 
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        wandb.finish()