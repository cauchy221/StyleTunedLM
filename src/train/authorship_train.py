import argparse
import random
import os
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses, SentencesDataset, models
from sentence_transformers.evaluation import TripletEvaluator, BinaryClassificationEvaluator
import re
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb


class Loss_with_logging(torch.nn.Module):
    def __init__(self, loss_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_model = loss_model
        
    def forward(self, *args, **kwargs):
        loss = self.loss_model(*args, **kwargs)
        wandb.log({"loss": loss.item()})
        return loss


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

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

def get_author_works(args, split):
    path = args.data_dir
    author_works_dict = {}

    for author in os.listdir(path):
        if author != "prompts" and author[-3:] != ".pt":
            texts = ""
            paragraphs = []
            author_path = os.path.join(path, author, split)
            for work in os.listdir(author_path):
                with open(os.path.join(author_path, work), "r") as f:
                    text = f.read()
                    text = clean_text(text)
                    texts += text
            for i in range(0, len(texts), args.max_sequence_length):
                paragraphs.append(texts[i:i+args.max_sequence_length])
            author_works_dict[author] = paragraphs

    return author_works_dict

def generate_balanced_pairwise_examples(size, author_texts):
    authors = list(author_texts.keys())
    positive_pairs = []
    negative_pairs = []

    # Generate positive pairs (same author)
    for author, texts in author_texts.items():
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                positive_pairs.append((texts[i], texts[j], 1))

    # Generate negative pairs (different authors)
    for i in tqdm(range(len(authors)), desc="Generating negative pairs"):
        for j in range(i + 1, len(authors)):
            texts_i = author_texts[authors[i]]
            texts_j = author_texts[authors[j]]
            random.shuffle(texts_i)
            random.shuffle(texts_j)
            count = min(len(texts_i), len(texts_j))
            for k in range(count):
                negative_pairs.append((texts_i[k], texts_j[k], 0))

    # Balance the pairs
    min_pairs = min(len(positive_pairs), len(negative_pairs))
    positive_pairs = random.sample(positive_pairs, min_pairs)[:size]
    negative_pairs = random.sample(negative_pairs, min_pairs)[:size]

    # Shuffle and combine
    pairs = positive_pairs + negative_pairs
    random.shuffle(pairs)
    return pairs

def generate_balanced_triplet_examples(size, author_texts):
    authors = list(author_texts.keys())
    triplets = []

    # First, find the maximum number of triplets possible per author
    min_positive_combinations = float('inf')
    for texts in author_texts.values():
        if len(texts) > 1:
            min_positive_combinations = min(min_positive_combinations, len(texts) * (len(texts) - 1))

    # Generate triplets with balanced positive and negative samples
    for anchor_author in tqdm(authors, desc="Generating triplets"):
        anchor_texts = author_texts[anchor_author]
        other_authors = [author for author in authors if author != anchor_author]
        negatives = []

        # Collect negatives from other authors
        for neg_author in other_authors:
            negatives.extend(author_texts[neg_author])

        # Shuffle negatives to randomize selection later
        random.shuffle(negatives)

        # Generate the triplets
        count = 0
        for i, anchor in enumerate(anchor_texts):
            for j, positive in enumerate(anchor_texts):
                if anchor != positive:
                    negative = negatives[count % len(negatives)]
                    triplets.append((anchor, positive, negative))
                    count += 1
                    if count >= min_positive_combinations:
                        break
            if count >= min_positive_combinations:
                break
    
    random.shuffle(triplets)
    return triplets[:size]

def transform_pairwise_to_input_examples(pairwise_data):
    input_examples = []
    for text1, text2, label in tqdm(pairwise_data, desc="Transforming pairs"):
        example = InputExample(texts=[text1, text2], label=label)
        input_examples.append(example)
    return input_examples

def transform_triplet_to_input_examples(triplet_data):
    input_examples = []
    for anchor, positive, negative in tqdm(triplet_data, desc="Transforming triplets"):
        example = InputExample(texts=[anchor, positive, negative])
        input_examples.append(example)
    return input_examples


if __name__ == "__main__":

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1006)
    parser.add_argument("--model", type=str, default="roberta-base")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_sequence_length", type=int, default=256)
    parser.add_argument("--if_prepare_data", type=bool, default=False)
    parser.add_argument("--mode", type=str, default="triplet")
    parser.add_argument("--data_dir", type=str, default="../../data/10TargetAuthors/")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--weights", type=str, default="../../weights/10TargetAuthors/authorship/")
    parser.add_argument("--max_train_size", type=int, default=100000)
    parser.add_argument("--max_test_size", type=int, default=25000)
    args = parser.parse_args()

    # wandb
    wandb.init(project="your-project", entity="your-name", config=args, name=f"{args.mode}")

    # seed
    seed_everything(args.seed)

    # model
    word_embedding_model = models.Transformer(args.model, max_seq_length=args.max_sequence_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=args.device)
    print("Finish loading model")

    # data
    if args.if_prepare_data:
        author_works_train = get_author_works(args, "train")
        author_works_test = get_author_works(args, "test")

        if args.mode == "pairwise":
            train_examples = generate_balanced_pairwise_examples(args.max_train_size, author_works_train)
            test_examples = generate_balanced_pairwise_examples(args.max_test_size, author_works_test)

            print(f"Train size: {len(train_examples)}, Test size: {len(test_examples)}")

            train_examples = transform_pairwise_to_input_examples(train_examples)
            test_examples = transform_pairwise_to_input_examples(test_examples)

        elif args.mode == "triplet":
            train_examples = generate_balanced_triplet_examples(args.max_train_size, author_works_train)
            test_examples = generate_balanced_triplet_examples(args.max_test_size, author_works_test)

            print(f"Train size: {len(train_examples)}, Test size: {len(test_examples)}")

            train_examples = transform_triplet_to_input_examples(train_examples)
            test_examples = transform_triplet_to_input_examples(test_examples)

        torch.save(train_examples, os.path.join(args.data_dir, f"{args.mode}_train.pt"))
        torch.save(test_examples, os.path.join(args.data_dir, f"{args.mode}_test.pt"))

    else:
        train_examples = torch.load(os.path.join(args.data_dir, f"{args.mode}_train.pt"))
        test_examples = torch.load(os.path.join(args.data_dir, f"{args.mode}_test.pt"))

    # prepare dataloaders
    train_dataset = SentencesDataset(train_examples, model)
    train_dl = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)

    # loss
    if args.mode == "pairwise":
        loss_fct = Loss_with_logging(losses.ContrastiveLoss(model))
    elif args.mode == "triplet":
        loss_fct = Loss_with_logging(losses.TripletLoss(model))

    # evaluator
    if args.mode == "pairwise":
        evaluator = BinaryClassificationEvaluator.from_input_examples(test_examples, 
                                                                      name='Style-pairwise', 
                                                                      show_progress_bar=True)
    elif args.mode == "triplet":
        evaluator = TripletEvaluator.from_input_examples(test_examples, 
                                                         name='Style-triplet', 
                                                         show_progress_bar=True)

    # train
    warmup_steps = int(len(train_dl) * 0.1)
    model.fit(train_objectives=[(train_dl, loss_fct)], 
              evaluator=evaluator, 
              epochs=args.epochs, 
              warmup_steps=warmup_steps, 
              output_path=os.path.join(args.weights, args.mode),
              save_best_model=True,
              checkpoint_path=os.path.join(args.weights, args.mode, "checkpoint"),
              checkpoint_save_steps=len(train_dl),
              checkpoint_save_total_limit=2
            )
    wandb.finish()