import os
import random
import shutil
import argparse

def split_data(source_directory, train_ratio, seed):
    random.seed(seed)

    files = [file for file in os.listdir(source_directory) if file.endswith('.txt')]
    random.shuffle(files)

    total_files = len(files)
    train_end = int(total_files * train_ratio)
    remaining_files = files[train_end:]
    split_index = len(remaining_files) // 2  # split half of the remaining files into validation and test sets
    
    train_files = files[:train_end]
    val_files = remaining_files[:split_index]
    test_files = remaining_files[split_index:]

    # Copy files to new directories
    train_dir = os.path.join(source_directory, 'train')
    val_dir = os.path.join(source_directory, 'val')
    test_dir = os.path.join(source_directory, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for file in train_files:
        shutil.copy(os.path.join(source_directory, file), train_dir)
    for file in val_files:
        shutil.copy(os.path.join(source_directory, file), val_dir)
    for file in test_files:
        shutil.copy(os.path.join(source_directory, file), test_dir)

    # Print file names for each dataset
    print("Training files:", train_files)
    print("Validation files:", val_files)
    print("Testing files:", test_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../data/10TargetAuthors/")
    parser.add_argument("--author", type=str, default="Samuel Richardson")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of training data")
    parser.add_argument("--seed", type=int, default=1006, help="Random seed for shuffling")
    args = parser.parse_args()

    split_data(args.data_dir + args.author, args.train_ratio, args.seed)
