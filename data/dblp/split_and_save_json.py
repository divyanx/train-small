from datasets import load_dataset
import os
import argparse

def create_splits(
    dataset_name: str,
    split_name: str,
    size: int,
    seed: int,
    train_frac: float,
    val_frac: float,
    output_dir: str
):
    # 1. Load and optionally truncate
    ds = load_dataset(dataset_name, split=split_name)
    ds = ds.shuffle(seed)
    if size and size < len(ds):
        ds = ds.select(range(size))

    # 2. Compute split indices
    n = len(ds)
    n_train = int(train_frac * n)
    n_val   = int(val_frac   * n)
    n_test  = n - n_train - n_val

    train_ds = ds.select(range(n_train))
    val_ds   = ds.select(range(n_train, n_train + n_val))
    test_ds  = ds.select(range(n_train + n_val, n))

    # 3. Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 4. Save each split as JSONL
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path   = os.path.join(output_dir, "validation.jsonl")
    test_path  = os.path.join(output_dir, "test.jsonl")

    train_ds.to_json(train_path, orient="records", lines=True)
    val_ds.to_json(val_path,   orient="records", lines=True)
    test_ds.to_json(test_path, orient="records", lines=True)

    print(f"Saved {len(train_ds)} train  → {train_path}")
    print(f"Saved {len(val_ds)}   validation → {val_path}")
    print(f"Saved {len(test_ds)}  test → {test_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Split DBLP Discovery into train/val/test JSONL")
    p.add_argument("--size",      type=int,   default=100,  help="Total records to use (N).")
    p.add_argument("--seed",      type=int,   default=108,     help="Random seed for shuffle.")
    p.add_argument("--train_frac",type=float, default=0.8,    help="Fraction for train split.")
    p.add_argument("--val_frac",  type=float, default=0.1,    help="Fraction for validation split.")
    p.add_argument("--out_dir",   type=str,   default="data_splits", help="Where to write JSONL files.")
    args = p.parse_args()

    create_splits(
        dataset_name="jpwahle/dblp-discovery-dataset",
        split_name="train",
        size=args.size,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        output_dir=args.out_dir
    )
