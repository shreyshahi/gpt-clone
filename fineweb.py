# script to download and process fineweb-edu

from datasets import load_dataset

save_path = "/mnt/datasets/fineweb-edu"

dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    cache_dir=save_path
)

print("Dataset is saved")

