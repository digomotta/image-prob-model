from datasets import load_dataset

# Load dataset split
ds = load_dataset("rod-motta/wpp-images-enriched", split="train")
print(ds)