"""
Inspect Dataset Structure
Quick script to check the structure of the medical_abstracts dataset.
"""

from datasets import load_dataset

print("Loading dataset...")
dataset = load_dataset("TimSchopf/medical_abstracts", split='train')

print(f"\nTotal records: {len(dataset)}")
print(f"\nColumn names: {dataset.column_names}")
print(f"\nFirst record:")
print(dataset[0])
print(f"\nFirst 3 records (sample):")
for i in range(min(3, len(dataset))):
    print(f"\n--- Record {i} ---")
    for key, value in dataset[i].items():
        print(f"{key}: {value[:100] if isinstance(value, str) and len(value) > 100 else value}")
