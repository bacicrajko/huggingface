from datasets import load_dataset
from transformers import AutoTokenizer

model_name = 'microsoft/Phi-3-mini-4k-instruct'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, trust_remote_code=True)
dataset = load_dataset(path='JM-Lee/Phi-3-mini-128K-instruct_instruction', split="train")

def merge_columns(examples):
    # Merge the 'system', 'instruction', and 'response' columns into one column
    merged_texts = [s + i + r for s, i, r in zip(examples['system'], examples['instruction'], examples['response'])]
    return {'merged_text': merged_texts}

# Apply the merge function to create a new column
dataset = dataset.map(merge_columns, batched=True)

# Remove the old columns
dataset = dataset.remove_columns(['system', 'instruction', 'response'])

# Now tokenize the merged column
def tokenize_function(examples):
    # Tokenize the new merged column
    return tokenizer(examples['merged_text'], padding="max_length", truncation=True)

# Apply tokenization
tokenized_ds = dataset.map(tokenize_function, batched=True)

# Save the tokenized dataset
tokenized_ds.save_to_disk('./dataset')

# Print the shape of the new dataset
print(f"Number of examples in the new dataset: {len(tokenized_ds)}")
print(f"Columns in the new dataset: {tokenized_ds.column_names}")
