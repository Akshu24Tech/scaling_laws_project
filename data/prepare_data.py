from datasets import load_dataset
from transformers import AutoTokenizer

def setup_data(cache_dir='./data/cache'):
    print('Setting up dataset...')
    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=cache_dir)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format("torch")
    
    print("Dataset setup complete.")
    return tokenized_datasets, tokenizer

if __name__ == '__main__':
    setup_data()