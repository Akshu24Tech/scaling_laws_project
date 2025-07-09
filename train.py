# In train.py
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import GPT2Config, get_scheduler
from data.prepare_data import setup_data
from models.transformer import SimpleLanguageModel

def train(config_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup data
    dataset, tokenizer = setup_data()
    train_loader = torch.utils.data.DataLoader(dataset['train'], batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset['validation'], batch_size=8)

    # Model configuration
    model_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=512,
        **config['model']
    )
    model = SimpleLanguageModel(model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Get non-embedding param count (N)
    n_params = model.count_non_embedding_params()
    print(f"Training model with {n_params/1e6:.2f}M non-embedding parameters.")

    # Optimizer and Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']))
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(config['training']['epochs']):
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids # In language modeling, the input is the label

            outputs = model(input_ids, attention_mask=attention_mask)
            
            # Shift logits and labels for next-token prediction
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids

            outputs = model(input_ids, attention_mask=attention_mask)
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.item()
    
    final_val_loss = total_loss / len(val_loader)
    print(f"Final Validation Loss: {final_val_loss}")
    
    # IMPORTANT: Save the results for analysis
    with open(f"analysis/results.csv", "a") as f:
        f.write(f"{n_params},{final_val_loss}\n")

if __name__ == '__main__':
    # Run experiments for each config
    # Make sure results.csv is empty before starting
    with open("analysis/results.csv", "w") as f:
        f.write("params,loss\n")
        
    train('configs/small_model.yaml')
    train('configs/medium_model.yaml')
    # Add more training runs here for other configs     