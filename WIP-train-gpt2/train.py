import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader 
from tqdm import tqdm
import logging
import os
from datasets import load_dataset

# Training hyperparameters
batch_size = 1
learning_rate = 6e-4
max_iters = 100000
eval_interval = 500
eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(data, tokenizer, batch_size):
    # Get random chunks from data
    max_length = tokenizer.model_max_length
    ix = torch.randint(len(data) - max_length, (batch_size,))
    x = torch.stack([data[i:i+max_length] for i in ix])
    y = torch.stack([data[i+1:i+max_length+1] for i in ix])
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = torch.ones_like(x)
    return x.to(device), y.to(device), attention_mask.to(device)

def save_checkpoint(model, tokenizer, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logging.info(f"Saved checkpoint to {save_dir}")

def train(model, tokenizer, train_data):
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    for iter in tqdm(range(max_iters)):
        if iter % eval_interval == 0:
            model.eval()
            
            losses = []
            with torch.no_grad():
                for k in range(eval_iters):
                    X, Y, attention_mask = get_batch(train_data, tokenizer, batch_size)
                    outputs = model(X, attention_mask=attention_mask, labels=Y)
                    losses.append(outputs.loss.item())

            tokens_seen = iter * batch_size * tokenizer.model_max_length
            logging.info(f"step {iter}: train loss {sum(losses)/len(losses):.4f}, tokens seen: {tokens_seen/1e6:.2f}M")

            with torch.no_grad():
                # Generate sample text
                prompt = "Once upon a time"
                context = torch.tensor([tokenizer.encode(prompt)], device=device)
                context_attention_mask = torch.ones_like(context, device=device)
                generated = model.generate(
                    context,
                    attention_mask=context_attention_mask,
                    max_length=100,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.8
                )
                generated_text = tokenizer.decode(generated[0])
                logging.info(f"Generated text:\n{generated_text}")
            
            model.train()
        
        # Sample batch and forward pass
        X, Y, attention_mask = get_batch(train_data, tokenizer, batch_size)
        outputs = model(X, attention_mask=attention_mask, labels=Y)
        loss = outputs.loss
        
        # Backward pass and optimize
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logging.info("Starting")

    # Load and tokenize wikitext-2 dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Tokenize and concatenate all texts
    train_texts = dataset['train']['text']
    train_tokens = []
    for text in train_texts:
        if text.strip():  # Skip empty lines
            tokens = tokenizer.encode(text)
            train_tokens.extend(tokens)
    
    train_data = torch.tensor(train_tokens, dtype=torch.long, device=device)
    logging.info(f"Dataset size: {len(train_data)/1e6:.2f}M tokens")

    # Sample and decode a random sequence from training data
    seq_length = 100
    start_idx = torch.randint(0, len(train_data) - seq_length, (1,)).item()
    sample_seq = train_data[start_idx:start_idx + seq_length]
    sample_text = tokenizer.decode(sample_seq)
    logging.info(f"Sample from training data:\n{sample_text}")

    # Initialize model and tokenizer
    config = GPT2Config(
        vocab_size=50257,  # GPT-2 vocabulary size
        n_positions=1024,   # Reduced from 1024
        n_ctx=1024,        # Reduced from 1024
        n_embd=768,       # Reduced from 768
        n_layer=12,        # Reduced from 12
        n_head=12,         # Reduced from 12
    )
    
    model = GPT2LMHeadModel(config)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model weights from scratch (don't load pretrained weights)
    model.apply(model._init_weights)
    model = model.to(device)
    logging.info(f"Model size: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    try:
        train(model=model, tokenizer=tokenizer, train_data=train_data)
        # Save final checkpoint
        save_checkpoint(model, tokenizer)
    except KeyboardInterrupt:
        logging.info("Training interrupted, saving checkpoint...")
        save_checkpoint(model, tokenizer)
        raise
