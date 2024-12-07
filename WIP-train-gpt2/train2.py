import os
import random
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from datasets import load_dataset
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logging.info("Starting")

if os.path.exists('trained-gpt2'):
    model = GPT2LMHeadModel.from_pretrained('trained-gpt2')
    logging.info("Loaded model from trained-gpt2")
else:
    config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12
    )
    model = GPT2LMHeadModel(config)
    model.apply(model._init_weights)
    logging.info("Initialized model with random weights")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

chunk_size = 1024

# Load dataset
# dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
dataset = load_dataset("pszemraj/simple_wikipedia_LM", "default")
train_tokens = tokenizer(
    [text for text in dataset['train']['text'][:10000] if text.strip()],
    truncation=True,
    max_length=chunk_size,
    return_tensors='pt',
    padding=False,
)['input_ids'].flatten()
train_data = train_tokens
logging.info(f"Dataset size: {len(train_data)/1e6:.2f}M tokens")

# Training parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
epochs = 10
gradient_accumulation_steps = 16

# Training loop
tokens_processed = 0
for epoch in range(epochs):
    total_epoch_loss = 0
    num_epoch_chunks = 0
    # Create chunks of fixed length
    num_chunks = len(train_data) // chunk_size
    chunk_indices = list(range(num_chunks))
    random.shuffle(chunk_indices)
    for i in tqdm(range(num_chunks)):
        # Get single chunk
        chunk_start = chunk_indices[i] * chunk_size
        chunk = train_data[chunk_start:chunk_start + chunk_size].to(device)
        chunk = chunk.unsqueeze(0)  # Add batch dimension
        
        # Create attention mask (1 for all tokens since we have no padding)
        attention_mask = torch.ones_like(chunk).to(device)
        
        # Forward pass
        outputs = model(
            input_ids=chunk,
            attention_mask=attention_mask,
            labels=chunk
        )
        loss = outputs.loss / gradient_accumulation_steps
        total_epoch_loss += loss.item()
        num_epoch_chunks += 1

        # Backward pass
        loss.backward()
        tokens_processed += chunk_size
        if (i + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
            logging.info(f'step={i} loss={loss.item():.4f} epoch_loss={total_epoch_loss / num_epoch_chunks:.4f} seen={tokens_processed/1e6:.2f}M')
    # Generate sample text at end of epoch
            model.eval()
            with torch.no_grad():
                # Start with prompt
                prompt = "In May 8, 1998, "
                context = tokenizer(prompt, return_tensors="pt").input_ids[:, :-1].to(device)
                
                # Generate 100 tokens
                output = model.generate(
                    context,
                    max_length=100,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.9
                )
                
                # Decode and print the generated text
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                logging.info(f"\nGenerated sample (epoch {epoch}):\n{generated_text}\n")
    
    model.train()
    

# Save model
model.save_pretrained('trained-gpt2')
tokenizer.save_pretrained('trained-gpt2')
