import logging
import os
import time
import math
import pickle

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from model import GPTConfig, GPT


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logging.info("Starting")


# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-shakespeare-char'
# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

data_dir = 'data/shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False # do we use bias inside LayerNorm and Linear layers?

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 20
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 100 # not super necessary potentially
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually

# system
device_type = 'cuda'
device = 'cuda'


os.makedirs(out_dir, exist_ok=True)

tb_writer = SummaryWriter(comment=f"shakespeare")

torch.manual_seed(1337)

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    data = np.memmap(
        os.path.join(data_dir, f'{split}.bin'),
        dtype=np.uint16, 
        mode='r',
    )
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
    x = x.pin_memory().to(device, non_blocking=True)
    y = y.pin_memory().to(device, non_blocking=True)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    logging.info(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    logging.info("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        logging.info("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    logging.info(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    logging.info(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


with torch.profiler.profile(profile_memory=True, record_shapes=True) as prof:
    # training loop
    X, Y = get_batch('train') # fetch the very first batch
    running_mfu = -1.0
    while True:
        with torch.profiler.record_function(f"step{iter_num}"):
            iteration_start_time = time.time()

            # determine and set the learning rate for this iteration
            lr = get_lr(iter_num) if decay_lr else learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            tb_writer.add_scalar("lr", lr, iter_num)

            if (iter_num+1) % 20 == 0:
                with torch.profiler.record_function("validation_loss"):
                    with torch.no_grad():
                        model.eval()
                        _, loss = model(*get_batch('val'))
                        tb_writer.add_scalar("val/loss", loss.item(), iter_num)
                        model.train()
                    torch.cuda.synchronize()

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            step_loss = 0.0
            for micro_step in range(gradient_accumulation_steps):
                with torch.profiler.record_function("forward"):
                    logits, loss = model(X, Y)
                    loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
                    torch.cuda.synchronize()
                with torch.profiler.record_function("backward"):
                    # step_loss += loss.item()
                    # immediately async prefetch next batch while model is doing the forward pass on the GPU
                    X, Y = get_batch('train')
                    loss.backward()
                    torch.cuda.synchronize()
            tb_writer.add_scalar("train/loss", step_loss, iter_num)
            # clip the gradient
            if grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # step the optimizer
            optimizer.step()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            torch.cuda.synchronize()
            # timing and logging
            if (iter_num+1) % 1 == 0:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                # lossf = loss.item() * gradient_accumulation_steps
                # if local_iter_num >= 5: # let the training loop settle a bit
                #     mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                #     tb_writer.add_scalar("mfu_percent", mfu*100, iter_num)
                #     running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                logging.info(f"iter {iter_num}: loss {step_loss:.4f}, time {time.time() - iteration_start_time:.3f}s")
            iter_num += 1

            tb_writer.add_scalar("Time/step", time.time() - iteration_start_time, iter_num)

            # termination conditions
            if iter_num > max_iters:
                break

prof.export_chrome_trace(f"{out_dir}/trace.json")