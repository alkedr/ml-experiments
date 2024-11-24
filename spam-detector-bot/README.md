# spam-detector-bot

Telegram bot that tells you whether a given message is spam.

Uses python-llama-cpp with GPU acceleration and quantized Qwen 2.5 7B Instruct. The prompt is written in Russian and tested mostly on messages in Russian.

## Usage

Download the model manually and put it in this directory.

To launch the bot:
```
TELEGRAM_TOKEN="..." ./run.sh python3 main.py
```

To run tests:
```
./run.sh pytest
```

`run.sh` is a script that:
1. Installs some (not all) dependencies using apt. Notably missing are CMake and CUDA.
2. Creates a venv.
3. Runs the command passed to it.

`run.sh` was written for my own environment (old Ubuntu 20.04 with CUDA 12 already installed), you may need to change it.
