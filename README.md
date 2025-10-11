# activation_checkpointing
My activation checkpointing algorithm from scratch

## Run the exp
```bash
# Setup uv packet manager
pip install uv

uv sync

# Default params
uv run python run.py

# Custom params
uv run python run.py --B 1 --L_list 2048 4096 8192

# Run the tensoboard
uv run tensorboard --logdir baseline --port 6006

# SSH tunnel on local machine
ssh -L 6006:localhost:6006 root@localhost -p 8022
```

## Tensorboard is avaliable here
http://localhost:6006/#pytorch_profiler


