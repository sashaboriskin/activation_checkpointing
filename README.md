# activation_checkpointing
My activation checkpointing algorithm from scratch

## Run the exp
```bash
# Setup uv packet manager
pip install uv
uv sync

# Run the baseline
uv run python run.py --method checkpoint --profile_dir checkpoint/

# Run the activation checkpoint version
uv run python run.py --method checkpoint --profile_dir checkpoint/

# Run with the concrete batch_size and seq_len_list
uv run python run.py --B 1 --L_list 2048 4096 8192

# Run the tensoboard
uv run tensorboard --logdir baseline --port 6006

# SSH tunnel on local machine
ssh -L 6006:localhost:6006 root@localhost -p 8022
```

## Tensorboard is avaliable here
http://localhost:6006/#pytorch_profiler


## GPU pick memory comparison

| seq_len | batch_size | Baseline (MB) | Checkpoint (MB) | 
|---:|---:|---:|---:|
| 2048  | 1 | 3603.4565 | 3001.5044 | 
| 4096  | 1 | 5097.8940 | 3017.5044 | 
| 8192  | 1 | 8086.7690 | 3053.9893 | 
| 12288 | 1 | 11075.6440 | 3390.3955 |
| 16384 | 1 | 14064.5190 | 3726.8018 | 

## Memory profiler L=16k, B=1


### Baseline
![Baseline L=16k B=1](img/baseline_b1_L16k.jpg) 

### With activation checkpointing
![Checkpoint L=16k B=1](img/checkpoint_b1_L16k.jpg) 