import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def main():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    dist.init_process_group(backend="gloo")

    model = torch.nn.Linear(10, 10)
    model = DDP(model)
    x = torch.randn(4, 10)
    y = model(x)

    loss = y.sum()
    loss.backward()

    print(f"[rank {rank}] training step complete, world_size={world_size}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
