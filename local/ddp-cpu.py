import os
import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(f"[rank {rank}] starting up. world_size={world_size}")

    dist.init_process_group(backend="gloo")

    print(f"[rank {rank}] process group initialized.")

    my_value = torch.tensor([float(rank + 1) * 10])
    print(f"[rank {rank}] before all-reduce: {my_value.item()}")

    dist.all_reduce(my_value, op=dist.ReduceOp.SUM)

    print(f"[rank {rank}] after all-reduce (sum across all ranks): {my_value.item()}")

    dist.destroy_process_group()
    print(f"[rank {rank}] done.")


if __name__ == "__main__":
    main()
