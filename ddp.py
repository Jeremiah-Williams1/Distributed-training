import os
import torch
import torch.distributed as dist


def main():
    # torchrun sets these env vars for us before the script even starts.
    # This is exactly what Kubeflow will set for us later, just by hand for now.
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(f"[rank {rank}] starting up. world_size={world_size}")

    # gloo = CPU-friendly backend. This call is the "find each other" step -
    # every process blocks here until all world_size processes have checked in.
    dist.init_process_group(backend="gloo")

    print(f"[rank {rank}] process group initialized. all processes found each other.")

    # Each rank starts with a different number - stand-in for "each worker
    # computed a different gradient from its slice of the data".
    my_value = torch.tensor([float(rank + 1) * 10])
    print(f"[rank {rank}] before all-reduce: {my_value.item()}")

    # This is the actual operation DDP runs after every training step, just
    # on a single number here instead of millions of gradient values.
    dist.all_reduce(my_value, op=dist.ReduceOp.SUM)

    print(f"[rank {rank}] after all-reduce (sum across all ranks): {my_value.item()}")

    dist.destroy_process_group()
    print(f"[rank {rank}] done.")


if __name__ == "__main__":
    main()