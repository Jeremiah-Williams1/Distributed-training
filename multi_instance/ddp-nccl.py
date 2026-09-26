import os
import torch
import torch.distributed as dist


def main():
    # torchrun sets these env vars for us before the script even starts.
    # This is exactly what Kubeflow will set for us later, just by hand for now.
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(f"[rank {rank}] starting up. world_size={world_size}")

    # nccl = NVIDIA's GPU-to-GPU collective comms library, bundled with the
    # CUDA build of torch. This call is the "find each other" step - every
    # process blocks here until all world_size processes have checked in.
    dist.init_process_group(backend="nccl")

    # One GPU per process here (--nproc_per_node=1), so local rank 0 always
    # means "this process's one GPU." Pin this process to it explicitly -
    # NCCL needs this even when there's only one GPU to choose from.
    torch.cuda.set_device(0)

    print(f"[rank {rank}] process group initialized. all processes found each other.")

    # Each rank starts with a different number - stand-in for "each worker
    # computed a different gradient from its slice of the data".
    # .cuda() moves it onto this process's GPU - without this, NCCL has
    # nothing to actually exercise and the tensor just sits on CPU.
    my_value = torch.tensor([float(rank + 1) * 10]).cuda()
    print(f"[rank {rank}] before all-reduce: {my_value.item()}")

    # This is the actual operation DDP runs after every training step, just
    # on a single number here instead of millions of gradient values.
    dist.all_reduce(my_value, op=dist.ReduceOp.SUM)

    print(f"[rank {rank}] after all-reduce (sum across all ranks): {my_value.item()}")

    dist.destroy_process_group()
    print(f"[rank {rank}] done.")


if __name__ == "__main__":
    main()
