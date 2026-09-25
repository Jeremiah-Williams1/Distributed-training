import os
import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29500")

    print(f"[rank {rank}] starting up with master_addr={master_addr}, master_port={master_port}")

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )

    value = torch.tensor([float(rank + 1) * 10])
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    print(f"[rank {rank}] summed value: {value.item()}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
