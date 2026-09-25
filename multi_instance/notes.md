# Multi-Instance Notes — Bare EC2, No Cluster

Two `g4dn.xlarge` instances, same VPC/subnet, same security group
(self-referencing rule). Instance 1 = master/rank 0, instance 2 = rank 1.

## Security group

Self-referencing inbound rule on the shared security group:
- Initially: Custom TCP, port `29500` only (for torchrun rendezvous)
- Widened to: Custom TCP, port range `0-65535`, source = the security
  group itself — needed because gloo/NCCL open a second, dynamically
  chosen port for the actual data channel beyond the rendezvous port.
  Safe here since it's scoped to just these two instances, not the
  internet.
- ICMP (ping) intentionally left blocked — not needed, `nc -zv` used for
  connectivity checks instead.

## Environment

```bash
conda create -n ddp python=3.13 -y
conda activate ddp
pip install torch --index-url https://download.pytorch.org/whl/cpu   # first pass, CPU
# later, for the GPU pass:
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu128
conda env export --no-builds > environment.yml
```
`environment.yml`'s `pip:` block needed `--extra-index-url
https://download.pytorch.org/whl/cu128` (or `/cpu` for the earlier pass)
added manually as the first line — `conda env create`'s pip step only
checks PyPI by default and won't find the `+cu128`/`+cpu` build tag
otherwise.

## Connectivity check (before running anything distributed)

```bash
sudo dnf install -y nmap-ncat   # nc not preinstalled on this AMI
nc -zv <other-instance-private-ip> 29500
```
"Connection refused" = good (path open, nothing listening yet). Use
private IPs, not public — same-VPC traffic should stay internal.

## CPU/gloo cross-instance run

Instance 1 (master):
```bash
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
  --master_addr=<instance-1-private-ip> --master_port=29500 \
  ddp-gloo.py
```
Instance 2:
```bash
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \
  --master_addr=<instance-1-private-ip> --master_port=29500 \
  ddp-gloo.py
```

## GPU/NCCL cross-instance run

Same `torchrun` args as above, just pointed at `ddp-nccl.py`, plus:
```bash
export NCCL_DEBUG=INFO
```
on both instances before launching, to see NCCL's transport selection.

**Result:** NCCL tried OFI/Libfabric (AWS EFA) → failed, no EFA hardware
on `g4dn.xlarge`. Tried InfiniBand → no device found. Fell back to
Socket (plain TCP over `ens5`) → worked. Final line before results:
`... use ring PXN 0 GDR 0` — GDR (GPUDirect RDMA) off, meaning both
directions crossed host CPU memory rather than going GPU-to-GPU
directly. Concrete evidence of the no-NVLink/no-EFA cost.

Both runs converged correctly: rank 0 = 10, rank 1 = 20, all-reduce sum
= 30 on both ranks, in both the gloo and NCCL passes.
