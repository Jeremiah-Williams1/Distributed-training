# Multi-instance notes

- This version uses explicit `MASTER_ADDR` and `MASTER_PORT` instead of relying only on `localhost`.
- Use `gloo` for CPU-only local multi-process testing.
- `NCCL` is typically preferred for GPU workloads on a cluster.
- Confirm firewall and port access before running across machines.
