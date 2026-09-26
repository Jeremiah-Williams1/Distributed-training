# Official PyTorch Image with CUDA 12.8
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# Copy your DDP training/benchmark script
COPY multi_instance/ddp-nccl.py .

ENTRYPOINT ["python", "ddp-nccl.py"]