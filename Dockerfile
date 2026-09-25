# Base image with conda already present - matches how we built the env
# on the EC2 instances, so this should need zero dependency surprises.
FROM continuumio/miniconda3

WORKDIR /app

# Same environment.yml both instances used - includes the cu128 torch
# build and the --extra-index-url fix from the bare-instance phase.
COPY environment.yml .
RUN conda env create -f environment.yml

# The already-proven cross-instance NCCL script. Swap this COPY (and the
# CMD below) for training/train_ddp.py once we move to real training.
COPY multi_instance/ddp-nccl.py .

# Bake the env activation into how the container runs, so `python
# ddp-nccl.py` below picks up the right interpreter/packages without
# needing an explicit `conda activate` at runtime (containers don't have
# an interactive shell to run that against).
SHELL ["conda", "run", "--no-capture-output", "-n", "ddp", "/bin/bash", "-c"]
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ddp", "python", "ddp-nccl.py"]
