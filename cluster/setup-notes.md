# Cluster Setup Notes — Project 4

## Constraint driving the whole sequence

AWS GPU vCPU quota (G/VT family) = 8. Two standalone `g4dn.xlarge`
instances (4 vCPU each) already use all 8. EKS GPU nodes draw from the
*same* quota. Conclusion: standalone instances and EKS GPU nodes can
never both be running at once — must stop one before scaling up the
other.

## Order of operations

1. Build + push the training image (doesn't need any GPU quota — just
   needs Docker, can run on either standalone instance):
   ```bash
   docker build -t ddp-nccl:v1 .
   aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
   aws ecr create-repository --repository-name ddp-nccl --region <region>
   docker tag ddp-nccl:v1 <account-id>.dkr.ecr.<region>.amazonaws.com/ddp-nccl:v1
   docker push <account-id>.dkr.ecr.<region>.amazonaws.com/ddp-nccl:v1
   ```

2. Create the cluster — GPU node group starts at 0, so this doesn't
   touch the GPU quota yet:
   ```bash
   eksctl create cluster -f cluster/eksctl-cluster.yaml
   ```
   Takes ~15-20 min (mostly the EKS control plane). Confirm after:
   ```bash
   kubectl get nodes
   ```
   Should show exactly 1 node (the `t3.medium` system node).

3. Install Kubeflow Trainer (v2) control plane + CRDs — runs on the
   system node, no GPU needed for this step.
   *(commands to add once we get here)*

4. **Before scaling up GPU nodes:** stop both standalone EC2 instances.
   ```bash
   aws ec2 stop-instances --instance-ids <instance-1-id> <instance-2-id>
   ```

5. Scale the GPU node group up:
   ```bash
   eksctl scale nodegroup --cluster=ddp-training-cluster --name=gpu-ng --nodes=2 --nodes-min=0 --nodes-max=2
   kubectl get nodes -o wide   # wait for both to show Ready
   kubectl get pods -n kube-system | grep nvidia   # confirm device plugin picked them up
   ```
   If `nvidia.com/gpu` doesn't show up under `kubectl describe node
   <gpu-node>` → Capacity, the NVIDIA device plugin daemonset needs
   installing manually — flag this if it happens, don't assume it's
   automatic.

6. Apply the `TrainJob` (once written).

7. **After the run**, scale GPU nodes back to 0 to stop paying / free
   the quota:
   ```bash
   eksctl scale nodegroup --cluster=ddp-training-cluster --name=gpu-ng --nodes=0
   ```
   Restart the standalone instances if/when needed again:
   ```bash
   aws ec2 start-instances --instance-ids <instance-1-id> <instance-2-id>
   ```

## Open items to fill in as we go

- Confirm region in `eksctl-cluster.yaml` matches actual account setup
- Record actual `eksctl create cluster` runtime + any errors hit
- Confirm whether NVIDIA device plugin installs automatically or needs
  a manual `kubectl apply`
