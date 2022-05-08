## Vision Transformer (ViT) model using PyTorch/XLA FSDP

This repo implements sharded training of a Vision Transformer (ViT) model using the FSDP algorithm.

**Warning: This repository is still experimental. Use at your own risk.**

---

### Installation

1. Allocate a v3-128 TPU VM pod (e.g. with name `rh-128-0` in zone `europe-west4-a`) from the `tpu-vm-pt-1.10` environment. You can also try out larger TPU pods such as v3-256 or v3-512.

2. Install the nightly 20220505 wheels for `torch`, `torchvision`, and `torch_xla`, and also the nightly `dev20220413` wheel for libtpu (you need to use these exact versions as older versions won't work). Also install `timm` and install the [XLA FSDP PR](https://github.com/pytorch/xla/pull/3431).

```bash
TPU_NAME=rh-128-0  # change to your TPU name
ZONE=europe-west4-a  # change to your TPU zone

gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} \
  --worker all \
  --command "
# torch, torchvision and torch_xla 20220505
sudo pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch-nightly+20220505-cp38-cp38-linux_x86_64.whl
sudo pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torchvision-nightly+20220505-cp38-cp38-linux_x86_64.whl
sudo pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-nightly+20220505-cp38-cp38-linux_x86_64.whl

# libtpu 20220413
sudo pip3 install https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/wheels/libtpu-nightly/libtpu_nightly-0.1.dev20220413-py3-none-any.whl

# ViT dependency
sudo pip3 install timm==0.4.12

# install the FSDP PR into PyTorch/XLA
cd ~ && rm -rf xla_fsdp_dev && git clone https://github.com/ronghanghu/xla.git xla_fsdp_dev
cd xla_fsdp_dev && git checkout xla_fsdp_rebased
sudo rm -rf /usr/local/lib/python3.8/dist-packages/torch_xla/distributed/fsdp
sudo cp -r ./torch_xla/distributed/fsdp /usr/local/lib/python3.8/dist-packages/torch_xla/distributed/

# clone this repo ViT FSDP example
cd ~ && rm -rf vit_10b_fsdp_example && git clone https://github.com/ronghanghu/vit_10b_fsdp_example.git
"
```

3. Download [ImageNet-1k](https://image-net.org/) to a shared directory (e.g. to `/datasets/imagenet-1k`) that can be accessed from all nodes, which should have the following structure (the validation images moved to labeled subfolders, following the [PyTorch ImageNet example](https://github.com/pytorch/examples/tree/master/imagenet#requirements)).
```
/datasets/imagenet-1k
|_ train
|  |_ <n0......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-N-name>.JPEG
|  |_ ...
|  |_ <n1......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-M-name>.JPEG
|  |  |_...
|  |  |_...
|_ val
|  |_ <n0......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-N-name>.JPEG
|  |_ ...
|  |_ <n1......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-M-name>.JPEG
|  |  |_...
|  |  |_...
```
You can use a Persistent Disk or a Filestore NFS on GCP to store the ImageNet-1k dataset.

### Running the experiments

1. Now log into your TPU VM.
```bash
TPU_NAME=rh-128-0  # change to your TPU name
ZONE=europe-west4-a  # change to your TPU zone

gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} --worker 0
```

2. Before running any experiments, first set up the gcloud ssh configuration on your TPM VM as follows (*only need to do it once*):
```bash
cd ${HOME} && gcloud compute config-ssh --quiet
```

3. Now we can run the experiments. For example, to train a ViT model with 10 billion parameters (5120 embed dim, 32 attention heads, 32 layers, and an MLP ratio of 4.0 that gives 20480 = 5120 * 4.0 feed-forward MLP dim), you can launch the following in a tmux session.
```bash
TPU_NAME=rh-128-0  # change to your TPU name
SAVE_DIR=~/vit_10b_fsdp_example_ckpts  # this can be any directory (it doesn't need to be a shared one across nodes)

mkdir -p ${SAVE_DIR}
cd ${HOME} && python3 -m torch_xla.distributed.xla_dist \
  --tpu=${TPU_NAME} --restart-tpuvm-pod-server --env PYTHONUNBUFFERED=1 -- \
python3 -u ~/vit_10b_fsdp_example/run_vit_training.py \
  --data_dir /datasets/imagenet-1k \
  --ckpt_dir ${SAVE_DIR} \
  --image_size 224 \
  --patch_size 14 \
  --embed_dim 5120 \
  --mlp_ratio 4.0 \
  --num_heads 32 \
  --num_blocks 32 \
  --batch_size 1024 \
  --num_epochs 300 \
  --lr 1e-3 \
  --weight_decay 0.1 \
  --clip_grad_norm 1.0 \
  --warmup_steps 10000 \
  --log_step_interval 20 \
  2>&1 | tee ${SAVE_DIR}/stdout_stderr_$(date +%Y-%m-%d_%H-%M-%S).log
```
Note that these hyperparameters (e.g. learning rate) are not necessarily optimal and you may need to tweak them to get the best performance. You can also use `--fake_data` to run on fake datasets (dummy images filled with all zeros). As a comparison, you can pass `--run_without_fsdp` to launch without FSDP, which can only fit much smaller model sizes.

You can also try running on models larger than the 10 billion size above. In general, you will need more TPU cores to fit more parameters. Don't worry if you see messages like `tcmalloc: large alloc 1677729792 bytes == 0x181ff4000` when trying to run this codebase on even larger models (e.g. 60B parameters) -- this message is [not an error](https://stackoverflow.com/questions/52351611/is-tcmalloc-large-alloc-a-warning-or-error-in-python). You can get rid of it by passing `--env TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=4294967296` in `torch_xla.distributed.xla_dist` to raise the tcmalloc report threshold to e.g. 4 GB.

In a few very large models (e.g. 120B parameters), sometimes one may encounter [TPU smem limit error](https://github.com/pytorch/xla/issues/3453#issuecomment-1083482546). If this happens, you can try passing `--mark_step_after_forward` to insert an `xm.mark_step` call between the forward and the backward pass, or try manually adding `xm.mark_step` between different layers in the model.
