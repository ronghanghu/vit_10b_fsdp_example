import argparse
import os
import pprint
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP, checkpoint_module
from timm.models.vision_transformer import (
    Block,
    PatchEmbed,
    _init_vit_weights,
    trunc_normal_,
)

from utils import (
    FakeImageNetDataset,
    SmoothedValue,
    get_warmup_cosine_scheduler,
    save_ckpt,
    load_ckpt,
)


def build_datasets(cfg, device):
    world_size = xm.xrt_world_size()
    rank = xm.get_local_ordinal()

    assert cfg.batch_size % world_size == 0
    local_batch_size = cfg.batch_size // world_size

    if not cfg.fake_data:
        xm.master_print(f"loading images from directory: {cfg.data_dir}")
        train_transform = T.Compose(
            [
                T.RandomResizedCrop(cfg.image_size, interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        train_dataset = torchvision.datasets.ImageFolder(os.path.join(cfg.data_dir, "train"), train_transform)
        val_transform = T.Compose(
            [
                T.Resize((cfg.image_size * 256) // 224, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(cfg.image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        val_dataset = torchvision.datasets.ImageFolder(os.path.join(cfg.data_dir, "val"), val_transform)
    else:
        xm.master_print("loading fake images")
        train_dataset = FakeImageNetDataset(cfg.image_size, 1281167)
        val_dataset = FakeImageNetDataset(cfg.image_size, 50000)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, drop_last=True, shuffle=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    train_loader = pl.MpDeviceLoader(train_loader, device)

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, drop_last=True, shuffle=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=local_batch_size,
        sampler=val_sampler,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = pl.MpDeviceLoader(val_loader, device)
    return (
        train_dataset,
        train_loader,
        train_sampler,
        val_dataset,
        val_loader,
        val_sampler,
    )


class FSDPViTModel(nn.Module):
    """
    To train large models that cannot fit into a single TPU, one should use nested
    FSDP (wrapping sub-modules with inner FSDP when building the entire model).
    This class provides an example for nested FSDP.
    """

    def __init__(
        self,
        image_size,
        patch_size,
        embed_dim,
        num_heads,
        num_blocks,
        mlp_ratio,
        pos_dropout,
        mlp_dropout,
        att_dropout,
        num_classes,
        grad_ckpt_wrap,
        fsdp_wrap,
    ):
        super().__init__()

        # image patch and positional embedding
        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        _init_vit_weights(self.patch_embed)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(pos_dropout)

        # vision transformer blocks
        blocks = []
        for idx in range(num_blocks):
            block = Block(  # using the ViT block from the timm library
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=mlp_dropout,
                attn_drop=att_dropout,
            )
            _init_vit_weights(block)  # note: init module weights BEFORE wrapping with FSDP
            # note: to use gradient checkpointing, wrap the module with gradient checkpointing
            # wrapper BEFORE wrapping it with FSDP
            block = fsdp_wrap(grad_ckpt_wrap(block))
            blocks.append(block)
            xm.master_print(f"built ViT block {idx}")
        self.blocks = nn.Sequential(*blocks)

        # classifier
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        _init_vit_weights(self.norm)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, image):
        x = self.patch_embed(image) + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        # here we use average pooling over image sequence (instead of using [CLS])
        # as in https://arxiv.org/abs/2106.04560
        logits = self.head(torch.mean(self.norm(x), dim=1))
        return logits


def build_fsdp_vit_model(cfg, device):
    """
    Create a ViT model with nested FSDP and gradient checkpointing
    """

    def fsdp_wrap(module):
        # note: to implement ZeRO-3, set `cfg.reshard_after_forward` to True
        return FSDP(
            module.to(device),
            reshard_after_forward=cfg.reshard_after_forward,
            flatten_parameters=cfg.flatten_parameters,
        )

    model = FSDPViTModel(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        num_blocks=cfg.num_blocks,
        mlp_ratio=cfg.mlp_ratio,
        pos_dropout=cfg.pos_dropout,
        mlp_dropout=cfg.mlp_dropout,
        att_dropout=cfg.att_dropout,
        num_classes=cfg.num_classes,
        grad_ckpt_wrap=checkpoint_module if cfg.grad_ckpt else (lambda x: x),
        fsdp_wrap=fsdp_wrap if not cfg.run_without_fsdp else (lambda m: m.to(device)),
    )
    # note: always wrap the base model with an outer (root) FSDP
    # (we don't need to apply gradient checkpointing to the base model)
    model = fsdp_wrap(model)
    return model


def run_logging(epoch, step, smoothed_loss, smoothed_time, loss, lr, device):
    loss_value = loss.item()
    reduced_loss = xm.mesh_reduce("loss_value", loss_value, sum)
    reduced_loss /= xm.xrt_world_size()
    smoothed_loss.update(reduced_loss, batch_size=1)
    xm.master_print(
        f"epoch {epoch} step {(step + 1)}, lr: {lr:.4f}, "
        f"loss: {smoothed_loss.avg:.4f}, "
        f"sec/iter: {smoothed_time.avg:.4f}, "
        f"TPU memory: {xm.get_memory_info(device)}"
    )


def train(cfg):
    batch_size = cfg.batch_size
    num_epochs = cfg.num_epochs
    device = xm.xla_device()
    rank = xm.get_local_ordinal()

    # build datasets
    train_dataset, train_loader, train_sampler, _, val_loader, _ = build_datasets(cfg, device)
    xm.rendezvous("loaded dataset")
    xm.master_print(f"\n=== dataset ===\n{pprint.pformat(train_dataset)}\n")

    # build model and loss
    model = build_fsdp_vit_model(cfg, device)
    loss_fn = torch.nn.CrossEntropyLoss()
    xm.rendezvous("loaded model")
    xm.master_print(f"\n=== model ===\n{pprint.pformat(model)}\n")

    parameters = list(model.parameters())
    xm.master_print(f"per-TPU (sharded) parameter num: {sum(p.numel() for p in parameters)}")

    # build optimizer and scheduler
    optimizer = torch.optim.AdamW(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    lr_scheduler = get_warmup_cosine_scheduler(
        optimizer, warmup_iteration=cfg.warmup_steps, max_iteration=len(train_dataset) // batch_size * num_epochs,
    )
    xm.rendezvous("loaded optimizer")
    xm.master_print(f"\n=== optimizer ===\n{pprint.pformat(optimizer)}\n")

    # resume training from previous checkpoint (in FSDP, each rank saves and loads its own checkpoint)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    if cfg.resume_epoch > 0:
        ckpt_path = os.path.join(cfg.ckpt_dir, f"epoch_{cfg.resume_epoch}_rank_{rank}.ckpt")
        load_ckpt(ckpt_path, model, optimizer, lr_scheduler)

    smoothed_loss = SmoothedValue(window_size=5)
    smoothed_time = SmoothedValue(window_size=5)
    xm.rendezvous("training begins")
    xm.master_print("training begins (the first few iterations are very slow due to compilation)")
    for epoch in range(cfg.resume_epoch + 1, num_epochs + 1):
        xm.master_print(f"starting epoch {epoch}")
        time_epoch_b = time_step_b = time.time()
        model.train()
        train_sampler.set_epoch(epoch)
        for step, (data, target) in enumerate(train_loader):
            # 1. forward pass
            output = model(data)
            loss = loss_fn(output, target)
            if cfg.mark_step_after_forward:
                # A workaround for very large models that exceed the TPU smem limit
                # (see https://github.com/pytorch/xla/issues/3453#issuecomment-1083482546)
                # Based on our internal tests, this is not needed for models with 60B or fewer parameters.
                xm.mark_step()

            # 2. backward pass and gradient clipping (if specified via a positive cfg.clip_grad_norm)
            loss.backward()
            if not cfg.run_without_fsdp:
                # !!! DO NOT reduce (sharded) gradients across XLA devices when using FSDP
                # !!! use `model.clip_grad_norm_` to clip based on full (instead of sharded) gradient's norm
                if cfg.clip_grad_norm > 0:
                    model.clip_grad_norm_(cfg.clip_grad_norm)
            else:
                # the baseline setting without FSDP (as a comparison)
                xm.reduce_gradients(optimizer)
                if cfg.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(parameters)

            # 3. parameter update
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)  # note: set_to_none saves more memory

            # 4. logging
            t_new = time.time()
            time_step_elapsed, time_step_b = t_new - time_step_b, t_new
            smoothed_time.update(time_step_elapsed, batch_size=1)
            is_first_iter = epoch == cfg.resume_epoch + 1 and step == 0
            if is_first_iter or (step + 1) % cfg.log_step_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                xm.add_step_closure(
                    run_logging, args=(epoch, step, smoothed_loss, smoothed_time, loss, lr, device),
                )

        time_epoch_elapsed = time.time() - time_epoch_b
        xm.master_print(f"epoch {epoch} done ({time_epoch_elapsed:.2f} sec)")

        # save checkpoint
        if epoch % cfg.ckpt_epoch_interval == 0 or epoch == num_epochs:
            ckpt_path = os.path.join(cfg.ckpt_dir, f"epoch_{epoch}_rank_{rank}.ckpt")
            save_ckpt(ckpt_path, model, optimizer, lr_scheduler, master_only=False)
        # evaluate on val
        if epoch % cfg.test_epoch_interval == 0 or epoch == num_epochs:
            accuracy, _, _ = eval_on_val(val_loader, model, device)
            xm.master_print(f"accuracy on val: {accuracy:.4f}")


def eval_on_val(val_loader, model, device):
    model.eval()
    local_correct = torch.zeros(1, dtype=torch.long, device=device)
    local_total = 0
    for data, target in val_loader:
        output = model(data)
        pred = output.argmax(dim=-1)
        local_correct.add_(pred.eq(target.view_as(pred)).sum())
        local_total += target.size(0)
    correct = xm.mesh_reduce("local_correct", local_correct.item(), sum)
    total = xm.mesh_reduce("local_total", local_total, sum)
    accuracy = correct / total
    return accuracy, correct, total


def main(device_id, cfg):
    xm.master_print(f"\n=== cfg ===\n{pprint.pformat(cfg)}\n")
    train(cfg)
    xm.master_print("training completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/datasets/imagenet-1k")
    parser.add_argument("--fake_data", action="store_true", dest="fake_data")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ckpt_dir", type=str, default="/tmp/vit_fsdp")
    parser.add_argument("--resume_epoch", type=int, default=0)
    parser.add_argument("--ckpt_epoch_interval", type=int, default=10)
    parser.add_argument("--test_epoch_interval", type=int, default=10)
    parser.add_argument("--log_step_interval", type=int, default=20)

    # the default model hyperparameters is a ViT with 10 billion parameters
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--embed_dim", type=int, default=5120)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=32)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--pos_dropout", type=float, default=0.0)
    parser.add_argument("--att_dropout", type=float, default=0.0)
    parser.add_argument("--mlp_dropout", type=float, default=0.0)
    parser.add_argument("--num_classes", type=int, default=1000)

    # these default learning hyperparameters are not necessarily optimal
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--no_grad_ckpt", action="store_false", dest="grad_ckpt")
    parser.add_argument("--no_reshard_after_forward", action="store_false", dest="reshard_after_forward")
    parser.add_argument("--flatten_parameters", action="store_true", dest="flatten_parameters")
    parser.add_argument("--mark_step_after_forward", action="store_true", dest="mark_step_after_forward")
    parser.add_argument("--run_without_fsdp", action="store_true", dest="run_without_fsdp")

    cfg = parser.parse_args()
    xmp.spawn(main, args=(cfg,))
