"""
Training script for GAHR text-based person re-identification.

Usage:
  Single GPU:
    python train.py --config s.config.yaml

  Multi-GPU (distributed):
    torchrun --nproc_per_node=4 train.py --config s.config.yaml

Before running:
  1. Download ViT-B-16.pt from https://openaipublic.blob.core.windows.net/clip/models/5806e77cd80f8b59890b7e101eabd169114205853ae88228be3956d58d5361a2/ViT-B-16.pt
  2. Update s.config.yaml:
     - model.checkpoint: path to ViT-B-16.pt
     - anno_dir: path to annotation directory (with data_captions_train.json etc.)
     - image_dir: path to image directory
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import yaml

from misc.utils import setup_seed, is_using_distributed, is_main_process, get_rank, get_world_size
from model.clip_model import clip_vitb
from model import lorentz as L
from dataset import (
    TrainDataset,
    TestImageDataset,
    TestTextDataset,
    train_collate_fn,
    build_train_transform,
    build_test_transform,
    _load_annotations,
)


# ---------------------------------------------------------------------------
# Config helper (dot-notation access for YAML dicts)
# ---------------------------------------------------------------------------
class Config:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)

    def __repr__(self):
        return str(self.__dict__)


def load_config(path):
    with open(path) as f:
        return Config(yaml.safe_load(f))


# ---------------------------------------------------------------------------
# Distributed init
# ---------------------------------------------------------------------------
def init_distributed(config):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(
            backend=config.distributed.backend,
            init_method=config.distributed.url,
            rank=rank,
            world_size=world_size,
        )
        torch.cuda.set_device(local_rank)
        config.device = local_rank
        print(f"[Rank {rank}] Distributed init done (world_size={world_size})")
    else:
        gpu = getattr(config, "device", 0)
        torch.cuda.set_device(gpu)
        config.device = gpu
        print(f"Single-GPU mode on cuda:{gpu}")


# ---------------------------------------------------------------------------
# CLIP weight loading
# ---------------------------------------------------------------------------
def load_clip_weights(model, config):
    ckpt_path = config.model.checkpoint
    if not os.path.isfile(ckpt_path):
        print(f"WARNING: checkpoint not found at {ckpt_path}, training from scratch")
        return model

    print(f"Loading CLIP checkpoint from {ckpt_path} ...")

    if config.model.ckpt_type == "saved":
        # Previously saved GAHR checkpoint — keys match directly
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded saved checkpoint. missing={msg.missing_keys[:5]}... unexpected={msg.unexpected_keys[:5]}...")
        return model

    # Original CLIP checkpoint (JIT or state_dict)
    try:
        jit_model = torch.jit.load(ckpt_path, map_location="cpu")
        state_dict = jit_model.state_dict()
    except Exception:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

    # Remap keys: text-encoder keys get 'encode_text.' prefix
    skip_keys = {"input_resolution", "context_length", "vocab_size"}
    new_sd = {}
    for key, val in state_dict.items():
        if key in skip_keys:
            continue
        if key.startswith("visual."):
            new_sd[key] = val
        elif key == "logit_scale":
            new_sd[key] = val
        else:
            new_sd[f"encode_text.{key}"] = val

    # Partial load (new modules like BiCrossAttention stay randomly initialized)
    model_sd = model.state_dict()
    loaded, skipped = 0, 0
    for key in new_sd:
        if key in model_sd and new_sd[key].shape == model_sd[key].shape:
            model_sd[key] = new_sd[key]
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(model_sd)
    print(f"  Loaded {loaded} params from CLIP, skipped {skipped}, "
          f"new params {len(model_sd) - loaded}")
    return model


# ---------------------------------------------------------------------------
# Optimizer & scheduler
# ---------------------------------------------------------------------------
def build_optimizer(model, config):
    new_param_keywords = [
        "Bi_cross_attention", "predictor", "local_logit_scale",
        "temp", "curv", "visual_alpha", "textual_alpha",
    ]
    new_params, pretrained_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in new_param_keywords):
            new_params.append(param)
        else:
            pretrained_params.append(param)

    lr = config.schedule.lr
    ratio = config.schedule.ratio_factor
    optimizer = torch.optim.AdamW(
        [
            {"params": pretrained_params, "lr": lr},
            {"params": new_params, "lr": lr * ratio},
        ],
        weight_decay=config.schedule.weight_decay,
        betas=tuple(config.schedule.betas),
        eps=config.schedule.eps,
    )
    return optimizer


def build_scheduler(optimizer, config, steps_per_epoch):
    total_steps = config.schedule.epoch * steps_per_epoch
    warmup_steps = config.schedule.epoch_warmup * steps_per_epoch
    lr = config.schedule.lr
    lr_start = config.schedule.lr_start
    lr_end = config.schedule.lr_end

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return lr_start / lr + (1.0 - lr_start / lr) * step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return lr_end / lr + 0.5 * (1.0 - lr_end / lr) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, epoch, config):
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    alpha_max = config.model.softlabel_ratio
    total_epochs = config.schedule.epoch
    print_period = config.log.print_period

    t0 = time.time()
    for step, batch in enumerate(loader):
        # Alpha ramps linearly over training
        global_step = epoch * n_batches + step
        total_steps = total_epochs * n_batches
        alpha = alpha_max * min(1.0, global_step / total_steps)

        losses = model(batch, alpha)
        loss = sum(losses.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if is_main_process() and (step + 1) % print_period == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            parts = " | ".join(f"{k}: {v.item():.4f}" for k, v in losses.items())
            elapsed = time.time() - t0
            print(
                f"  Epoch [{epoch}] Step [{step+1}/{n_batches}]  "
                f"loss: {loss.item():.4f} ({parts})  "
                f"lr: {lr_now:.2e}  alpha: {alpha:.3f}  "
                f"time: {elapsed:.0f}s"
            )

    avg_loss = total_loss / n_batches
    if is_main_process():
        print(f"  Epoch [{epoch}] avg_loss: {avg_loss:.4f}  time: {time.time()-t0:.0f}s")
    return avg_loss


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, image_loader, text_loader, config):
    model.eval()
    m = model.module if hasattr(model, "module") else model
    curv = m.curv

    # Extract image features
    image_feats, image_ids = [], []
    for images, ids in tqdm(image_loader, desc="Encoding images", disable=not is_main_process()):
        images = images.to(config.device)
        feats = m.encode_image(images, curv)
        image_feats.append(feats.cpu())
        image_ids.extend(ids if isinstance(ids, list) else ids.tolist())
    image_feats = torch.cat(image_feats, dim=0)
    image_ids = torch.tensor(image_ids)

    # Extract text features
    text_feats, text_ids = [], []
    for tokens, ids in tqdm(text_loader, desc="Encoding texts", disable=not is_main_process()):
        tokens = tokens.to(config.device)
        feats = m.encode_text(tokens, curv)
        text_feats.append(feats.cpu())
        text_ids.extend(ids if isinstance(ids, list) else ids.tolist())
    text_feats = torch.cat(text_feats, dim=0)
    text_ids = torch.tensor(text_ids)

    # Compute similarity via negative Lorentz distance
    _curv = curv.exp().cpu()
    # Process in chunks to avoid OOM for large test sets
    chunk_size = 256
    n_texts = text_feats.size(0)
    n_images = image_feats.size(0)

    # --- Text-to-Image retrieval ---
    t2i_ranks = []
    for i in range(0, n_texts, chunk_size):
        tf = text_feats[i : i + chunk_size]
        sim = -L.pairwise_dist(tf, image_feats, _curv)  # [chunk, n_images]
        sorted_idx = sim.argsort(dim=1, descending=True)
        for j in range(tf.size(0)):
            qid = text_ids[i + j]
            gallery_ids = image_ids[sorted_idx[j]]
            match_positions = (gallery_ids == qid).nonzero(as_tuple=True)[0]
            rank = match_positions[0].item() + 1 if len(match_positions) > 0 else n_images + 1
            t2i_ranks.append(rank)
    t2i_ranks = np.array(t2i_ranks)

    # --- Image-to-Text retrieval ---
    i2t_ranks = []
    for i in range(0, n_images, chunk_size):
        imf = image_feats[i : i + chunk_size]
        sim = -L.pairwise_dist(imf, text_feats, _curv)  # [chunk, n_texts]
        sorted_idx = sim.argsort(dim=1, descending=True)
        for j in range(imf.size(0)):
            qid = image_ids[i + j]
            gallery_ids = text_ids[sorted_idx[j]]
            match_positions = (gallery_ids == qid).nonzero(as_tuple=True)[0]
            rank = match_positions[0].item() + 1 if len(match_positions) > 0 else n_texts + 1
            i2t_ranks.append(rank)
    i2t_ranks = np.array(i2t_ranks)

    def _metrics(ranks):
        return {
            "R@1": (ranks <= 1).mean() * 100,
            "R@5": (ranks <= 5).mean() * 100,
            "R@10": (ranks <= 10).mean() * 100,
        }

    t2i = _metrics(t2i_ranks)
    i2t = _metrics(i2t_ranks)
    print(f"  [t2i] R@1: {t2i['R@1']:.2f}  R@5: {t2i['R@5']:.2f}  R@10: {t2i['R@10']:.2f}")
    print(f"  [i2t] R@1: {i2t['R@1']:.2f}  R@5: {i2t['R@5']:.2f}  R@10: {i2t['R@10']:.2f}")
    return t2i["R@1"]


# ---------------------------------------------------------------------------
# Checkpoint save / resume
# ---------------------------------------------------------------------------
def save_checkpoint(model, optimizer, epoch, best_r1, output_dir, filename="best.pth"):
    m = model.module if hasattr(model, "module") else model
    state = {
        "model": m.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_r1": best_r1,
    }
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    torch.save(state, path)
    print(f"  Saved checkpoint to {path} (R@1={best_r1:.2f})")


def load_resume(model, optimizer, path):
    ckpt = torch.load(path, map_location="cpu")
    m = model.module if hasattr(model, "module") else model
    m.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("best_r1", 0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GAHR Training")
    parser.add_argument("--config", default="s.config.yaml", help="Path to config YAML")
    parser.add_argument("--output_dir", default="output", help="Directory to save checkpoints")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only")
    args = parser.parse_args()

    config = load_config(args.config)
    init_distributed(config)
    setup_seed(config.misc.seed)

    # Download NLTK data for EDA (text augmentation)
    if is_main_process():
        import nltk
        nltk.download("wordnet", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    if is_using_distributed():
        dist.barrier()

    # ---- Data ----
    input_res = tuple(config.experiment.input_resolution)
    train_transform = build_train_transform(input_res)
    test_transform = build_test_transform(input_res)

    train_dataset = TrainDataset(
        config.anno_dir,
        config.image_dir,
        train_transform,
        back_trans=config.experiment.back_trans,
    )
    test_annos = _load_annotations(config.anno_dir, "test")
    test_image_dataset = TestImageDataset(test_annos, config.image_dir, test_transform)
    test_text_dataset = TestTextDataset(test_annos, context_length=config.experiment.text_length)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_using_distributed() else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_collate_fn,
    )
    test_image_loader = DataLoader(
        test_image_dataset,
        batch_size=config.data.test_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
    test_text_loader = DataLoader(
        test_text_dataset,
        batch_size=config.data.test_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    if is_main_process():
        print(f"Train samples: {len(train_dataset)}, "
              f"Test images: {len(test_image_dataset)}, "
              f"Test captions: {len(test_text_dataset)}")

    # ---- Model ----
    model = clip_vitb(config)
    model = load_clip_weights(model, config)
    model = model.cuda()

    if is_using_distributed():
        model = DDP(model, device_ids=[get_rank()], find_unused_parameters=True)

    # ---- Optimizer & Scheduler ----
    optimizer = build_optimizer(
        model.module if is_using_distributed() else model, config
    )
    scheduler = build_scheduler(optimizer, config, len(train_loader))

    start_epoch = 0
    best_r1 = 0.0
    if args.resume:
        start_epoch, best_r1 = load_resume(model, optimizer, args.resume)
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch}, best_r1={best_r1:.2f}")

    # ---- Eval only ----
    if args.eval_only:
        if is_main_process():
            evaluate(model, test_image_loader, test_text_loader, config)
        return

    # ---- Training loop ----
    for epoch in range(start_epoch, config.schedule.epoch):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if is_main_process():
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{config.schedule.epoch - 1}")
            print(f"{'='*60}")

        train_one_epoch(model, train_loader, optimizer, scheduler, epoch, config)

        if is_main_process():
            r1 = evaluate(model, test_image_loader, test_text_loader, config)
            if r1 > best_r1:
                best_r1 = r1
                save_checkpoint(model, optimizer, epoch, best_r1, args.output_dir)
            print(f"  Current R@1: {r1:.2f}  |  Best R@1: {best_r1:.2f}")

        if is_using_distributed():
            dist.barrier()

    if is_main_process():
        print(f"\nTraining complete. Best R@1: {best_r1:.2f}")


if __name__ == "__main__":
    main()
