# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GAHR is a PyTorch research codebase for text-based person re-identification (ReID). It implements a CLIP-based dual-encoder architecture (ViT vision encoder + text transformer) with Lorentzian (hyperbolic) embeddings and multiple contrastive loss objectives.

## Project Structure

- `model/` — All model code lives here
  - `clip_model.py` — Main `CLIP` model class; orchestrates encoders and computes all losses (NITC, RITC, CITC, BAI)
  - `visual_transformer.py` — ViT-B/16 image encoder with patch dropout and Lorentz projection
  - `text_transformer.py` — Text encoder (12-layer transformer, vocab 49408) with Lorentz projection
  - `base_transformer.py` — Shared transformer primitives (`ResidualAttentionBlock`, `Transformer`, `LayerNorm`, `QuickGELU`)
  - `bi_crossattention.py` — `BiCrossAttention` for fine-grained image-text alignment (sigmoid attention)
  - `lorentz.py` — Lorentzian geometry utilities (`exp_map0`, `pairwise_dist`, etc.)
  - `eda.py` — Easy Data Augmentation (synonym replacement, random swap/delete/insert via NLTK WordNet)
  - `shared_modules.py` — `AllGather` autograd function for multi-GPU distributed training
- `s.config.yaml` — Training configuration (hyperparameters, loss weights, schedule, data paths)

## Architecture

The model has two parallel encoder branches that map inputs into a shared 512-dim hyperbolic embedding space:

1. **Image path:** RGB image → patch embedding (16×16 patches) → 12-layer ViT → L2 normalize → Lorentz exp_map → hyperbolic embedding
2. **Text path:** tokenized text (max 77 tokens) → token embedding → 12-layer transformer (causal mask) → EOS pooling → L2 normalize → Lorentz exp_map → hyperbolic embedding

Four loss functions are computed jointly in `CLIP.forward()`:
- **NITC** (weight 1.0): Soft multi-scale contrastive loss with teacher soft labels
- **RITC** (weight 1.0): KL-divergence regularization using Lorentzian distances
- **CITC** (weight 0.1): Cross-modal + in-modal cyclic consistency
- **BAI** (weight 0.03): Bilateral attention-based instance loss via `BiCrossAttention`

## Key Dependencies

PyTorch, NLTK (wordnet/stopwords), einops, numpy. External modules referenced but not in repo: `misc.utils`, `text_utils.tokenizer`.

## Configuration

All hyperparameters are in `s.config.yaml`. Key sections: `experiment` (loss ratios, augmentation), `schedule` (optimizer/LR), `model` (checkpoint paths, embed dim), `data` (batch sizes), `distributed` (NCCL backend).

## Conventions

- Factory functions create model instances: `clip_vitb(config)`, `visual_transformer(...)`, `text_transformers(...)`
- Config object is passed through all components
- Distributed training uses `AllGather` custom autograd op; multi-GPU code checks `is_main_process()` from `misc.utils`
- Mixed precision via `torch.autocast`; optional gradient checkpointing in transformer layers
- `LayerNorm` has a global `LayerNorm.disable` flag to turn off normalization
