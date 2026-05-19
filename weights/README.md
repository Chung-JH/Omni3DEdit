# Local Weight Layout

This directory stores manually provided model weights that are not part of the original git repository.

## SEVA Base Weights

- `seva/model.safetensors`
  - Default local SEVA base model used by `seva.utils_mmdit.load_model()` and `seva.utils.load_model()`.
- `seva/modelv1.1.safetensors`
  - Backup SEVA base model. Do not use automatically unless `model.safetensors` shows structure mismatch.

## HuggingFace Cache Models

These HuggingFace-managed model cache directories were moved into the project for reuse:

- `huggingface/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K`
  - OpenCLIP ViT-H-14 cache used by `open_clip`.
- `huggingface/hub/models--facebook--VGGT-1B`
  - VGGT-1B cache used by `VGGT.from_pretrained("facebook/VGGT-1B")`.

The original HuggingFace cache paths in `~/.cache/huggingface/hub/` are symlinks back to these project-local directories, so existing library calls still resolve without changing code.

For a fresh machine or moved checkout, either recreate those symlinks or run commands with:

```bash
HF_HUB_CACHE=/path/to/Omni3DEdit/weights/huggingface/hub
```

## Other Required Weights

These are kept in their original project-expected locations:

- `../checkpoints/omni3dedit_remove.ckpt`
- `../checkpoints/omni3dedit_add.ckpt`
- `../checkpoints/omni3dedit_appearance.ckpt`
- `../sdxl-vae/sdxl_vae.safetensors`
