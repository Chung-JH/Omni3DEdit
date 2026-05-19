# Omni3DEdit Server Package

Package date: 2026-05-16

This bundle is split into two archives:

- `omni3dedit_conda_env.tar.gz`
  - Packed Conda environment from local env `omni3dedit`.
  - `vggt` was installed as editable locally, so the environment was packed with `--ignore-editable-packages`. Use `PYTHONPATH=.:src/vggt` on the server.
- `omni3dedit_project_runtime.tar`
  - Project code, configs, demo data, local weights, HuggingFace cache models, Omni3DEdit checkpoints, and SDXL VAE.
  - Excludes `.git`, `.vscode`, `server_packages`, and Python cache files.

## Extract

```bash
mkdir -p ~/envs/omni3dedit
tar -xzf omni3dedit_conda_env.tar.gz -C ~/envs/omni3dedit
source ~/envs/omni3dedit/bin/activate
conda-unpack

mkdir -p ~/Project
tar -xf omni3dedit_project_runtime.tar -C ~/Project
cd ~/Project/Omni3DEdit
```

## Runtime Environment

```bash
export SEVA_MODEL_PATH="$PWD/weights/seva/model.safetensors"
export SEVA_VAE_PATH="$PWD/sdxl-vae/sdxl_vae.safetensors"
export HF_HUB_CACHE="$PWD/weights/huggingface/hub"
export PYTHONPATH="$PWD:$PWD/src/vggt"
export MPLCONFIGDIR=/tmp/matplotlib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Quick Checks

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
python -c "import seva.ssim_psnr; print('ssim ok')"
HF_HUB_OFFLINE=1 python - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "open_clip_pytorch_model.bin"))
print(hf_hub_download("facebook/VGGT-1B", "model.safetensors"))
PY
```

## Remove Reproduction Command

The local RTX 4060 8GB machine OOMed during the second VGGT-1B GPU initialization. Use a larger-GPU server, or change VGGT initialization/offload logic.

```bash
CUDA_VISIBLE_DEVICES=0 \
USE_FLASH_ATTENTION=1 \
HF_HUB_DISABLE_XET=1 \
python -u -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node=1 \
  --master_addr=localhost \
  --master_port=12346 \
  main.py \
  --base=configs/train/seva_edit_mmdit_remove.yaml \
  --no_date \
  --train=False \
  --debug=False \
  --resume=checkpoints/omni3dedit_remove.ckpt
```

## Weight Layout

- SEVA: `weights/seva/`
- HuggingFace cache: `weights/huggingface/hub/`
- Omni3DEdit checkpoints: `checkpoints/`
- SDXL VAE: `sdxl-vae/`

