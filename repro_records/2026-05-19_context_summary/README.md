# Omni3DEdit Conversation Context Summary

Date: 2026-05-19  
Local repo: `/home/ubuntu/Project/Omni3DEdit`  
Server repo discussed: `/zhongjiahui/Project/Omni3DEdit`

## Purpose

This document summarizes the debugging context from the prior conversation so a new chat can continue without re-deriving the same facts.

The main topic was making Omni3DEdit scripts run on a server without moving code or weights, mostly by fixing symlinks, Python paths, HuggingFace cache paths, and VAE loading behavior.

## Current Local Worktree State

At the time this summary was written, `git status --short` showed:

```text
 M scripts/run_appearance_1gpu.sh
?? server_packages/
?? weights/
```

The only tracked file with an uncommitted local diff was:

```text
scripts/run_appearance_1gpu.sh
```

The diff changes the appearance checkpoint path from the original typo:

```text
checkpoints/omni3dedit_appearance.ckpt
```

to:

```text
checkpoints/omni3dedit_appearance.ckpt
```

No current uncommitted diff was present in:

```text
seva/modules/autoencoder.py
seva/utils_mmdit.py
configs/train/seva_edit_mmdit_*.yaml
```

## Existing Records

Important existing notes:

- `repro_records/2026-05-14_omni3dedit/README.md`: earlier reproduction and environment setup notes.
- `repro_records/2026-05-17_scripts_test/README.md`: local script test results.
- `server_packages/2026-05-16_omni3dedit/SERVER_SETUP.md`: server package/runtime guidance.

The 2026-05-17 script test found that all `scripts/*.sh` passed `bash -n`, but none completed a successful local train/inference run in the local 1-GPU RTX 4060 8GB environment.

## Server Path Strategy

The server project lives at:

```text
/zhongjiahui/Project/Omni3DEdit
```

The goal was to avoid moving actual weight files. The intended approach was to create symlinks so project-relative paths exist:

```text
checkpoints/
weights/seva/
weights/huggingface/hub/
sdxl-vae/
```

Important project-expected paths:

```text
checkpoints/omni3dedit_add.ckpt
checkpoints/omni3dedit_remove.ckpt
checkpoints/omni3dedit_appearance.ckpt
weights/seva/model.safetensors
sdxl-vae/sdxl_vae.safetensors
```

Note: the original scripts and README use the typo `appearance` for the appearance checkpoint. One local script was changed to `appearance`, but checkpoint filenames on disk may still use `appearance`.

## HuggingFace Cache

`HF_HUB_CACHE` is not explicitly read by project code via `os.environ.get(...)`. It is read implicitly by HuggingFace-related libraries.

It affects calls such as:

- `hf_hub_download(...)` in `seva/utils.py` and `seva/utils_mmdit.py`
- `VGGT.from_pretrained("facebook/VGGT-1B")`
- `open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")`
- Transformer/OpenCLIP loaders if those config paths are exercised

The recommended setting was:

```bash
export HF_HUB_CACHE="$PWD/weights/huggingface/hub"
```

Alternatively, symlink the default cache:

```text
/root/.cache/huggingface/hub -> /zhongjiahui/weights/huggingface/hub
```

That symlink was later requested to be removed. The safe command is:

```bash
rm /root/.cache/huggingface/hub
```

Do not use a trailing slash when removing a directory symlink.

## VGGT Code Issue

The server hit:

```text
ModuleNotFoundError: No module named 'vggt.models.vggt'
```

Root cause: local `src/vggt` is a nested Git repository/gitlink, not normal files in the parent repo. The parent repo had a gitlink:

```text
160000 b02cc03ceee70821ed1231a530c1992507ef9862 src/vggt
```

but no usable `.gitmodules` mapping for automatic checkout on the server.

Fast server fix:

```bash
cd /zhongjiahui/Project/Omni3DEdit
rm -rf src/vggt
git clone https://github.com/facebookresearch/vggt.git src/vggt
cd src/vggt
git checkout b02cc03ceee70821ed1231a530c1992507ef9862
cd ../..
```

Validation:

```bash
cd /zhongjiahui/Project/Omni3DEdit
PYTHONPATH="$PWD:$PWD/src/vggt" python -c "from vggt.models.vggt import VGGT; print('OK')"
```

The scripts originally used:

```bash
PYTHONPATH=. \
```

For server use, they need:

```bash
PYTHONPATH="$PWD:$PWD/src/vggt" \
```

or at least:

```bash
PYTHONPATH=.:src/vggt \
```

## SEVA Base Model Loading

The server also hit HuggingFace download attempts for:

```text
stabilityai/stable-virtual-camera/model.safetensors
```

That happens in:

```text
seva/utils_mmdit.py
```

Current local `seva/utils_mmdit.py` includes local-first loading via `SEVA_MODEL_PATH` and fallback paths:

```text
SEVA_MODEL_PATH
weights/seva/model.safetensors
./model.safetensors
```

If the server still tries to download `stabilityai/stable-virtual-camera`, check:

```bash
cd /zhongjiahui/Project/Omni3DEdit
echo "$SEVA_MODEL_PATH"
ls -lh "$SEVA_MODEL_PATH"
grep -n "SEVA_MODEL_PATH\|_resolve_local_weight_path\|hf_hub_download" seva/utils_mmdit.py
```

If `grep` does not show `SEVA_MODEL_PATH`, the server code is missing the local-first patch.

Expected runtime setting:

```bash
export SEVA_MODEL_PATH="$PWD/weights/seva/model.safetensors"
```

## VAE Loading Issue

The conversation went through two VAE paths.

### Initial Release Behavior

`seva/modules/autoencoder.py` in Initial release uses Diffusers:

```python
AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
    subfolder="vae",
    force_download=False,
    low_cpu_mem_usage=False,
)
```

This can fail on the server with:

```text
401 Client Error
stabilityai/stable-diffusion-2-1-base is not a local folder and is not a valid model identifier
```

That means Diffusers attempted to access HuggingFace and the server lacked permission/login or cached files.

Possible fixes:

```bash
huggingface-cli login
```

or ensure a complete local Diffusers cache for `stabilityai/stable-diffusion-2-1-base/vae`.

### Local SGM VAE Patch

Earlier local work added an SGM local VAE loader that used:

```text
SEVA_VAE_PATH
./sdxl-vae/sdxl_vae.safetensors
```

That patch was introduced in commit:

```text
b0c670e 2026-05-15 10:59:46 +0800 Chung-JH feat: modify seva代码和.yaml
```

The SGM-format VAE key check on the server showed:

```text
down_blocks 0
up_blocks 0
encoder.down 74
decoder.up 10
```

This indicates an SGM-style state dict, not a Diffusers `down_blocks/up_blocks` state dict.

As of this summary, local `seva/modules/autoencoder.py` currently matches Initial release behavior, not the SGM local VAE patch.

Decision point for the next conversation:

- If keeping Initial release behavior, fix HuggingFace auth/cache for Diffusers VAE.
- If avoiding HuggingFace auth, restore the local SGM VAE patch and use `sdxl-vae/sdxl_vae.safetensors`.

## CLIP And DINO Usage

CLIP is used by:

```text
seva/modules/conditioner.py
```

The relevant call is:

```python
open_clip.create_model_and_transforms(
    "ViT-H-14", pretrained="laion2b_s32b_b79k"
)
```

It encodes conditioning images via `encode_image(...)`.

DINO is not explicitly loaded by Omni3DEdit code as a standalone model. It appears through VGGT internals. `VGGT.from_pretrained("facebook/VGGT-1B")` constructs VGGT, whose aggregator uses DINOv2-style patch embedding, e.g. `dinov2_vitl14_reg`.

Automatic downloads happen because:

- OpenCLIP loads pretrained CLIP weights.
- VGGT uses HuggingFace Hub via `PyTorchModelHubMixin`.
- Diffusers uses HuggingFace Hub for `stable-diffusion-2-1-base` VAE if Initial release behavior is active.

After a successful download, reruns should reuse cache unless:

- the cache path changes,
- a different user runs the script,
- the cache structure is incomplete,
- symlinks point to missing files,
- `HF_HUB_OFFLINE=1` is set but local cache is incomplete.

## Common Commands For Server Debugging

Check Python path and VGGT:

```bash
cd /zhongjiahui/Project/Omni3DEdit
which python
PYTHONPATH="$PWD:$PWD/src/vggt" python -c "from vggt.models.vggt import VGGT; print('OK')"
```

Check SEVA local model path:

```bash
cd /zhongjiahui/Project/Omni3DEdit
echo "$SEVA_MODEL_PATH"
ls -lh "$SEVA_MODEL_PATH"
grep -n "SEVA_MODEL_PATH\|_resolve_local_weight_path\|hf_hub_download" seva/utils_mmdit.py
```

Check VAE key format:

```bash
cd /zhongjiahui/Project/Omni3DEdit
python - <<'PY'
from safetensors.torch import load_file
sd = load_file("sdxl-vae/sdxl_vae.safetensors", device="cpu")
print(list(sd)[:20])
print("down_blocks", sum("down_blocks" in k for k in sd))
print("up_blocks", sum("up_blocks" in k for k in sd))
print("encoder.down", sum(k.startswith("encoder.down") for k in sd))
print("decoder.up", sum(k.startswith("decoder.up") for k in sd))
PY
```

Avoid PDB during server runs:

```text
--debug
```

in scripts causes failures to stop in `(Pdb)`. For unattended server runs, change it to:

```text
--debug=False
```

## Recommended Next Steps

1. Decide VAE strategy first:
   - Diffusers VAE with HuggingFace login/cache.
   - Local SGM VAE patch with `sdxl-vae/sdxl_vae.safetensors`.
2. Fix script `PYTHONPATH` to include `src/vggt`.
3. Ensure `src/vggt` exists on server at commit `b02cc03ceee70821ed1231a530c1992507ef9862`.
4. Ensure `SEVA_MODEL_PATH` points to a real local `model.safetensors`, or server code includes the local-first `utils_mmdit.py` patch.
5. Remove or disable `--debug` for server batch runs.
6. Re-run one 1-GPU script first before trying 2-GPU scripts.
