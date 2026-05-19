# Omni3DEdit 对话上下文总结

日期：2026-05-19
本地仓库：`/home/ubuntu/Project/Omni3DEdit`
讨论中的服务器仓库：`/zhongjiahui/Project/Omni3DEdit`

## 目的

本文总结上一轮对话中的调试上下文，方便新对话继续处理问题，不必重新推导已经确认过的事实。

主要目标是在不移动代码和权重文件的前提下，让 Omni3DEdit 脚本能在服务器上运行。排查重点包括符号链接、Python 路径、HuggingFace 缓存路径，以及 VAE 加载行为。

## 当前本地工作区状态

编写本总结时，`git status --short` 显示：

```text
 M scripts/run_appearance_1gpu.sh
?? server_packages/
?? weights/
```

当时唯一有未提交 diff 的已跟踪文件是：

```text
scripts/run_appearance_1gpu.sh
```

该 diff 将 appearance checkpoint 路径从原始拼写错误：

```text
checkpoints/omni3dedit_appearance.ckpt
```

改为：

```text
checkpoints/omni3dedit_appearance.ckpt
```

以下文件当时没有未提交 diff：

```text
seva/modules/autoencoder.py
seva/utils_mmdit.py
configs/train/seva_edit_mmdit_*.yaml
```

## 已有记录

重要已有记录：

- `repro_records/2026-05-14_omni3dedit/README.md`：早期复现与环境配置记录。
- `repro_records/2026-05-17_scripts_test/README.md`：本地脚本测试结果。
- `server_packages/2026-05-16_omni3dedit/SERVER_SETUP.md`：服务器打包和运行指导。

2026-05-17 的脚本测试结论是：所有 `scripts/*.sh` 都通过了 `bash -n` 语法检查，但在本地单卡 RTX 4060 8GB 环境中，没有任何脚本完成一次成功的训练或推理。

## 服务器路径策略

服务器上的项目路径是：

```text
/zhongjiahui/Project/Omni3DEdit
```

目标是不移动真实权重文件。预期做法是创建符号链接，让项目相对路径存在：

```text
checkpoints/
weights/seva/
weights/huggingface/hub/
sdxl-vae/
```

项目期望的重要路径：

```text
checkpoints/omni3dedit_add.ckpt
checkpoints/omni3dedit_remove.ckpt
checkpoints/omni3dedit_appearance.ckpt
weights/seva/model.safetensors
sdxl-vae/sdxl_vae.safetensors
```

注意：原始脚本和 README 中 appearance checkpoint 的文件名曾有拼写问题。一个本地脚本已经改成 `appearance`，但磁盘上的 checkpoint 文件名可能仍然使用旧拼写。

## HuggingFace 缓存

`HF_HUB_CACHE` 并不是由项目代码通过 `os.environ.get(...)` 显式读取，而是由 HuggingFace 相关库隐式读取。

它会影响以下调用：

- `seva/utils.py` 和 `seva/utils_mmdit.py` 中的 `hf_hub_download(...)`
- `VGGT.from_pretrained("facebook/VGGT-1B")`
- `open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")`
- 如果相关配置路径被执行，也会影响 Transformer/OpenCLIP 加载器

推荐设置：

```bash
export HF_HUB_CACHE="$PWD/weights/huggingface/hub"
```

另一种做法是把默认缓存路径做成符号链接：

```text
/root/.cache/huggingface/hub -> /zhongjiahui/weights/huggingface/hub
```

后来要求移除这个符号链接。安全命令是：

```bash
rm /root/.cache/huggingface/hub
```

删除目录符号链接时不要在路径末尾加斜杠。

## VGGT 代码问题

服务器遇到过：

```text
ModuleNotFoundError: No module named 'vggt.models.vggt'
```

根因：本地 `src/vggt` 是一个嵌套 Git 仓库/gitlink，不是父仓库中的普通文件。父仓库记录了一个 gitlink：

```text
160000 b02cc03ceee70821ed1231a530c1992507ef9862 src/vggt
```

但没有可用于服务器自动 checkout 的 `.gitmodules` 映射。

服务器上的快速修复：

```bash
cd /zhongjiahui/Project/Omni3DEdit
rm -rf src/vggt
git clone https://github.com/facebookresearch/vggt.git src/vggt
cd src/vggt
git checkout b02cc03ceee70821ed1231a530c1992507ef9862
cd ../..
```

验证：

```bash
cd /zhongjiahui/Project/Omni3DEdit
PYTHONPATH="$PWD:$PWD/src/vggt" python -c "from vggt.models.vggt import VGGT; print('OK')"
```

脚本原本使用：

```bash
PYTHONPATH=. \
```

服务器运行时需要：

```bash
PYTHONPATH="$PWD:$PWD/src/vggt" \
```

至少也需要：

```bash
PYTHONPATH=.:src/vggt \
```

## SEVA 基座模型加载

服务器还遇到过对以下文件的 HuggingFace 下载尝试：

```text
stabilityai/stable-virtual-camera/model.safetensors
```

触发位置：

```text
seva/utils_mmdit.py
```

当前本地 `seva/utils_mmdit.py` 已包含基于 `SEVA_MODEL_PATH` 和 fallback 路径的本地优先加载逻辑：

```text
SEVA_MODEL_PATH
weights/seva/model.safetensors
./model.safetensors
```

如果服务器仍然尝试下载 `stabilityai/stable-virtual-camera`，检查：

```bash
cd /zhongjiahui/Project/Omni3DEdit
echo "$SEVA_MODEL_PATH"
ls -lh "$SEVA_MODEL_PATH"
grep -n "SEVA_MODEL_PATH\|_resolve_local_weight_path\|hf_hub_download" seva/utils_mmdit.py
```

如果 `grep` 看不到 `SEVA_MODEL_PATH`，说明服务器代码缺少本地优先加载补丁。

预期运行时设置：

```bash
export SEVA_MODEL_PATH="$PWD/weights/seva/model.safetensors"
```

## VAE 加载问题

上一轮对话中讨论了两条 VAE 路径。

### 初始发布版行为

初始发布版的 `seva/modules/autoencoder.py` 使用 Diffusers：

```python
AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
    subfolder="vae",
    force_download=False,
    low_cpu_mem_usage=False,
)
```

服务器上可能失败并报：

```text
401 Client Error
stabilityai/stable-diffusion-2-1-base is not a local folder and is not a valid model identifier
```

这表示 Diffusers 尝试访问 HuggingFace，但服务器缺少权限、登录状态或完整本地缓存。

可选修复：

```bash
huggingface-cli login
```

或者确保 `stabilityai/stable-diffusion-2-1-base/vae` 的本地 Diffusers 缓存完整。

### 本地 SGM VAE 补丁

早前本地改动加入了 SGM 本地 VAE 加载器，使用：

```text
SEVA_VAE_PATH
./sdxl-vae/sdxl_vae.safetensors
```

该补丁引入于提交：

```text
b0c670e 2026-05-15 10:59:46 +0800 Chung-JH feat: modify seva代码和.yaml
```

服务器上的 SGM 格式 VAE key 检查显示：

```text
down_blocks 0
up_blocks 0
encoder.down 74
decoder.up 10
```

这说明该 state dict 是 SGM 风格，而不是 Diffusers `down_blocks/up_blocks` 风格。

截至本总结编写时，本地 `seva/modules/autoencoder.py` 当前匹配初始发布版行为，不包含 SGM 本地 VAE 补丁。

下一轮对话需要先决定：

- 如果保留初始发布版行为，就修复 Diffusers VAE 的 HuggingFace 登录或缓存。
- 如果要避免 HuggingFace 登录，就恢复本地 SGM VAE 补丁，并使用 `sdxl-vae/sdxl_vae.safetensors`。

## CLIP 和 DINO 使用情况

CLIP 使用位置：

```text
seva/modules/conditioner.py
```

相关调用：

```python
open_clip.create_model_and_transforms(
    "ViT-H-14", pretrained="laion2b_s32b_b79k"
)
```

它通过 `encode_image(...)` 编码条件图像。

Omni3DEdit 代码没有显式单独加载 DINO。DINO 通过 VGGT 内部出现。`VGGT.from_pretrained("facebook/VGGT-1B")` 会构造 VGGT，它的 aggregator 使用 DINOv2 风格的 patch embedding，例如 `dinov2_vitl14_reg`。

自动下载的原因：

- OpenCLIP 会加载预训练 CLIP 权重。
- VGGT 通过 `PyTorchModelHubMixin` 使用 HuggingFace Hub。
- 如果初始发布版 VAE 行为生效，Diffusers 会通过 HuggingFace Hub 加载 `stable-diffusion-2-1-base` VAE。

成功下载后，重新运行通常会复用缓存，除非：

- 缓存路径改变；
- 换了另一个用户运行脚本；
- 缓存结构不完整；
- 符号链接指向缺失文件；
- 设置了 `HF_HUB_OFFLINE=1`，但本地缓存不完整。

## 服务器调试常用命令

检查 Python 路径和 VGGT：

```bash
cd /zhongjiahui/Project/Omni3DEdit
which python
PYTHONPATH="$PWD:$PWD/src/vggt" python -c "from vggt.models.vggt import VGGT; print('OK')"
```

检查 SEVA 本地模型路径：

```bash
cd /zhongjiahui/Project/Omni3DEdit
echo "$SEVA_MODEL_PATH"
ls -lh "$SEVA_MODEL_PATH"
grep -n "SEVA_MODEL_PATH\|_resolve_local_weight_path\|hf_hub_download" seva/utils_mmdit.py
```

检查 VAE key 格式：

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

避免服务器运行时进入 PDB：

```text
--debug
```

脚本中包含该参数时，失败会停在 `(Pdb)`。无人值守的服务器运行应改为：

```text
--debug=False
```

## 建议后续步骤

1. 先决定 VAE 策略：
   - 使用 Diffusers VAE，并配置 HuggingFace 登录或缓存。
   - 使用本地 SGM VAE 补丁，并指定 `sdxl-vae/sdxl_vae.safetensors`。
2. 修复脚本 `PYTHONPATH`，加入 `src/vggt`。
3. 确保服务器上存在 `src/vggt`，且位于提交 `b02cc03ceee70821ed1231a530c1992507ef9862`。
4. 确保 `SEVA_MODEL_PATH` 指向真实存在的本地 `model.safetensors`，或者服务器代码包含 `utils_mmdit.py` 的本地优先补丁。
5. 服务器批处理运行时移除或禁用 `--debug`。
6. 先重新运行一个单卡脚本，再尝试双卡脚本。
