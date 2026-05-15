# Omni3DEdit 快速复现记录

日期：2026-05-14  
仓库：`/home/ubuntu/Project/Omni3DEdit`  
Commit：`13da5f2c9e873c0f763ac09387f65c7153556d6a`

## 结论

本次继续复现已绕过 `stabilityai/stable-virtual-camera/model.safetensors` gated 下载问题，改为优先加载本地 SEVA 权重。当前本地 SEVA 权重已归档到 `weights/seva/`。同时处理了后续暴露出的本地 VAE、OpenCLIP、VGGT 和 demo 数据过滤问题。

remove 任务最终未生成图片，当前阻塞点是 8GB RTX 4060 显存不足：在 train/test dataset 分别初始化 `VGGTPipeline` 时，第二次将 `facebook/VGGT-1B` 搬到 GPU 触发 OOM：

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 32.00 MiB.
GPU 0 has a total capacity of 8.00 GiB of which 2.66 GiB is free.
```

因此按计划停止后续 add 和 appearance/color_change 推理，没有 `results_unitrain/` 图片产物可分析。

## 继续执行记录

本轮新增验证：

- `weights/seva/model.safetensors`：4.8G，SHA256 `10e69ea003c313e6bdfc7ee40376d1c19ea6036c20bd384e94b483dec8350396`
- `weights/seva/modelv1.1.safetensors`：4.8G，SHA256 `b45d13cb89fe652d607c31989e5f71c2f1217a4f0939301fb60f3b941d6636da`
- 两个 safetensors 均可用 `safetensors.torch.load_file(..., device="cpu")` 打开，均为 1146 个 tensor。
- `SEVA_MODEL_PATH=weights/seva/model.safetensors` 的路径解析验证通过，不再调用 `hf_hub_download()`。
- 本地 `sdxl-vae/sdxl_vae.safetensors` 可构造 SGM `AutoencoderKL`，加载结果为 0 missing、2 个 EMA 相关 unexpected key。
- OpenCLIP `ViT-H-14/laion2b_s32b_b79k` 已补齐并归档到 `weights/huggingface/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K`，原 HuggingFace cache 路径为 symlink。
- `facebook/VGGT-1B` 已补齐并归档到 `weights/huggingface/hub/models--facebook--VGGT-1B`，原 HuggingFace cache 路径为 symlink。

本轮新增补丁：

- `seva/utils_mmdit.py`、`seva/utils.py`：新增 `SEVA_MODEL_PATH`/仓库根目录本地 SEVA safetensors 优先解析。
- `seva/modules/autoencoder.py`：优先使用 `SEVA_VAE_PATH` 或 `./sdxl-vae/sdxl_vae.safetensors` 构造本地 SGM VAE；本地不存在时保留原 Diffusers 远程回退。
- `configs/train/seva_edit_mmdit_{remove,add,appearance}.yaml`：将 demo 数据过滤条件改为 OmegaConf list，避免原 `("...")` 字符串导致空数据集；remove/add/appearance 分别匹配 1 个 demo JSON。
- 保留已有 `seva/ssim_psnr.py` 的 `basicsr` 兼容补丁。

完整代码 diff 见：

- `logs/code_changes.diff`

本轮 remove 重跑日志：

- `logs/remove_inference_rerun.log`
- `logs/nvidia_smi_after_oom.txt`

最后一次 remove 命令：

```bash
SEVA_MODEL_PATH=weights/seva/model.safetensors \
SEVA_VAE_PATH=./sdxl-vae/sdxl_vae.safetensors \
HF_HUB_DISABLE_XET=1 \
CUDA_VISIBLE_DEVICES=0 \
USE_FLASH_ATTENTION=1 \
PYTHONPATH=.:src/vggt \
MPLCONFIGDIR=/tmp/matplotlib \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/omni3dedit/bin/python -u -m torch.distributed.run \
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

## 环境与数据

- Conda 环境：`omni3dedit`
- Python：`3.10.20`
- PyTorch：`2.5.0+cu124`
- CUDA 检测：`torch.cuda.is_available() == True`，`device_count == 1`
- GPU：NVIDIA GeForce RTX 4060，8GB 显存
- Demo 数据校验：
  - `demo_remove_book.json`：10 pairs，missing 0
  - `demo_add_teddy_bear.json`：10 pairs，missing 0
  - `demo_color_bear_black.json`：10 pairs，missing 0

详细记录见：

- `logs/env_verify.txt`
- `logs/nvidia_smi_before_inference.txt`
- `logs/data_validate.txt`
- `logs/pip_freeze.txt`

## 权重记录

已下载到 README 期望路径：

```text
checkpoints/omni3dedit_remove.ckpt      19G
checkpoints/omni3dedit_add.ckpt         19G
checkpoints/omni3dedit_apperance.ckpt   19G
sdxl-vae/sdxl_vae.safetensors           320M
```

SHA256 记录见 `checksums.txt`：

```text
40ee450cd20b2b64c32dd379d63d56beb91fdb6b1ef384882afbd5b8a91d5f2e  checkpoints/omni3dedit_remove.ckpt
7ce174a880acec5d5cab3738f47733368fd0eeabb26c95156954732404ffc988  checkpoints/omni3dedit_add.ckpt
a2bd302eba68feb0d9a33e723c54bbe0b35c0b4bd1732282fab22e4749fd7cd7  checkpoints/omni3dedit_apperance.ckpt
63aeecb90ff7bc1c115395962d3e803571385b61938377bc7089b36e81e92e2e  sdxl-vae/sdxl_vae.safetensors
10e69ea003c313e6bdfc7ee40376d1c19ea6036c20bd384e94b483dec8350396  weights/seva/model.safetensors
b45d13cb89fe652d607c31989e5f71c2f1217a4f0939301fb60f3b941d6636da  weights/seva/modelv1.1.safetensors
```

下载说明：

- `sdxl_vae.safetensors` 可直接下载。
- Omni3DEdit checkpoint 使用 `HF_HUB_DISABLE_XET=1` 才能稳定续传。
- 下载过程中出现过 HuggingFace Xet TLS EOF 和普通 HTTP `IncompleteRead`，均通过续传完成。

## 修改记录

本轮累计修改的仓库文件：

- `seva/ssim_psnr.py`
- `seva/utils_mmdit.py`
- `seva/utils.py`
- `seva/modules/autoencoder.py`
- `configs/train/seva_edit_mmdit_remove.yaml`
- `configs/train/seva_edit_mmdit_add.yaml`
- `configs/train/seva_edit_mmdit_appearance.yaml`

`seva/ssim_psnr.py` 的原因：`basicsr==1.4.2` 导入时依赖 `torchvision.transforms.functional_tensor`，该模块在当前 `torchvision==0.20.0+cu124` 中不存在，导致程序在模型初始化前失败。

处理：将 `reorder_image`、`to_y_channel`、`rgb2ycbcr_pt` 三个小工具函数内联到 `seva/ssim_psnr.py`，避免导入 `basicsr` 顶层模块。其余修改用于本地 SEVA/VAE 加载和 demo 数据过滤。完整 diff 见：

- `logs/code_changes.diff`

验证：

```text
conda run -n omni3dedit python -c "import seva.ssim_psnr; print('ssim_psnr import ok')"
```

输出：

```text
ssim_psnr import ok
```

环境安装偏离：

- README 的 `pip install -r requirements.txt` 首次失败于 `xformers` git 源码构建，因为构建阶段尚未安装 `torch`。
- 实际采用：
  - 先安装 `torch==2.5.0+cu124`、`torchvision==0.20.0+cu124`、`torchaudio==2.5.0+cu124`
  - 再安装过滤掉远程 `vggt` 与源码 `xformers` 行的 requirements
  - 使用本地 `src/vggt` editable install
  - 安装预编译 `xformers==0.0.28.post2 --no-deps`

## 旧推理记录

首次执行 remove：

```bash
CUDA_VISIBLE_DEVICES=0 \
USE_FLASH_ATTENTION=1 \
PYTHONPATH=.:src/vggt \
MPLCONFIGDIR=/tmp/matplotlib \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/miniconda3/envs/omni3dedit/bin/python -u -m torch.distributed.run \
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

旧结果：

- 程序成功进入 PyTorch Lightning 初始化。
- CUDA 可用，DDP 单进程启动成功。
- 在 `SevaEngine.__init__` 中调用 `seva.utils_mmdit.load_model()` 时尝试下载 `stabilityai/stable-virtual-camera/model.safetensors`。
- HuggingFace 返回 gated repo 401，推理退出，状态码 1。
- 该阻塞已由本轮本地 `SEVA_MODEL_PATH=weights/seva/model.safetensors` 补丁解决；当前最新阻塞为 VGGT 显存 OOM，见上方“继续执行记录”。

日志：

- `logs/remove_inference.log`
- `logs/add_inference.log`
- `logs/appearance_inference.log`

add 与 appearance 在本轮仍未执行；原因从旧的 gated SEVA 下载阻塞变为 remove 已进入 VGGT 初始化后发生 8GB 显存 OOM，按计划停止后续任务。

## 中间结果与实验结果

中间结果：

- 已完成 conda 环境补齐。
- 已完成 demo JSON 与图像路径校验。
- 已完成 3 个 Omni3DEdit task checkpoint 和 SDXL VAE 下载。
- 已完成 checkpoint checksum。
- 已完成 remove 推理启动验证。
- 已完成本地 SEVA、SDXL VAE、OpenCLIP、VGGT 权重加载验证。
- 已确认 demo remove 数据过滤可匹配 1 个 JSON。

实验结果：

- 没有生成 `sampled.png`。
- 没有生成 per-view edited PNG。
- 没有生成可用于定性分析的结果图片。

结果分析：

- 当前失败发生在第二个 VGGT-1B 实例搬到 GPU 的阶段，早于实际采样和图像生成。
- 因此无法判断 object removal/addition/color_change 的编辑质量、多视角一致性、伪影或源视角保真度。
- 8GB 显存风险已确认：remove 在 VGGT 初始化阶段 OOM，add/appearance 未继续执行。

## 后续复现条件

要继续完成真实图片复现，主要需要更高显存或减少重复常驻模型：

1. 使用更高显存 GPU 重新执行 remove，建议至少超过 8GB，因为当前在第二个 VGGT 实例搬到 GPU 时 OOM。
2. 或修改数据集/预处理初始化逻辑，避免 train/test dataset 各自常驻一个 VGGT-1B GPU 实例，例如复用 VGGT、延迟加载、CPU/offload，或只初始化当前推理实际需要的数据集。

本地 SEVA gated 权重问题已通过 `SEVA_MODEL_PATH` 解决；当前不再需要 HuggingFace 登录来加载 SEVA 基座。
