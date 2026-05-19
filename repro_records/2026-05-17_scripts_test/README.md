# Omni3DEdit scripts 测试记录

日期：2026-05-17  
仓库：`/home/ubuntu/Project/Omni3DEdit`  
Commit：`7a77e6729f4ad1e8b47d2121e1a87f3ddc142542`

## 结论

本次对 `scripts/` 下 12 个 shell 脚本做了两层测试：

- `bash -n` 语法检查：12 个脚本全部通过。
- 真实启动测试：在本机环境下，没有脚本完成一次成功的训练或推理。

当前主要阻塞分为三类：

- `run_remove_1gpu.sh` 在恢复 `checkpoints/omni3dedit_remove.ckpt` 时明确报错：`KeyError: "filename 'storages' not found"`。
- 其余单卡脚本大多能走到 checkpoint 恢复阶段，但随后子进程被 `SIGKILL`。
- 双卡脚本在当前 1-GPU 机器上不可用；实测也未完成启动，最终同样表现为子进程被 `SIGKILL`。

## 测试环境

- Conda 环境：`omni3dedit`
- Python：`3.10`
- GPU：`NVIDIA GeForce RTX 4060 8GB`
- 可见 GPU 数量：`1`

运行时按 [server_packages/2026-05-16_omni3dedit/SERVER_SETUP.md](/home/ubuntu/Project/Omni3DEdit/server_packages/2026-05-16_omni3dedit/SERVER_SETUP.md:1) 补齐了以下环境变量：

- `SEVA_MODEL_PATH`
- `SEVA_VAE_PATH`
- `HF_HUB_CACHE`
- `MPLCONFIGDIR`
- `PYTORCH_CUDA_ALLOC_CONF`
- `HF_HUB_OFFLINE`
- `HF_HUB_DISABLE_XET`

## 逐项结果

| 脚本 | 结果 | 说明 |
| --- | --- | --- |
| `scripts/run_add_1gpu.sh` | 失败 | 能启动到 `Restoring states from the checkpoint path at checkpoints/omni3dedit_add.ckpt`，随后子进程 `SIGKILL`。 |
| `scripts/run_add_2gpu.sh` | 失败 | 当前机器仅 1 张 GPU；双卡启动未成功，最终 `local_rank` 子进程 `SIGKILL`。 |
| `scripts/run_appearance_1gpu.sh` | 失败 | 能启动到 `Restoring states from the checkpoint path at checkpoints/omni3dedit_appearance.ckpt`，随后子进程 `SIGKILL`。 |
| `scripts/run_appearance_2gpu.sh` | 失败 | 当前机器仅 1 张 GPU；双卡启动未成功，最终 `local_rank` 子进程 `SIGKILL`。 |
| `scripts/run_remove_1gpu.sh` | 失败 | 明确报错：`KeyError: "filename 'storages' not found"`，发生在 Lightning 恢复 `checkpoints/omni3dedit_remove.ckpt` 时。 |
| `scripts/run_remove_2gpu.sh` | 失败 | 当前机器仅 1 张 GPU；双卡启动未成功，最终子进程 `SIGKILL`。 |
| `scripts/train_add_1gpu.sh` | 失败 | 模型和数据初始化可通过，进入 DDP 后在 `Summoning checkpoint.` 附近被 `SIGKILL`。 |
| `scripts/train_add_2gpu.sh` | 失败 | 当前机器仅 1 张 GPU；双卡启动未成功，最终子进程 `SIGKILL`。 |
| `scripts/train_appearance_1gpu.sh` | 失败 | 到 `Summoning checkpoint.` 附近后被 `SIGKILL`。 |
| `scripts/train_appearance_2gpu.sh` | 失败 | 当前机器仅 1 张 GPU；双卡启动未成功，最终子进程 `SIGKILL`。 |
| `scripts/train_remove_1gpu.sh` | 失败 | 到 `Summoning checkpoint.` 附近后被 `SIGKILL`。 |
| `scripts/train_remove_2gpu.sh` | 失败 | 当前机器仅 1 张 GPU；双卡启动未成功，最终子进程 `SIGKILL`。 |

## 额外观察

- 默认 shell 环境下不能直接运行项目，未激活 `omni3dedit` 时连 `torch` 都不可导入。
- 脚本本身没有自动设置服务端运行文档中的关键环境变量，依赖调用者预先准备环境。
- 本次测试没有修改仓库代码文件。
- 本次测试曾在仓库内生成 `logs/` 运行目录，并在 `/tmp/omni3dedit_script_tests/` 生成临时日志；后者已清理。

## 建议

- 优先排查 `checkpoints/omni3dedit_remove.ckpt` 的可读性问题，至少要解释 `filename 'storages' not found`。
- 给 `scripts/*.sh` 增加前置自检，至少检查：
  - 当前 Python 是否来自 `omni3dedit` 环境
  - GPU 数量是否满足 `1gpu/2gpu` 脚本要求
  - `SEVA_MODEL_PATH`、`SEVA_VAE_PATH`、`HF_HUB_CACHE` 是否可用
  - `--resume` 指向的 checkpoint 是否能被最小化读取
