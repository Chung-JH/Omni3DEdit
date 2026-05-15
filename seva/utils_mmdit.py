import os

import safetensors.torch
import torch
from huggingface_hub import hf_hub_download

from seva.model_mmdit import Seva, SevaParams


def seed_everything(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


def _resolve_local_weight_path(weight_name: str) -> str | None:
    env_model_path = os.environ.get("SEVA_MODEL_PATH")
    if env_model_path:
        env_model_path = os.path.expanduser(env_model_path)
        if os.path.isdir(env_model_path):
            env_model_path = os.path.join(env_model_path, weight_name)
        if os.path.isfile(env_model_path):
            return env_model_path

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for repo_weight_path in (
        os.path.join(repo_root, "weights", "seva", weight_name),
        os.path.join(repo_root, weight_name),
    ):
        if os.path.isfile(repo_weight_path):
            return repo_weight_path

    return None


def load_model(
    pretrained_model_name_or_path: str = "stabilityai/stable-virtual-camera",
    weight_name: str = "model.safetensors",
    device: str | torch.device = "cuda",
    verbose: bool = False,
) -> Seva:
    local_weight_path = _resolve_local_weight_path(weight_name)
    if local_weight_path is not None:
        weight_path = local_weight_path
    elif os.path.isdir(pretrained_model_name_or_path):
        weight_path = os.path.join(pretrained_model_name_or_path, weight_name)
    else:
        weight_path = hf_hub_download(
            repo_id=pretrained_model_name_or_path, filename=weight_name
        )
        _ = hf_hub_download(
            repo_id=pretrained_model_name_or_path, filename="config.yaml"
        )

    state_dict = safetensors.torch.load_file(
        weight_path,
        device=str(device),
    )

    # with torch.device("meta"):
    #     model = Seva(SevaParams()).to(torch.bfloat16)

    model = Seva(SevaParams()).to(torch.bfloat16)
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    if verbose:
        print_load_warning(missing, unexpected)
    model = model.to(device)
    return model


if __name__ == "__main__":
    model = load_model(pretrained_model_name_or_path="stabilityai/stable-virtual-camera", weight_name="model.safetensors", device="cuda", verbose=True)
    print(model)
