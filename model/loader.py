from pkg_utils import (
is_safetensors_available,
is_torch_available,
is_torch_cuda_available,
device,
torch_version,
PackageNotFoundError)
from os import PathLike
from typing import Union
if is_torch_available():
    from torch import load
device = device()

def load_state_dict(pretrained_model_file_path: Union[str, PathLike], device = device):
    if pretrained_model_file_path.endswith('.safetensors') and is_safetensors_available():
        from safetensors.torch import load_file as safe_load_file
        return safe_load_file(pretrained_model_file_path)
    elif pretrained_model_file_path.endswith('.bin') or pretrained_model_file_path.endswith(".pt") and is_torch_available():
       return load(pretrained_model_file_path, map_location=device)

def load_weight(model, state_dict):
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if key.endswith(".g"):
            new_key = key[:-2] + ".weight"
        elif key.endswith(".b"):
            new_key = key[:-2] + ".bias"
        elif key.endswith(".w"):
            new_key = key[:-2] + ".weight"
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    start_model = model
    if hasattr(model, "transformer") and all(not s.startswith('transformer.') for s in state_dict.keys()):
        start_model = model.transformer
    load(start_model, prefix="")
    model.set_tied()
    return model

if __name__ == '__main__':
    print(device)
