from hashlib import sha256

import torch


def hash_script(script: torch.jit.ScriptModule) -> str:
    return sha256(
        script.graph.__repr__().encode("utf-8")
        + list(script.parameters()).__repr__().encode("utf-8")
    ).hexdigest()
