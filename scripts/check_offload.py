"""Check that OmniVoice.to('cpu') actually releases VRAM.

`.to()` only moves nn.Module attributes; anything held as a plain tensor or a
non-Module submodule stays on the GPU, which silently breaks exclusive
residency.
"""

from __future__ import annotations

import torch
from omnivoice import OmniVoice


def mib() -> int:
    return torch.cuda.memory_allocated() // 1024 // 1024


def reserved_mib() -> int:
    return torch.cuda.memory_reserved() // 1024 // 1024


def main() -> None:
    print(f"baseline           allocated={mib():>5} reserved={reserved_mib():>5}")
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
    print(f"loaded             allocated={mib():>5} reserved={reserved_mib():>5}")
    print(f"is nn.Module: {isinstance(model, torch.nn.Module)}")

    model.generate(text="مرحبا", language="arb")
    print(f"after generate     allocated={mib():>5} reserved={reserved_mib():>5}")

    model.to("cpu")
    torch.cuda.empty_cache()
    print(f"after .to('cpu')   allocated={mib():>5} reserved={reserved_mib():>5}")

    import gc

    gc.collect()
    torch.cuda.empty_cache()
    print(f"after gc+empty     allocated={mib():>5} reserved={reserved_mib():>5}")

    # What is still on the GPU?
    stragglers: list[str] = []
    for name, value in vars(model).items():
        if isinstance(value, torch.nn.Module):
            devices = {str(p.device) for p in value.parameters()}
            if any(d.startswith("cuda") for d in devices):
                stragglers.append(f"{name} ({type(value).__name__}) {devices}")
        elif isinstance(value, torch.Tensor) and value.is_cuda:
            stragglers.append(f"{name} (Tensor {tuple(value.shape)})")

    print("\nstill on GPU after offload:" if stragglers else "\nnothing left on GPU")
    for s in stragglers:
        print(f"  {s}")


if __name__ == "__main__":
    main()
