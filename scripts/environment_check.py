"""Persistent venv readiness marker tied to Python, CUDA torch and requirements."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def fingerprint():
    import torch, torchvision
    if torch.version.cuda is None and os.getenv('ALLOW_CPU') != '1':
        raise RuntimeError('CPU-only torch. Choose a CUDA PyTorch RunPod image; do not overwrite torch with generic pip wheels.')
    if torch.version.cuda and not torch.cuda.is_available() and os.getenv('ALLOW_CPU') != '1':
        raise RuntimeError('CUDA build exists but no usable GPU. Check GPU allocation/driver before setup.')
    if torch.cuda.is_available():
        # An import alone misses new-GPU architecture and torchvision ABI issues.
        device = 'cuda'
        x = torch.ones(1,device=device) + 1
        torchvision.ops.nms(torch.tensor([[0.,0.,1.,1.]],device=device), x, .5)
        torch.cuda.synchronize()
    requirements = hashlib.sha256()
    for path in sorted(ROOT.glob('requirements*.txt')):
        requirements.update(path.read_bytes())
    return dict(python=list(sys.version_info[:2]), torch=torch.__version__, torchvision=torchvision.__version__,
                cuda=torch.version.cuda, requirements=requirements.hexdigest())


def main():
    marker = Path(sys.prefix)/'.setup.json'
    current = fingerprint()
    action = sys.argv[1]
    if marker.exists():
        old = json.loads(marker.read_text())
        if any(old.get(k) != current[k] for k in ['python','torch','torchvision','cuda']):
            raise RuntimeError('Persistent environment runtime changed. Use the original image or choose a NEW VENV_DIR and rerun setup. Existing environment was left intact.')
    if action == 'ready':
        if not marker.exists() or json.loads(marker.read_text()) != current:
            sys.exit(3)
        for module in ['fastapi','uvicorn','nuscenes','aiortc','av','psutil','pynvml','httpx']:
            importlib.import_module(module)
        print('Persistent environment ready; dependency installation skipped.')
    elif action == 'stamp':
        marker.write_text(json.dumps(current,indent=2))
    else:
        print(json.dumps(current))


if __name__ == '__main__':
    main()
