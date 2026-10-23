"""Compare the runtime with the tested one: package versions (warnings) and the custom OGBench build (errors)."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import zipfile

EXPECTED = {
    'torch': '2.5.1', 'torchvision': '0.20.1', 'lightning': '2.1.4',
    'numpy': '1.26.4', 'gym': '0.19.0', 'gymnasium': '0.29.1',
    'mujoco': '3.2.4', 'mujoco-py': '2.1.2.14', 'ogbench': '1.0.1',
    'moviepy': '1.0.3', 'diffusers': '0.31.0',
    'rotary-embedding-torch': '0.8.5', 'stable-baselines3': '2.3.2',
}
WHEEL_SHA256 = '97626b36c14c8af362de1c82762dc94574434fa406a51a109cd65c0ccd1018bb'


def main():
    errors, warnings, versions = [], [], {}
    for package, expected in EXPECTED.items():
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
        if (versions[package] or '').split('+')[0] != expected:
            warnings.append(f'{package}: tested {expected}, found {versions[package]}')
    wheel = Path(__file__).resolve().parents[1] / 'dockerfile/ogbench-1.0.1-py3-none-any.whl'
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != WHEEL_SHA256:
        errors.append('Bundled custom OGBench wheel checksum differs')
    try:
        dist = metadata.distribution('ogbench')
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if name.startswith('ogbench/') and name.endswith('.py'):
                    installed = Path(dist.locate_file(name))
                    if not installed.is_file() or installed.read_bytes() != archive.read(name):
                        errors.append(f'Custom OGBench source mismatch: {name}')
    except metadata.PackageNotFoundError:
        errors.append('ogbench is not installed')
    import torch
    if torch.version.cuda != '12.4':
        errors.append(f'Expected PyTorch CUDA runtime 12.4, got {torch.version.cuda}')
    print(json.dumps({'python': sys.version, 'versions': versions,
                      'torch_cuda_runtime': torch.version.cuda,
                      'cuda_available': torch.cuda.is_available(),
                      'warnings': warnings, 'errors': errors}, indent=2))
    # CPU-only checks are supported; absence of a GPU is not an environment mismatch.
    return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
