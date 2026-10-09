"""CPU import/CLI identity check; GPU inference is a separate deployment gate."""
import argparse
import importlib.metadata as metadata
import importlib.util
import json
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true", help="Require CUDA and import fused CUDA modules")
args = parser.parse_args()
expected = {
    "torch": "2.8.0", "transformers": "5.17.0", "peft": "0.21.0",
    "huggingface-hub": "1.32.0", "flash-linear-attention": "0.5.2",
    "fla-core": "0.5.2", "triton": "3.7.1", "causal-conv1d": "1.7.0",
}
actual = {name: metadata.version(name) for name in expected}
for name, version in expected.items():
    assert actual[name].split("+")[0] == version, (name, actual[name], version)
import torch
from kev.serve import app

assert importlib.util.find_spec("causal_conv1d") is not None
assert importlib.util.find_spec("fla") is not None
if args.gpu:
    assert torch.cuda.is_available(), "GPU qualification requires a visible CUDA device"
    import causal_conv1d
    from kev.checkpoint import fused_available
    assert fused_available(), "The packaged FLA version is not supported by Kev"
paths = {route.path for route in app.routes}
assert {"/v1/systemone", "/v1/models", "/openapi.json"} <= paths
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch._C._GLIBCXX_USE_CXX11_ABI, "causal-conv1d wheel requires CXX11 ABI"
subprocess.run(["python", "-m", "kev.serve", "--help"], check=True, stdout=subprocess.DEVNULL)
print(json.dumps({"versions": actual, "cuda": torch.version.cuda,
                  "fused_import_tested": args.gpu, "gpu_inference_tested": False}))
