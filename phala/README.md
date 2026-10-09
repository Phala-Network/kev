# Phala Kev image

Builds this fork's Apache-2.0 [Kev runtime](https://github.com/jaredpalmer/kev/tree/5e42a7a03f28134853dd3ff77461457e921e5ec1)
from an immutable Phala source release. The initial fork base is
`5e42a7a03f28134853dd3ff77461457e921e5ec1`; the serving implementation is unchanged. This is a prefill-only typed
decision service, **not an OpenAI chat model**. Its API is `POST /v1/systemone`
and TypeSafe `GET /v1/models` (a `models` array, not an OpenAI `data` array).

The Dockerfile copies the released fork checkout and installs its frozen
`uv.lock` serving environment, and applies the author's CUDA additions at exact
versions and wheel hashes. Python 3.13 Linux/amd64, Torch 2.8.0 CUDA 12.8,
Transformers 5.17.0, PEFT 0.21.0, FLA 0.5.2 and causal-conv1d 1.7.0 follow the
upstream Modal recipe. Triton is explicitly 3.7.1: this intentionally overrides
Torch's 3.4.0 dependency pin because the author's Hopper recipe requires
Triton >=3.7.1. This known metadata conflict is not permission to upgrade Torch.
All runtime code and dependencies are installed at build time.

## Build and qualify

From the root of a clean archive of the released `Phala-Network/kev` commit,
on the authorized Linux/amd64 Docker builder:

```sh
docker build -f phala/Dockerfile --pull --platform linux/amd64 --build-arg SOURCE_REVISION=COMMIT --build-arg VERSION=v0.1.1 -t ghcr.io/phala-network/kev:v0.1.1 .
docker run --rm --network none --entrypoint hf ghcr.io/phala-network/kev:v0.1.1 --help
docker run --rm --network none --entrypoint hf ghcr.io/phala-network/kev:v0.1.1 download --help
docker run --rm --network none ghcr.io/phala-network/kev:v0.1.1 python /opt/kev-verify/verify_image.py
docker run --rm --network none --gpus all ghcr.io/phala-network/kev:v0.1.1 python /opt/kev-verify/verify_image.py --gpu
```

The final build executes both HF commands and checks CPU-safe imports, versions
and CLI. CUDA extension/FLA imports are checked only with `--gpu`, because Triton
may initialise a CUDA driver during import. Even that check does not prove model
inference. Publish the build inputs at a source
commit, immutable tag and published Release before pushing the image; record and
read back the resulting registry digest. The source repository and OCI source label
are `https://github.com/Phala-Network/kev`; `SOURCE_REVISION` must identify the exact
fork commit used as the build context. Do not rebuild from an upstream download or
use a deployment Compose commit as runtime provenance. Deploy only a tag-plus-digest reference.
The build has no credentials and includes no model weights.

## Model artifacts and launch

Use this same image for two direct `hf download` one-shot services sharing an
HF cache; the backend waits for both successful downloads:

| Artifact | Exact revision |
| --- | --- |
| `jaredpalmer/kev-4b` | `6cfce5c2fa4b4bd64026336ab649c5ca78857d52` |
| `Qwen/Qwen3.5-4B-Base` | `1001bb4d826a52d1f399e183466143f4da7b741b` |

Point `--run` **and** `--fallback` to the same exact Kev snapshot path to prevent
the author's local-development fallback from selecting another checkpoint.
Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` in the serving container,
with `HF_HOME` identical to both downloaders. The checkpoint's `head.pt` records
the base revision; verify that metadata against the table before acceptance.
The head SHA-256 is `dd633435998ecc751ac538717a3742e32149500fabf7d7276287dbf0693f347c`;
the adapter SHA-256 is `90e817356246e7f18bfa7ca3d31794cd4fbeb3332a66a84cb51d9ceae925f2b2`.
Do not override the checkpoint's calibration temperature.

Launch with an explicit flat command such as `python -m kev.serve --run SNAPSHOT
--fallback SNAPSHOT --host 0.0.0.0 --port 8008`. Assign the same GPU as Nemotron
but a separate process and JIT cache. Preserve Nemotron's model/context/precision
and PIG. Start Nemotron first so its static allocation is established before Kev.
Kev needs actual free memory for weights, graph buffers, cached states and the
largest accepted forward pass; the author's 14.3 GB residency is not a peak bound.
Do not treat SGLang `mem-fraction-static` as a hard cap on all dynamic allocations.

Use existing unified `TOKEN` via `KEV_API_KEY: ${TOKEN:?...}` in Kev and
`TOKEN: ${TOKEN:?...}` in TAIL. No credential goes in command arguments or image
layers. TAIL owns public authentication, not admission or a Governor profile.
TAIL v0.1.2 does not forward `/v1/systemone`; use the separately qualified TAIL
release that adds the exact POST route. Keep the backend port private. A private
curl probe of `/openapi.json` proves that the server has loaded and begun serving;
public health/docs/native paths must not bypass the intended ingress auth policy.

Qualify through the actual TAIL chain: missing/wrong/correct token, models with
CUDA/bf16 and exact checkpoint, choice/noul/score probability shape and finite
normalised values, repeated-state cache correctness, invalid payload 422,
overlength refusal without silent truncation, and recovery after disconnect.
Run one overlapping Nemotron stream and Kev decision to prove same-GPU coexistence,
then confirm Nemotron's queue/reservations clear and Kev's queued count returns to
zero without OOM/restart. Include an appropriately sized long state for memory
qualification; short decisions alone do not qualify the native 65,536-state-token
limit. No capacity sweep or throughput claim is implied by these functional gates.
