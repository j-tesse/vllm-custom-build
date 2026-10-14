# vLLM Custom Docker Build

This directory builds the base vLLM Docker image. It does **not** run models directly — model containers are launched via `vllm.sh` in this directory (aliased as `vllm` in `~/.bash_aliases`), which uses this image as its base.

The image targets an RTX 5090 (Blackwell, compute 12.0) with cutting-edge CUDA drivers.

## Documentation

General information goes to CLAUDE.md; information related to `vllm.sh` goes right into the script file (as comments).

Documentation must be kept in sync and useful. Use terse language, avoid fluffy or chatty sections. Focus on the "why" if the "how" is obvious from looking at the code. Target audience is a coding agent such as yourself.

## Why a custom build?

We build vLLM **from current upstream source** on top of NVIDIA's `nvcr.io/nvidia/vllm` container so we can track bleeding-edge vLLM (features/fixes ahead of any NVIDIA release) while keeping NVIDIA's Blackwell-tuned CUDA/PyTorch runtime. The official `vllm/vllm-openai:latest` image lags and its CUDA/torch combo isn't matched to the RTX 5090 + our driver, so a custom build is needed to reconcile NVIDIA's base torch with the newer vLLM source and its CUDA-extension deps.

> **HISTORY:** The original rationale (below, kept for context) was about **driver 580.x**, which capped at CUDA 13.0 and forced an `nvcc 13.0` override + the `26.03` base pin. The driver is now **610.43.02 (CUDA 13.3)**, so on **2026-07-17** the base was bumped to **`26.06-py3`** (torch 2.13, CUDA 13.3) and all the driver-580 workarounds were deleted — see "Base image (26.06)" below. The paragraphs that follow describe the retired 580.x constraints.
>
> _(Retired 580.x rationale)_ The official image used CUDA 12.9, incompatible with driver 580.x; standard PyTorch cu130 triggered forward-compat error 803. Driver 580.x needed the CUDA 13.1 runtime (only in NVIDIA's internal PyTorch), vLLM's CUDA extensions needed full dev headers absent from the NVIDIA container, and nvcc 13.1 generated PTX driver 580.x couldn't run — so we substituted the CUDA 13.0 compiler.

## Hardware

| Component | Details |
|-----------|---------|
| GPU | RTX 5090 (Blackwell, sm_120) |
| Driver | 610.43.02 (CUDA 13.3 runtime; UMD 13.3) |
| CPU | Ryzen 9 9900X (12-core / 24 threads) |
| RAM | ~91 GB |

## Files

| File | Purpose |
|------|---------|
| `update.sh` | Updates `vllm/` to latest upstream (fast-forward only) and kicks off an async build. Automates the mechanical part of `/latest`; does **not** watch or fix the build |
| `build-async.sh` | **Preferred build script** — detaches from terminal, survives SSH disconnects |
| `build.sh` | Synchronous build script (called internally by `build-async.sh`) |
| `Dockerfile` | 2-stage Dockerfile, base `26.06-py3` (see below) |
| `Dockerfile.26.03` | Archived previous 3-stage build (driver-580.x era); kept for reference/rollback |
| `PLAN-base-26.06.md` | Base-bump plan + measured dependency audit |
| `k2_horizon_mova_int4.patch`, `k2_horizon_mova_int4_compile.patch` | Fork patch + our torch.compile fix, applied to site-packages in the runtime stage; let `vllm.sh 2` serve INT4 MoVA value experts. See below |
| `vllm/` | vLLM source, tracked as a **git submodule** → `github.com/vllm-project/vllm` (pins the built commit) |
| `vllm.sh` | Interactive Docker model menu (aliased as `vllm`); launches model containers from `vllm-custom:latest` |
| `build.number` | Persisted monotonic build counter; `build.sh` increments it (on success only) and tags the image `vllm-custom:b<N>` |
| `build.history` | Append-only log of successful builds: build number, date/time, and vLLM version. Distinct from `build.log` (full build output) |
| `bench-inference.ts` | Inference benchmark against a **running** server (TTFT, decode tok/s, MTP acceptance). Node-native TS, no deps. Appends each run to `bench-results/bench.jsonl`; `--report` prints the saved history as a table |
| `bench-results/` | Gitignored benchmark history (JSONL, one record per run). Read it with `bench-inference.ts --report` |

## Dockerfile strategy (2 stages, base `26.06-py3`)

1. **`builder`** — starts from `nvcr.io/nvidia/vllm:26.06-py3` (torch 2.13, CUDA 13.3) and builds vLLM from current source into a wheel using the base's **native nvcc 13.3**.
2. **Runtime** — fresh `nvcr.io/nvidia/vllm:26.06-py3` base, installs the wheel (`--no-deps` to preserve NVIDIA's custom torch), then pins the handful of runtime deps that current source needs newer than the base's vLLM 0.22.1.

> **26.03 → 26.06 bump (2026-07-17):** the old `cuda-devel` stage + nvcc-13.0 override, the `register_opaque_type` hoist sed, and the `cuda_view.cu` patch were all **deleted** — verified unnecessary on torch 2.13 / driver 610. The previous 3-stage 26.03 Dockerfile is archived as `Dockerfile.26.03`; see `PLAN-base-26.06.md` for the measured dep audit. Validated by building b14 and serving `vllm.sh 2` (Qwen NVFP4 MoE + MTP) end-to-end.

### Pins the runtime stage still needs (source ahead of 26.06's vLLM 0.22.1)
Because we build bleeding-edge source on a base tuned for 0.22.1, these stay pinned (guards fail-fast if a symbol/torch regresses). **The Dockerfile is the source of truth for versions**; as of b26: `transformers==5.19.0`, `xgrammar==0.2.8`, `flashinfer-python/-cubin==0.7.0.post1` (+`--extra-index-url https://flashinfer.ai/whl/`), `humming-kernels==0.1.16`, `nvidia-cutlass-dsl==4.8.0` + `quack-kernels==0.6.5` (matched cute-DSL pair), plus tokenspeed-mla, tilelang, tvm-ffi, compressed-tensors, and common.txt floors (hf_hub, tokenizers, mcp 2.x, oss-harmony, instanttensor, …).

**Audit every pin on every `vllm/` update, not just flashinfer.** b25 (2026-10-12) built clean but every config crash-looped at config load (`humming.dtypes has no attribute float4e0m3`) because a large upstream jump moved ~20 requirements past our pins. The fix is runtime-stage only (builder stays cached, ~4 min rebuild). Audit: inside the built image, check each line of `vllm/requirements/{common,cuda}.txt` with `packaging.Requirement(...).specifier.contains(importlib.metadata.version(name))`; expected leftover skew is only torch/torchvision/torchaudio/torchcodec/PyNvVideoCodec (NVIDIA torch, deliberately kept) and numba patch versions.

**The flashinfer pin tracks `vllm/requirements/cuda.txt` — diff it on every update.** It is the one pin upstream moves *for* us, and a stale one is not caught by the build: b19 (2026-09-04) built clean against 0.6.14 while source had moved to 0.6.18, then crash-looped every config at cudagraph capture with `TypeError: xqa_batch_decode_with_kv_cache() got an unexpected keyword argument 'q_cu_seq_lens'`. The XQA decode path is gated on `is_device_capability_family(120)`, so it is **sm_120 / this box only** — upstream CI does not cover it, and the failure surfaces at *runtime on the first model start*, not at build time. The `RUN` block now guards `q_cu_seq_lens` alongside `kv_cache_sf`; add a guard for whatever kwarg the next skew lands on rather than trusting the version number.

### K2-Horizon MoVA INT4 patch (`vllm.sh 2`, added 2026-09-28)
Upstream `k2_horizon.py` supports only bf16 MoVA value experts (`self_attn.v_experts.*`). Every INT4 checkpoint dies in `profile_run` with `'ColumnParallelLinear' object has no attribute 'weight'`. `k2_horizon_mova_int4.patch` is cyankiwi fork commit `46485cf572` minus its test file. It folds compressed-tensors packed experts into the existing `use_int4_w4a16` fused-MoE layout, and fixes `_ensure_block_size_k_divisible` for group_size 32. Upstream's #56637 is Intel-only.
- `k2_horizon_mova_int4_compile.patch` is ours and applies on top of the fork patch. The fork picks its int4 tile config per call (`try_get_optimal_moe_config`: JSON reads + NVML), which Dynamo can't trace under vLLM's fullgraph compile, so EngineCore died in profile_run (`_FuncPtr.__call__`). The fix moves that path into the opaque custom op `torch.ops.vllm.k2_horizon_mova_int4`. Validated on b24 (2026-09-28).
- Applied with `patch --forward` against **site-packages in the runtime stage**, not in the builder. Both files are pure Python, and this placement lets a patch-only change reuse the cached builder layers (the CUDA compile).
- **On a `vllm/` update the patch may reject, and the build then fails loudly (intended).** Check whether upstream has merged an equivalent first. If it has, delete the block and the patch file. If it hasn't, rebase both patches onto the new source.

## Transformers pin (load-bearing) & Gemma 4 MTP

The runtime stage installs the vLLM wheel with `--no-deps` (to preserve NVIDIA's CUDA 13.1 torch), so **transformers is never pulled as a wheel dependency** — the version is whatever the explicit `RUN pip install transformers==<X>` line sets. Removing that line leaves the base image's transformers **4.57.5**, which is 4.x and doesn't recognize the `gemma4` architecture at all → every Gemma 4 config breaks at model load. The pin is required for all Gemma 4 options, not optional.

**Pinned to `transformers==5.19.0`** (5.12.1 from 2026-07-02 until b26; source now requires >=5.16.1,<5.20) to enable Gemma 4 MTP:

- Gemma 4 MTP (Multi-Token Prediction) speculative decoding uses Google's drafter `google/gemma-4-31B-it-assistant` (~0.5B), enabled by `vllm.sh` configs **`4m`** (105k ctx) and **`4`** (the same stack at 90k ctx + a 24 GiB host-RAM KV tier; see below), with a KV budget sized so the drafter weights fit at gpu-mem-util 0.982. **`vllm.sh`'s `4m` block is the source of truth for these numbers — it carries the full dated tuning log. Do not trust figures quoted here over that block.** Current: `CONTEXT_SIZE=105000` with `--num-gpu-blocks-override 6780`.
  - **The tuning lever changed on 2026-08-03.** Before that, the KV pool was whatever vLLM's profiler handed out and the only lever was trimming `CONTEXT_SIZE` under it. Since then the pool is **pinned** by `--num-gpu-blocks-override`, deliberately claiming more than the profiler offers (the profiler under-estimates because the torch.compile working set is still resident when the peak is measured) and making warm and cold boots identical. So **re-tuning now means adjusting the block override, not just lowering context.**
  - **Both directions are load-bearing, and they fail differently.** Too few blocks → the old failure: KV cache too small at startup, engine won't boot. Too many → the pool eats the activation headroom and the engine **boots fine, serves for hours, then dies mid-request** with a CUDA OOM on a prefill buffer, which `--restart unless-stopped` silently masks as a restart. Check `RestartCount`, not just health.
  - History: ran **128k** until the 2026-07-17 vLLM update (Qwen 3.6 + new default kernels/cudagraphs) dropped the pool to ~6.86 GiB (~85k tokens) and 128k OOMed at startup → retuned to `83968`. Raised to **105000** on 2026-08-03 once the 7000-block override pinned the pool at 119,182 tokens. Override cut to **6780** on 2026-09-09 (115,437 tokens, 1.10x) after b21's larger non-KV footprint left only 154.88 MiB free and a 168 MiB prefill buffer began crashing the engine hours into serving.
  - **Re-derive after every image bump — this is routinely forgotten** (it was skipped across b18→b21, which is what caused the 2026-09-09 crashes). Boot cold, read `GPU KV cache size: <N> tokens`, and confirm real headroom via `nvidia-smi` — note the `gpu_worker.py:871` "kv cache memory in use" line reports the *profiled* figure, not the pinned pool, so it does not reflect the override. All of this is a `vllm.sh`-only fix; no image rebuild.
  - **Config `4` has its own override (6400); never sync it to `4m`'s.** Its RAM KV tier forces it to run without expandable_segments, which costs ~500 MiB of headroom. Re-derive it separately after an image bump, using a long-prompt stress test rather than idle headroom. Details are in the `vllm.sh` `4` block.
- The drafter's `config.json` has `model_type: gemma4_assistant`. vLLM does **not** register this in its config registry (true even on upstream main) — it relies on **transformers** to recognize it, and that model_type only landed in **transformers 5.12.x**. `5.7.0` does NOT have it (despite the drafter stamping `5.7.0.dev0`); with anything <5.12 the engine crashes at config load: "Transformers does not recognize `gemma4_assistant`".
- Validated 2026-07-02: 5.12.1 boots cleanly against our wheel, preserves NVIDIA's torch, and drives the drafter at ~80–91% draft acceptance (mean accept length ~1.8 → near ~1.8× decode). MTP is text-only (drafter has no vision tower); image prompts still run on the target.

**The 26.06 base does NOT fix this** — it ships **transformers 5.6.0** (below 5.12.1, no `gemma4_assistant`), so the explicit `transformers==5.12.1` pin is still required after the base bump. (The base bump itself was done on 2026-07-17 and did retire the nvcc override / `cuda_view.cu` / hoist patches — see "Dockerfile strategy" above — but the transformers pin is independent of the base and stays.)

## DiffusionGemma (`vllm.sh` `3` / `3n`) — the one config with a constrained *request* shape

Added 2026-09-22, replacing the DeepSeek-R1-Distill-32B-AWQ config. Both validated on b22: `3` = `RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic` (20k ctx, 385 tok/s mean, 8.0 steps/canvas) and `3n` = `nvidia/diffusiongemma-26B-A4B-it-NVFP4` (**131k ctx, 444 tok/s mean, 9.8 steps/canvas** — faster *and* 6× the context, so prefer `3n` unless you specifically want the checkpoint upstream evaluates). **The `3|3n` block in `vllm.sh` is the source of truth** for flags and sizing; only the facts that outlive a retune are repeated here. No image work was needed — b22 already builds `diffusion_gemma.py`.

Every other config here is autoregressive; this one denoises a 256-token canvas per step. Two consequences worth knowing before touching anything:

- **Requests are validated differently.** vLLM 400s `temperature != 1.0`, `seed`, `min_p`, `min_tokens`, `logit_bias`, `bad_words`, `allowed_token_ids` (`sampling_params.py` `_validate_diffusion`) and all structured outputs / `response_format` — which also rules out `tool_choice: "required"` or a named function. `--enable-auto-tool-choice` is fine (text parsing). `vllm test` and `bench-inference.ts` were fixed in the same change to stop sending an explicit temperature; **any other client that sends one will fail against these two configs only.** That includes whatever `sync_agent_models` points pi/OpenClaw at.
- **Never add `--speculative-config`.** `canvas_length` *is* `num_speculative_tokens` for this model (`config/vllm.py`), and a spec config silently overrides it.

Two traps for a future image bump:

- **Attention backend.** The model passes a per-request causal *tensor*; only TRITON_ATTN and FA4 handle it, and FlashInfer is hard-rejected. On sm_120 FA falls back to FA2 (FA4 is auto-selected only for capability major 10), so the configs pass `--attention-backend TRITON_ATTN` explicitly rather than trusting the Gemma 4 auto-fallback. Upstream validates the mixed-causal kernels on SM90/SM100 only — this box is not covered by their CI.
- **Memory is tight on the FP8 variant, and the lever is not the obvious one.** Measured on b22: weights 25.86 GiB, KV 68–95 KiB/token (it is *not* cheap — the 25 sliding layers draw from the same pool as the 5 full-attention ones, so the 1024-token window does not bound it). NVFP4 is ~17.5 GiB instead, because modelopt quantized **only the MoE experts** (attention, dense MLPs, router, vision tower and lm_head stay bf16) — don't expect a 4× shrink there either.
  - The KV pool always expands to fill `util × TOTAL`, so **lowering `CONTEXT_SIZE` alone frees nothing**; utilization (or `--num-gpu-blocks-override`) is the lever, exactly as for configs 4/4m.
  - What warmup needs lives in the gap *outside* that budget, and on this model it is large: the diffusion sampler materializes `[seqs, canvas, vocab]` fp32 buffers, i.e. **268 MiB per sequence per buffer**. `--max-num-seqs 1` is load-bearing, not a default — with 2, this config OOMed in inductor autotune at 0.95 *and* 0.93 utilization.
  - This box also has a permanent ~500 MiB GPU resident (the `laya` container), which caps utilization at ~0.967 before vLLM's startup free-memory check refuses to boot at all. **Configs `4`/`4m` are pinned at 0.982 and would hit that same check on a cold boot while laya holds its allocation** — they work today only because laya allocates lazily.

## Building

> **Routine update + build:** run `./update.sh` — it fast-forwards `vllm/` to upstream `main` (aborts on a non-fast-forward or a dirty tree; never forces), reports the version delta, and starts a detached build. Pass `--no-build` to update only. It does **not** block on the build and does **not** diagnose failures — a large upstream jump can break a grep-guarded Dockerfile patch or pin, which needs a human/agent (the `/latest` skill covers watch-and-fix).

> **Always use `build-async.sh`**, not `build.sh` directly. It survives SSH disconnects, captures the exit code, and supports agent-friendly polling.

```bash
./build-async.sh start       # start detached build
./build-async.sh status      # human-readable status + last 20 log lines
./build-async.sh status --json  # machine-readable JSON (for agents)
./build-async.sh wait        # block until done, exits with build's exit code
./build-async.sh log         # tail -f the live log
./build-async.sh stop        # kill the running build
```

Blocking agent workflow:
```bash
./build-async.sh start && ./build-async.sh wait
```

Key output files (in this directory):

| File | Contents |
|------|----------|
| `build.log` | Full stdout+stderr of the build |
| `build.pid` | PID of running build (deleted on completion) |
| `build.exit` | Exit code on completion (`0` = success) |

Grep-able log markers: `BUILD_STARTED`, `BUILD_SUCCESS`, `BUILD_FAILED`

Build args (editable in `build.sh`):

| Arg | Default | Notes |
|-----|---------|-------|
| `TORCH_CUDA_ARCH_LIST` | `12.0` | Blackwell only — keeps build fast |
| `MAX_JOBS` | `16` | Leaves 8 threads for the system |
| `NVCC_THREADS` | `4` | Parallelism inside nvcc |

Expected build time: 60 minutes. Final image: ~15 GB.

## Output image tags

Every build applies three tags:

- `vllm-custom:b<N>` — monotonically increasing build number (from `build.number`, bumped on success only)
- `vllm-custom:<git-describe>` — vLLM's own version, e.g. `vllm-custom:v0.22.1rc0-19-g50c80d79230`
- `vllm-custom:latest` — always points at the most recent build

## After building

1. Verify: `docker images | grep vllm-custom`
2. No tag edit needed — `vllm.sh` is pinned to `vllm-custom:latest`, which always tracks the newest build. (Pin to a specific `b<N>` tag only if you need to roll back.)
3. Test: `./vllm.sh rec 4` (removes and re-creates the vLLM docker container for Gemma4:31b)

## Base image bump to `26.06-py3` (DONE 2026-07-17, b14)

The base was moved from `26.03.post1` to `nvcr.io/nvidia/vllm:26.06-py3` (torch 2.13, CUDA 13.3, NVIDIA vLLM 0.22.1) on driver 610. Outcome:

- Deleted the `cuda-devel` stage, nvcc-13.0 override, `register_opaque_type` hoist sed, and `cuda_view.cu` patch — all verified unnecessary on torch 2.13 (register_opaque_type has `hoist` natively; `cuda_view.cu` compiles unpatched against 2.13's stable ABI).
- Dropped the `openai==2.45.0` and `fastsafetensors` pins (26.06 satisfies them).
- Kept `transformers==5.12.1` and the CUDA/kernel pins listed under "Dockerfile strategy".
- Fixed the cutlass/quack `ThrMma` crash via the matched `nvidia-cutlass-dsl==4.6.0` + `quack-kernels==0.6.1` pair.

Validated: b14 serves `vllm.sh 2` (Qwen NVFP4 MoE + MTP) end-to-end. `Dockerfile.26.03` archives the previous build; `PLAN-base-26.06.md` has the full audit/rationale.

> **Reminder for the next base bump:** we track bleeding-edge vLLM source, so *any* fixed NVIDIA base will lag some deps — expect to keep a set of CUDA/kernel pins in the runtime stage no matter what. Trial in a separate `Dockerfile.<ver>`, diff shipped versions against `vllm/requirements/{common,cuda}.txt` (see `PLAN-base-26.06.md` step 0 for the audit snippet), and validate `vllm.sh 2` (the cute-DSL warmup path) before promoting.
