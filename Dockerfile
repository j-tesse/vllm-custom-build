# vLLM from source on NVIDIA's 26.11-py3 base (torch 2.15.0a0, CUDA 13.5, NVIDIA
# vLLM 0.33.0). Requires driver >= CUDA 13.5 UMD (620.70.10 here).
#
# Bumped from 26.06 on 2026-11-26: upstream main moved to torch 2.14
# (30e79c3a766, #52183), and DeepGEMM's torch::stable usage (TORCH_VERSION_2_14_0,
# stable::permute/index_select/bitwise_*) no longer compiles against 26.06's
# torch 2.13. Source pins torch==2.14.0; NVIDIA's 2.15a0 is a superset of that
# stable API. Dockerfile.26.06 / Dockerfile.26.03 archive the previous builds.

# Stage 1: Build the vLLM wheel against the base's native torch + nvcc 13.5.
FROM nvcr.io/nvidia/vllm:26.11-py3 AS builder

ARG TORCH_CUDA_ARCH_LIST="12.0"
ARG MAX_JOBS=16
ARG NVCC_THREADS=4

WORKDIR /workspace/vllm-build

# Remove old vLLM (keep PyTorch and all other CUDA libs)
RUN pip uninstall -y vllm vllm-flash-attn 2>/dev/null || true

# elfutils headers: DeepGEMM's deep_jit includes <elfutils/libdwfl.h> (upstream
# #56876 added elfutils-devel to its build stage). Without it deepgemm_C fails.
# Still absent from 26.11.
RUN apt-get update && apt-get install -y --no-install-recommends libdw-dev \
    && rm -rf /var/lib/apt/lists/*

# Build tools (setuptools_rust imported at setup.py module load; see 26.03 notes).
RUN pip install --no-cache-dir setuptools wheel cmake ninja packaging setuptools_scm setuptools_rust

COPY vllm/ /workspace/vllm-build/

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV MAX_JOBS=${MAX_JOBS}
ENV NVCC_THREADS=${NVCC_THREADS}
ENV VLLM_DOCKER_BUILD_CONTEXT=1

# Pre-create vllm_flash_attn/cute so find_packages() discovers it at setup() time.
RUN mkdir -p vllm/vllm_flash_attn/cute && touch vllm/vllm_flash_attn/cute/__init__.py

# Unset VLLM_FLASH_ATTN_SRC_DIR so cmake clones flash-attention fresh instead of
# symlinking to the base's older snapshot (bdist_wheel trips on the dangling cute
# symlink otherwise). 26.11 still sets it to /opt/vllm/vllm-flash-attn/.
RUN env -u VLLM_FLASH_ATTN_SRC_DIR python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38


# Stage 2: Runtime — 26.11 base + freshly built wheel.
FROM nvcr.io/nvidia/vllm:26.11-py3

# 26.11 ships per-arch flashinfer-jit-cache-sm* wheels (0.7.2 builds) besides
# flashinfer-jit-cache. Drop them all: we pin a different flashinfer below, and
# a stale prebuilt JIT cache for another version must not be picked up.
# The uninstall leaves an empty flashinfer_jit_cache/providers/ dir behind. That
# dir still imports as a namespace package, so flashinfer's jit/env.py thinks
# the cache is installed and dies at import on `flashinfer_jit_cache.__version__`
# (first 26.11 build, 2026-11-26). rm it and assert the import really fails.
RUN pip uninstall -y vllm vllm-flash-attn flashinfer-cubin \
        $(pip list --format=freeze 2>/dev/null | grep -io '^flashinfer-jit-cache[^=]*') \
        2>/dev/null || true; \
    rm -rf /usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache && \
    ! pip list 2>/dev/null | grep -qi '^flashinfer-jit-cache' && \
    ! python3 -c "import flashinfer_jit_cache" 2>/dev/null

# --no-deps preserves NVIDIA's custom torch (the wheel's torch==2.14.0 pin would
# otherwise pull stock torch and cause a c10 ABI mismatch).
COPY --from=builder /workspace/vllm-build/dist/*.whl /tmp/
RUN pip install --no-cache-dir --no-deps /tmp/*.whl && rm /tmp/*.whl

# transformers: NO pin needed on 26.11 — it ships 5.20.2, inside source's
# >=5.16.1,<5.25 and new enough for Gemma 4 MTP's `gemma4_assistant` (>=5.12).
# On 26.06 (5.6.0) this pin was load-bearing; keep the guard so a future base
# that regresses fails here instead of at Gemma 4 config load.
RUN python3 -c "from transformers.models.auto.configuration_auto import CONFIG_MAPPING; assert 'gemma4_assistant' in CONFIG_MAPPING"

# Runtime deps where source is ahead of (or, tilelang, behind) 26.11's vLLM 0.33.0
# baseline. --no-deps wheel install leaves the base's versions in place, so pin
# explicitly to match vllm/requirements/{common,cuda}.txt. AUDIT ON EVERY UPDATE
# (b25 shipped humming 0.1.10 vs source's 0.1.16 → every config crash-looped at
# config load on `humming.dtypes.float4e0m3`). Audit snippet: run a script that
# checks packaging.Requirement(line).specifier against importlib.metadata for
# each line of both files inside the built image; ignore torch/torchvision/
# torchaudio/torchcodec/PyNvVideoCodec (NVIDIA torch stays) and numba skew.
# Everything else in those files (cutlass-dsl 4.8.0, quack 0.6.5, tokenspeed
# 2.0.3, tvm-ffi 0.1.11, the common.txt floors) is satisfied by 26.11 as-is.
RUN pip install --no-cache-dir \
    'humming-kernels[cu13]==0.1.20' \
    tilelang==0.1.12 \
    compressed-tensors==0.20.1 && \
    python3 -c "import humming.dtypes as d; d.float4e0m3" && \
    python3 -c "import torch; assert 'nv' in torch.__version__, f'NVIDIA torch got clobbered: {torch.__version__}'"

# xgrammar: 26.11 ships 0.2.8; source pins 0.2.9.
RUN pip install --no-cache-dir xgrammar==0.2.9 && \
    python3 -c "from xgrammar import StructuralTag, normalize_tool_choice" && \
    python3 -c "import torch; assert 'nv' in torch.__version__, f'NVIDIA torch got clobbered: {torch.__version__}'"

# cute-DSL FA4 stack: base already matches source (nvidia-cutlass-dsl 4.8.0 +
# quack-kernels 0.6.5). These two move together — each quack release declares
# an exact cutlass-dsl version — so if source bumps one, pin both here, never
# one alone. The guard imports the modules vLLM's cute-DSL kernel warmup pulls
# in (the 26.03 ThrMma crash-loop came from a mismatched pair).
RUN python3 -c "from quack.compile_utils import make_fake_tensor; import quack.layout_utils"

# flashinfer: 26.11 ships an NVIDIA 0.7.2.post1 build. Keep this pin equal to
# `vllm/requirements/cuda.txt` — it moves with source (0.6.14 → 0.6.18
# 2026-08-30; → 0.7.0.post1 at b26; → 0.7.4 on 26.11). cubin left PyPI at
# 0.6.14 → needs flashinfer's own index (--extra-index-url).
#
# TWO guards, both for real skew we have shipped:
#   kv_cache_sf     — the fp8-KV NVFP4 prefill regression.
#   q_cu_seq_lens   — added to the XQA decode kernel by vLLM #52157/#49718.
#     b19 shipped source that passes it against 0.6.14, which doesn't take it:
#     every SM12x FlashInfer config crash-looped in cudagraph capture with
#     "TypeError: ... unexpected keyword argument 'q_cu_seq_lens'". The
#     kv_cache_sf guard passed (0.6.14 has it), so nothing caught the build.
#     The XQA path is sm_120-gated, i.e. this box only — CI upstream won't
#     cover it for us. Import from flashinfer.decode, the same module vLLM's
#     _lazy_import_wrapper uses; it is not re-exported at package top level.
#
# The `nvidia-cutlass-dsl[cu13]` repeat is load-bearing. flashinfer declares
# `nvidia-cutlass-dsl>=4.6.2a0` — the `a0` suffix makes pip accept PRE-RELEASES,
# and the flashinfer extra-index serves `4.8.0.dev0`. Without the repeat this
# step silently swaps in that dev build, whose _mlir libs are out of sync with
# their Python shim ("...ir has no attribute 'OpOperands'", 2026-09-04).
RUN pip install --no-cache-dir --extra-index-url https://flashinfer.ai/whl/ \
        flashinfer-python==0.7.4 flashinfer-cubin==0.7.4 \
        'nvidia-cutlass-dsl[cu13]==4.8.0' && \
    python3 -c "import importlib.metadata as m; v = m.version('nvidia-cutlass-dsl'); assert v == '4.8.0', f'cute-DSL pin was clobbered by a flashinfer dep: {v}'" && \
    python3 -c "from quack.compile_utils import make_fake_tensor; import quack.layout_utils" && \
    python3 -c "import inspect, flashinfer; from flashinfer import BatchPrefillWithPagedKVCacheWrapper as W; assert 'kv_cache_sf' in inspect.signature(W.run).parameters, 'FlashInfer run() is missing kv_cache_sf — version skew with vLLM main'" && \
    python3 -c "import inspect; from flashinfer.decode import xqa_batch_decode_with_kv_cache as f; assert 'q_cu_seq_lens' in inspect.signature(f).parameters, 'FlashInfer xqa_batch_decode_with_kv_cache() is missing q_cu_seq_lens — version skew with vLLM main'" && \
    python3 -c "import torch; assert 'nv' in torch.__version__, f'NVIDIA torch got clobbered: {torch.__version__}'"

# b12x: DROPPED on 26.11. Source's setup.py `b12x` extra is now empty ("kept for
# backwards compatibility"), and no vllm.sh config has used it since Nemotron
# Bolt was removed (2026-10-17).

# K2-Horizon MoVA INT4 (vllm.sh `2`): cyankiwi fork commit 46485cf572 minus its
# test file. Upstream k2_horizon.py stacks `expert.weight` for the MoVA value
# experts, which only exists unquantized, so every INT4 checkpoint dies in
# profile_run with "'ColumnParallelLinear' object has no attribute 'weight'".
# The patch folds the compressed-tensors packed experts into the existing
# use_int4_w4a16 fused-MoE layout (they stay 4-bit) and fixes
# _ensure_block_size_k_divisible for group_size=32. Upstream's equivalent
# (#56637) is Intel-only, hence the fork. Rebased (offsets only, zero fuzz) onto
# 46e324e5827 on 2026-11-26.
# k2_horizon_mova_int4_compile.patch is OURS, layered on top. It moves the fork's
# quantized path into an opaque custom op (torch.ops.vllm.k2_horizon_mova_int4).
# The fork picks the int4 tile config per call via try_get_optimal_moe_config,
# which reads JSON + calls NVML, and so is untraceable under vLLM's fullgraph
# compile: EngineCore died in profile_run with "Dynamo does not know how to
# trace method `__call__` of class `_FuncPtr`" (the fork was evidently only
# run eager). It also makes the packed tensors buffers, so they are not
# baked into the graph as constants.
# Applied HERE, against site-packages, not in the builder: both files are pure
# Python, so the builder (the ~60-min CUDA compile) stays cache-hittable.
# `--forward` fails the build on a reject OR on "already applied". When
# upstream merges a fix, drop this block rather than forcing it.
COPY k2_horizon_mova_int4.patch k2_horizon_mova_int4_compile.patch /tmp/
RUN cd /usr/local/lib/python3.12/dist-packages && \
    patch -p1 --forward --no-backup-if-mismatch < /tmp/k2_horizon_mova_int4.patch && \
    patch -p1 --forward --no-backup-if-mismatch < /tmp/k2_horizon_mova_int4_compile.patch && \
    rm /tmp/k2_horizon_mova_int4*.patch && \
    grep -q 'def pack_quantized_v_experts' vllm/model_executor/models/k2_horizon.py && \
    grep -q 'torch.ops.vllm.k2_horizon_mova_int4(' vllm/model_executor/models/k2_horizon.py && \
    python3 -c "import vllm.model_executor.models.k2_horizon; from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32"

COPY --from=builder /workspace/vllm-build/examples /workspace/examples
COPY --from=builder /workspace/vllm-build/benchmarks /workspace/benchmarks

ENV VLLM_USAGE_SOURCE=production-docker-image
ENTRYPOINT ["vllm", "serve"]
