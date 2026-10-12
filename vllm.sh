#!/bin/bash

################################################################################
# vLLM Model Management Script — interactive Docker menu for a vLLM server.
#
# Box: Ryzen 9 9900X / RTX 5090 (32GB VRAM) / 96GB RAM / Ubuntu 24.04.
# Needs docker (nvidia runtime), curl, python3. OpenAI API on port 8000.
# Only one model runs at a time (single GPU).
#
# Functions:
#   vllm()              interactive menu, or pass an option/alias
#   get_vllm_model_id() print the running server's model ID
#
# Menu (see the case statement for per-config details & quantization notes):
#   1  Qwen3.8-27B QAT-NVFP4      (110k ctx, bf16 KV; '1c'/'1k' = old unsloth PTQ)
#   2  K2-Horizon-MoVA-36B-A4B INT4 (cyankiwi AWQ; 44k ctx; needs image patches)
#   3  DiffusionGemma-26B-A4B FP8   (20k ctx, block diffusion; '3n' = NVFP4)
#   4  Gemma-4-31B-IT-NVFP4       (RedHatAI+MTP, 90k ctx + 24GB RAM KV tier; '4m' 105k no RAM tier, '4f' fp4 KV, '4mc' turbo text)
#   5  Nemotron-3.5-Lightning-30B-A3B-NVFP4 (Mamba/MoE hybrid, native MTP)
#
# Control verbs are WORDS, never numbers. The old numeric 5/6/7 (logs/test/stop)
# were retired when Nemotron took slot 5 — a digit now always means "start a
# model". Each verb is matched directly by its case branch, so the same token
# works both as an argument and as an answer to the interactive prompt:
#   logs | l     live logs
#   test | t     test inference against the running model
#   stop | s | kill | k    stop all running vLLM containers
#
# Usage:
#   vllm | vllm 1 | vllm 1c | vllm 5 | vllm logs | vllm stop
#   vllm status       list vllm_* containers (running + stopped) via `docker ps`
#   vllm ctx          show ctx length + KV-cache concurrency of running model
#   vllm rec [<cfg>]  recreate a container; bare form infers running config + confirms
#                     (-y / `yes |` skips), `vllm rec <cfg>` recreates without prompting
#
# CPU offload (--cpu-offload-gb): keeps weights in RAM for models > 32GB VRAM.
# Much slower — PCIe ~32 GB/s vs VRAM ~1000 GB/s. All current configs use 0.
#
# Notes:
# - Stopped containers are restarted (not recreated) if they exist.
# - Starting a model (fresh run, restart, or `rec`) first stops every OTHER
#   running vllm_* container and waits for its VRAM to drain — one GPU, one port.
#   So switching configs needs no manual `vllm stop`.
# - `vllm test` reads multi-line input, terminated by "END" on its own line.
# - vLLM sources for valid flags/quant names: ~/src/vllm/vllm
#   (e.g. quantization/__init__.py under model_executor/layers).
################################################################################

# Default image (a case branch may override per config). Images use `vllm serve`
# as ENTRYPOINT, so the model is a positional arg (CMD_PREFIX empty). Do NOT add
# `--model`: vLLM main deprecated it and crashes (IndexError in argparse_utils).
#   vllm/vllm-openai:latest          upstream stable, CUDA 13, has Gemma 4
#   nvcr.io/nvidia/vllm:26.03.post1-py3  NVIDIA-built, ~2 minors behind (no Gemma 4)
#   vllm-custom:latest               local CUDA 13.1 build of vLLM main (Blackwell)
DOCKER_IMAGE="vllm/vllm-openai:latest"
CMD_PREFIX=()

function sync_agent_models() {
  # Point BOTH agents — pi (~/.pi/agent) and OpenClaw (~/.openclaw/openclaw.json)
  # — at the model we just started, by delegating to update-agent-models.sh.
  #
  # Call this AFTER wait_for_healthy, never before: that script reads the id and
  # max_model_len actually being SERVED from /v1/models, which is the truth even
  # on the stopped-container restart path (where --max-model-len is baked in at
  # create time and can differ from CONTEXT_SIZE in this file). The old
  # pre-start, pi-only version wrote CONTEXT_SIZE and could disagree with the
  # running server. INPUT_MODES has to be passed in — /v1/models doesn't report
  # modalities — and CONTEXT_WINDOW is only a fallback for endpoints that omit
  # max_model_len (vLLM reports it; colibrì doesn't).
  local input_modes="$1" ctx="$2"
  local sync=~/scripts/update-agent-models.sh
  [ -x "$sync" ] || return 0
  INPUT_MODES="$input_modes" CONTEXT_WINDOW="$ctx" "$sync" \
    || echo "Warning: agent model sync failed — pi/OpenClaw configs left unchanged."
}

function log_vllm_concurrency() {
  # Surface vLLM's KV-cache concurrency log line ("Maximum concurrency for N
  # tokens per request: X"), i.e. how many parallel requests fit the KV cache.
  local container="$1"
  local CYAN='\033[96m' BOLD='\033[1m' RESET='\033[0m'
  local line
  line=$(docker logs "$container" 2>&1 | grep "Maximum concurrency" | tail -n1)
  [ -n "$line" ] && echo -e "${CYAN}${BOLD}${line#*\] }${RESET}"
}

function wait_for_healthy() {
  # Poll the server's /health endpoint OURSELVES, report elapsed time, append a row
  # to vllm-startups.csv, and surface KV concurrency. Args: container, start-epoch.
  # Shows a live elapsed-seconds counter (redrawn in place via \r) while waiting.
  #
  # This deliberately does NOT read `docker inspect .State.Health.Status`. That
  # field only changes when DOCKER runs a probe, and these containers are created
  # with --health-start-period 60s / --health-start-interval 1s / --health-interval
  # 60s: the 1s cadence applies ONLY inside the start period, so from t=60s the
  # status refreshes once a MINUTE. Every boot here takes longer than 60s, so the
  # old 1s loop spent its time re-reading a stale value and returned on the next
  # whole minute — quantizing every startup to a 60s grid. It shows up as 94 of 158
  # rows in vllm-startups.csv landing within 2s of a 60s multiple (121, 241, 301,
  # 361, 422, 601...). Measured on the 2026-08-27 gemma boot: ready at 69s
  # ("Application startup complete." in the log), reported at 120s — ~51s of pure
  # dead wait, ~30s on average, up to 59s. Probing /health ourselves is exact.
  #
  # It also takes effect on the SIX ALREADY-CREATED containers with no `vllm rec`,
  # which raising --health-start-period would not: healthcheck settings are baked
  # in at container-create time, this loop is not.
  #
  # The docker healthcheck is left in place on purpose — `vllm status` / `docker ps`
  # still report (healthy), just on their own lazy schedule. Nothing else in this
  # repo or ~/.bash_aliases reads .State.Health.Status (llama.sh has its own copy).
  #
  # Not reading Docker's verdict also means losing --health-retries as the failure
  # signal, so crashes are detected here instead. Containers run with `--restart
  # unless-stopped`, so a boot that dies is silently restarted and would otherwise
  # spin this loop forever: a bump in .RestartCount IS that crash. A state that is
  # neither running nor restarting is a container that stayed down.
  # The loop is otherwise unbounded, as before — 3am.sh wraps the call in `timeout`.
  local container="$1" start="$2" csv=~/scripts/vllm-startups.csv
  local dur state status restarts base_restarts
  # NB: RestartCount is TOP-LEVEL in the inspect JSON, not under .State — a
  # {{.State.RestartCount}} template fails with "map has no entry for key".
  base_restarts=$(docker inspect --format='{{.RestartCount}}' "$container" 2>/dev/null) || base_restarts=0
  while true; do
    dur=$(( $(date +%s) - start ))
    # One inspect per iteration covers both fields; a failure here means the
    # container is gone, per the empty-state branch.
    state=$(docker inspect --format='{{.State.Status}} {{.RestartCount}}' "$container" 2>/dev/null) || state=""
    status="${state%% *}"
    restarts="${state##* }"
    if [ -n "$state" ] && [ "$restarts" -gt "$base_restarts" ]; then
      printf "\r\033[KContainer crashed and was auto-restarted (%s -> %s) after %ds.\n" \
        "$base_restarts" "$restarts" "$dur"
      printf "Check logs with: docker logs %s\n" "$container"
      return 1
    fi
    case "$status" in
      running|restarting|created) ;;
      "")
        printf "\r\033[KContainer %s no longer exists.\n" "$container"
        return 1
        ;;
      *)
        printf "\r\033[KContainer %s is '%s' after %ds. Check logs with: docker logs %s\n" \
          "$container" "$status" "$dur" "$container"
        return 1
        ;;
    esac
    # Probe INSIDE the container's network namespace (same command the docker
    # healthcheck runs), not from the host. A host-side `curl localhost:8000`
    # only proves *something* holds the port, and would happily report a
    # half-started container healthy on the strength of a foreign listener —
    # observed while testing: a dummy container that had bound nothing was
    # declared healthy at 0s because the previous vLLM still answered on 8000.
    # wait_for_port_release makes that unlikely but not impossible (it gives up
    # and starts anyway after PORT_FREE_TIMEOUT). exec attributes the answer to
    # this container unconditionally, costs ~47ms, and fails fast (~7ms) when the
    # container isn't running. --max-time so a wedged server can't stall the loop.
    if [ "$status" = "running" ] \
       && docker exec "$container" curl -sf --max-time 3 http://localhost:8000/health >/dev/null 2>&1; then
      printf "\r\033[KContainer is healthy! (took %ds)\n" "$dur"
      [ -f "$csv" ] || echo "timestamp,container,duration_seconds" > "$csv"
      echo "$(date '+%Y-%m-%d %H:%M:%S'),$container,$dur" >> "$csv"
      log_vllm_concurrency "$container"
      return 0
    fi
    printf "\rWaiting for container to become healthy... (%ds)\033[K" "$dur"
    sleep 1
  done
}

function wait_for_gpu_release() {
  # Poll until the given host PIDs no longer hold CUDA memory, or GPU_FREE_TIMEOUT
  # (90s) passes. `docker stop` returns as soon as the container is gone, but the
  # card drains ASYNCHRONOUSLY — start vLLM into that window and it profiles its
  # KV pool against VRAM the old server still owns (same lesson as 3am.sh).
  # Only the PIDs handed in are watched, deliberately: other GPU users on this box
  # (ComfyUI, the SD WebUI) legitimately keep the card busy and must never block a
  # vLLM start the way a "wait for 0 MiB" check would.
  local pids="$1" deadline live pid busy
  [ -n "${pids// /}" ] || return 0
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  deadline=$(( $(date +%s) + ${GPU_FREE_TIMEOUT:-90} ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    live=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    busy=
    if [ -n "$live" ]; then
      for pid in $pids; do
        if echo "$live" | grep -qx -- "$pid"; then busy=1; break; fi
      done
    fi
    if [ -z "$busy" ]; then
      printf "\r\033[K"
      return 0
    fi
    printf "\rWaiting for the GPU to release VRAM from the stopped container...\033[K"
    sleep 1
  done
  printf "\r\033[KWarning: GPU still held after %ss — starting anyway.\n" "${GPU_FREE_TIMEOUT:-90}"
  return 0
}

function wait_for_port_release() {
  # Poll until the host TCP port is free, or PORT_FREE_TIMEOUT (30s) passes.
  # Same async-teardown lesson as wait_for_gpu_release, different resource:
  # `docker stop` returns before the daemon has torn down the port binding.
  #
  # Losing this race is far nastier than a failed start. Docker answers 500
  # ("Bind for 0.0.0.0:8000 failed: port is already allocated") and then EMPTIES
  # the container's persisted NetworkSettings.Networks map. That map — not
  # HostConfig.NetworkMode — is what a later `docker start` reads to decide what
  # to attach, so every subsequent start silently joins NOTHING: no sbJoin, no
  # error, HostConfig still showing `bridge` + the 8000 binding. The container
  # comes up with only `lo`, and vLLM dies in ModelConfig.__post_init__ resolving
  # huggingface.co ("Temporary failure in name resolution") — which reads as a
  # DNS/network fault rather than a Docker one. `docker start` will NEVER undo
  # it; only `docker rm` (i.e. `vllm rec <cfg>`) clears the map.
  # Cost on 2026-08-25: ~40min of misdiagnosis and ~120 crash-loop restarts.
  local port="${1:-8000}" deadline
  command -v ss >/dev/null 2>&1 || return 0
  deadline=$(( $(date +%s) + ${PORT_FREE_TIMEOUT:-30} ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if ! ss -tln "sport = :$port" 2>/dev/null | grep -q LISTEN; then
      printf "\r\033[K"
      return 0
    fi
    printf "\rWaiting for port %s to be released...\033[K" "$port"
    sleep 1
  done
  printf "\r\033[KWarning: port %s still bound after %ss — starting anyway.\n" "$port" "${PORT_FREE_TIMEOUT:-30}"
  return 0
}

function sample_token_usage() {
  # Fold the tokens served since the last cron sample into vllm-token-usage.json
  # BEFORE the server goes away. That script diffs the live /metrics counters
  # against its stored sample, and the counters reset to 0 when vLLM restarts —
  # so everything served between the last 10-minute cron tick and a stop is lost
  # unless we take one final sample here. Called on every path that stops a
  # container (stop verb, rm, rec, and stop_other_vllm_containers).
  #
  # Safe to call repeatedly: it accumulates deltas, so an extra sample splits a
  # window in two rather than double-counting it. Never fatal — a missing
  # script, missing jq, or an unreachable server must not block a stop.
  local script="$HOME/scripts/vllm-token-usage.sh"
  [ -x "$script" ] || return 0
  "$script" || true
}

function stop_other_vllm_containers() {
  # Single GPU and a single port 8000: only one vLLM server can run at a time, so
  # every OTHER running vllm_* container is stopped before this one starts. Called
  # on every start path (fresh `docker run`, stopped-container restart, and `rec`),
  # so switching configs never needs a manual `vllm stop` first. Arg: the container
  # about to be started — it is left alone. Containers are created with
  # `--restart unless-stopped`, so an explicit `docker stop` keeps them down.
  local keep="$1" others name pids
  others=$(docker ps --filter "name=vllm_" --format "{{.Names}}" | grep -vx -- "$keep")
  [ -n "$others" ] || return 0

  # Grab the host PIDs BEFORE stopping: once the container is gone `docker top`
  # has nothing left to report, and wait_for_gpu_release needs them to tell our
  # draining processes apart from unrelated CUDA apps.
  pids=""
  while read -r name; do
    [ -n "$name" ] || continue
    echo "Stopping other vLLM container: $name"
    pids+=" $(docker top "$name" -eo pid 2>/dev/null | tail -n +2 | tr '\n' ' ')"
  done <<< "$others"

  sample_token_usage
  echo "$others" | xargs -r docker stop
  wait_for_gpu_release "$pids"
  # The GPU and the port drain independently — a container can be off the card
  # while the daemon still holds its 8000 binding. Both waits are needed.
  wait_for_port_release 8000
}

function one_running_vllm() {
  # Echo the single running vLLM container name; error (to stderr) + return 1 if
  # none or more than one are running.
  local names
  names=$(docker ps --filter "name=vllm_" --format "{{.Names}}")
  if [ -z "$names" ]; then
    echo "Error: No running vLLM container." >&2
    return 1
  fi
  if [ "$(echo "$names" | wc -l)" -gt 1 ]; then
    echo "Error: Multiple vLLM containers running. Stop all but one first ('vllm stop')." >&2
    return 1
  fi
  echo "$names"
}

function rec_config_for_container() {
  # Reverse of the DOCKER_NAME assignments in the vllm() case statement: map a
  # running container name back to the menu token that produced it, so bare
  # `vllm rec` can recreate the exact configuration. The model id alone is
  # ambiguous (configs 1, 1c and 1k share unsloth/Qwen3.8-27B-NVFP4, and 4m/4f
  # share RedHatAI/gemma-4-31B-it-NVFP4), so we key off the unique container name.
  # Keep this in sync with the DOCKER_NAME values in the case statement below.
  case "$1" in
    vllm_qwen_3_8_qat)           echo "1" ;;
    vllm_qwen_3_8_228k)          echo "1c" ;;
    vllm_qwen_3_8_bf16kv)        echo "1k" ;;
    vllm_k2_horizon_mova_36b_int4) echo "2" ;;
    vllm_diffusiongemma_26b_fp8) echo "3" ;;
    vllm_diffusiongemma_26b_nvfp4) echo "3n" ;;
    vllm_gemma4-31b-nvfp4-mtp-ram) echo "4" ;;
    vllm_gemma4-31b-nvfp4-mtp)   echo "4m" ;;
    vllm_gemma4-31b-nvfp4-mtp-hc) echo "4mc" ;;
    vllm_gemma4-31b-nvfp4-mtp-fp4kv) echo "4f" ;;
    vllm_nemotron35_lightning_30b_a3b) echo "5" ;;
    *) return 1 ;;
  esac
}

function get_vllm_model_id() {
  # Print the loaded model's ID from /v1/models (used by aid() in ~/.bash_aliases).
  local host="http://localhost:8000/v1" model_id
  if ! curl -s --head "$host/models" > /dev/null; then
    echo "Error: Cannot connect to vLLM at $host (is it running?)" >&2
    exit 1
  fi
  model_id=$(curl -s "$host/models" | python3 -c "import sys, json; print(json.load(sys.stdin)['data'][0]['id'])")
  if [ -z "$model_id" ]; then
    echo "Error: Could not extract model ID." >&2
    exit 1
  fi
  echo "$model_id"
}

function vllm() {
  local RM_MODE=
  local REC_MODE=
  local REC_INFERRED=
  local REC_YES=
  local REC_REPLY=
  # Per-config extra `docker run` flags (e.g. --env VLLM_ATTENTION_BACKEND=...,
  # --ulimit memlock=-1:-1) — spliced raw into `docker run`, not just env vars.
  # Default empty; a case branch below may set it. Baked into the container at
  # create time, so the stopped-container restart path inherits it automatically.
  local ENV_ARGS=()

  # Color codes
  local CYAN='\033[96m'
  local GREEN='\033[92m'
  local YELLOW='\033[93m'
  local MAGENTA='\033[95m'
  local RED='\033[91m'
  local GRAY='\033[90m'
  local BOLD='\033[1m'
  local RESET='\033[0m'

  # Check if argument was provided
  if [ -n "$1" ]; then
    # Only the arg-shaped forms (rm/rec consume extra args) need pre-processing.
    # Control verbs (logs|l, test|t, stop|s|kill|k, status|st|ps) fall through to
    # `*)` and are matched verbatim by their own case branches below — which is
    # what lets the interactive prompt accept them too, since that path assigns
    # model_choice straight from `read` without passing through here.
    case "$1" in
      rm)
        RM_MODE=1
        model_choice="$2"
        ;;
      rec)
        REC_MODE=1
        # Consume 'rec'; remaining args are an optional config token + -y/--yes.
        # This shift mutates $@, but the only later read of $1 (the DETACHED
        # branch) is guarded by `-z "$REC_MODE"`, so rec mode never reaches it.
        shift
        for arg in "$@"; do
          case "$arg" in
            -y|--yes) REC_YES=1 ;;
            *) model_choice="$arg" ;;
          esac
        done
        ;;
      *)
        model_choice="$1"
        ;;
    esac
  else
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${CYAN}║${RESET}${BOLD}                        vLLM Model Selection${RESET}                                ${CYAN}║${RESET}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════════════╝${RESET}"
    echo ""
    echo -e "${BOLD}1)${RESET} ${CYAN}Qwen3.8-27B-QUASAR-NVFP4 (QAT)${RESET}"
    echo -e "   ${GRAY}110k context | bf16 KV cache | native MTP spec-decode${RESET}"
    echo -e "   ${GRAY}Qwen 3.8 27B hybrid-attention, QAT NVFP4 + bf16 lm_head${RESET}"
    echo -e "   ${GRAY}('1c'/'1k' = older unsloth PTQ quant: 228k ctx / bf16 KV @ 76k)${RESET}"
    echo ""
    echo -e "${BOLD}2)${RESET} ${CYAN}K2-Horizon-MoVA-36B-A4B (cyankiwi AWQ INT4)${RESET}"
    echo -e "   ${GRAY}44k context | fp8 KV | ~130 tok/s | no spec-decode${RESET}"
    echo -e "   ${GRAY}IFM MoE + mixture-of-value-experts attention, agentic/reasoning${RESET}"
    echo ""
    echo -e "${BOLD}3)${RESET} ${CYAN}DiffusionGemma-26B-A4B-FP8 (block diffusion)${RESET}"
    echo -e "   ${GRAY}20k context | ~385 tok/s | denoises a 256-token canvas per step${RESET}"
    echo -e "   ${GRAY}Gemma 4 MoE dLLM (3.8B active), multimodal, thinking off by default${RESET}"
    echo -e "   ${GRAY}('3n' = nvidia NVFP4 quant, 131k ctx; no temperature/seed/JSON-mode)${RESET}"
    echo ""
    echo -e "${BOLD}4)${RESET} ${CYAN}Gemma-4-31B-IT-NVFP4 (RedHatAI) + MTP + RAM KV tier${RESET}"
    echo -e "   ${GRAY}90k context (1.12x concurrency) | fp8 KV | 24GB host-RAM prefix cache${RESET}"
    echo -e "   ${GRAY}Gemma 4 IT, compressed-tensors NVFP4, multimodal, Google MTP drafter${RESET}"
    echo -e "   ${GRAY}(use '4m' for 105k ctx without the RAM tier, '4mc' for text-only Turbo high-context,${RESET}"
    echo -e "   ${GRAY} '4f' for MTP + NVFP4 KV cache)${RESET}"
    echo ""
    echo -e "${BOLD}5)${RESET} ${CYAN}Nemotron-3.5-Lightning-30B-A3B-NVFP4${RESET}"
    echo -e "   ${GRAY}256k context | fp8 KV cache | native MTP spec-decode${RESET}"
    echo -e "   ${GRAY}NVIDIA Mamba-2/MoE hybrid, 3B active of 30B (text-only)${RESET}"
    echo ""
    echo -e "${BOLD}l)${RESET} ${YELLOW}Show live logs of running container${RESET}"
    echo ""
    echo -e "${BOLD}t)${RESET} ${GREEN}Test inference with running model${RESET}"
    echo ""
    echo -e "${BOLD}s)${RESET} ${RED}Stop all running vLLM containers${RESET}"
    echo ""
    echo -ne "${BOLD}Select model (1-5) or action (l/t/s):${RESET} "
    read model_choice
  fi

  # Bare `vllm rec` (no config token): infer which configuration is running from
  # the container name and recreate that one.
  if [ -n "$REC_MODE" ] && [ -z "$model_choice" ]; then
    local running
    running=$(docker ps --filter "name=vllm_" --format "{{.Names}}")
    if [ -z "$running" ]; then
      echo -e "${RED}Error: No running vLLM container to recreate.${RESET}"
      return 1
    fi
    if [ "$(echo "$running" | wc -l)" -gt 1 ]; then
      echo -e "${RED}Error: Multiple vLLM containers running; can't infer which to recreate. Specify a config, e.g. 'vllm rec 1'.${RESET}"
      return 1
    fi
    model_choice=$(rec_config_for_container "$running")
    if [ -z "$model_choice" ]; then
      echo -e "${RED}Error: Running container '$running' doesn't match a known configuration.${RESET}"
      return 1
    fi
    REC_INFERRED=1
    echo -e "${GRAY}Detected running container ${BOLD}$running${RESET}${GRAY} -> config '$model_choice'.${RESET}"
  fi

  case $model_choice in
    1)
      # QAT NVFP4 checkpoint (QUASAR-QAT), bf16 KV. REPLACED the unsloth PTQ
      # checkpoint + fp8 KV on 2026-09-07 after an audit of why this config felt
      # markedly worse than gemma4 4m. Rollback: options 1c/1k still run unsloth.
      #
      # WHY THE SWAP — three measured asymmetries vs 4m, none of them "too few bits":
      #  1. lm_head. unsloth quantizes lm_head to FP8; RedHatAI's gemma-4 (4m,
      #     the config that feels good) keeps lm_head AND embeddings at BF16.
      #     QUASAR also keeps lm_head/embeds/MTP/vision out of the quant (its
      #     `ignore` list), so the output projection no longer perturbs every
      #     logit. This is the one lever where 4m and old-1 differed outright.
      #  2. KV exposure. This model is a HYBRID: layer_types is 48
      #     linear_attention (DeltaNet, fixed-size recurrent state, NOT KV-cached
      #     and NOT affected by --kv-cache-dtype) + only 16 full_attention layers
      #     -- and those 16 are the model's ONLY exact long-range recall path.
      #     With GQA at 4 KV heads x head_dim 256 that is 32,768 B/token of
      #     long-range KV, vs 81,920 B/token for gemma-4 (10 full layers x 16 KV
      #     heads), whose other 50 layers are sliding-window 1024 so their quant
      #     error cannot accumulate along the sequence. Same --kv-cache-dtype fp8
      #     flag, very different amount of redundancy absorbing the error. The old
      #     unsloth checkpoint also baked a STATIC per-tensor fp8 KV scheme
      #     (kv_cache_scheme: dynamic=false, strategy=tensor, static_minmax).
      #     QUASAR ships kv_cache_scheme: null, so there is no baked scheme to
      #     silently override the CLI (the `auto` trap documented in 1k does not
      #     apply here) -- but keep passing the dtype explicitly anyway.
      #  3. Size. QUASAR is 20.56 GB on disk vs unsloth's 23.42 GB (it is QAT, so
      #     all 496 linears are NVFP4 W4A4 rather than unsloth's 168-of-496 with
      #     attention+DeltaNet held at FP8). That ~2.7 GiB is what PAYS for the
      #     bf16 KV below without collapsing context to 1k's 74k.
      #
      # WHAT QAT BUYS is a reasoned bet, NOT a measured one. QUASAR's own card
      # reports GPQA-D 0.9091 vs unsloth 0.8939 vs BF16 0.9141, but at n=396 x 2
      # runs that gap is ~6 questions, z=1.02, 95% CI [-1.4%, +4.5%] -- NOT
      # significant, and neither is BF16-vs-unsloth. Do not cite that table as
      # proof. The defensible claims are (1) and (3) above, which are structural.
      #
      # RISK: QUASAR quantizes attention and DeltaNet to W4A4, which the old
      # checkpoint did not (it was W4A4 on MLPs only). That is new NVFP4
      # activation-quant kernel surface on sm_120, i.e. exactly the class of
      # thing that produced the b19 XQA `q_cu_seq_lens` breakage -- upstream CI
      # does not cover this box. It can fail at FIRST MODEL START, not at build.
      # If engine init dies in a flashinfer/cutlass kernel, first try
      # --kernel-config '{"enable_flashinfer_autotune":false}' (the 4m lever);
      # if it still fails, roll back to 1c/1k rather than chasing it.
      #
      # CONTEXT_SIZE IS AN ESTIMATE -- RE-DERIVE ON FIRST BOOT. Old config 1
      # measured a 151,351-token pool at fp8 (~49.4 KB/token) = ~6.96 GiB of KV.
      # bf16 KV costs ~86.1 KB/token (measured in 1k: 1.75x, not 2x, because the
      # DeltaNet/Mamba state is sized separately and ~6.25% is lost to padding
      # layers). Freeing ~2.7 GiB of weights gives ~9.6 GiB of KV -> ~120k tokens
      # bf16. 110000 is ~0.92x of that estimate: a THIN margin, set deliberately
      # (2026-09-07) in preference to the safer 98304 first tried. The estimate
      # itself is uncertain -- it assumes QUASAR's resident footprint scales like
      # unsloth's on-disk delta, and the old checkpoint's resident size ran
      # ~1.1 GiB ABOVE its on-disk size (scale swizzling into the FlashInfer
      # NVFP4 layout). So treat a startup failure as expected-if-unlucky, not as
      # a broken config. The tell is an engine-init abort reading "KV cache is
      # needed, which is larger than the available KV cache memory" (or an
      # "estimated maximum model length is <N>" line, which hands you the answer
      # directly); drop to 98304 and it will boot. After it DOES boot, read the
      # real number and re-tune toward ~1.0x concurrency:
      #   docker logs vllm_qwen_3_8_qat 2>&1 | grep -i "GPU KV cache size"
      #   curl -s localhost:8000/metrics | grep cache_config_info
      # NOTE CONTEXT_SIZE is part of the torch.compile cache key, so each change
      # costs a full recompile -- including the retry after a failed boot.
      #
      # --num-gpu-blocks-override IS DELIBERATELY ABSENT. The old value (112) was
      # derived for unsloth's weights at fp8 KV and is actively wrong here: both
      # the weight footprint and the per-token KV cost changed. Let the profiler
      # size the pool; only pin it again after measuring, and re-derive the
      # ~19-block holdback if you do.
      #
      # --chat-template: QUASAR ships Qwen's CANONICAL template, which is OLDER
      # than the one unsloth patched. Two regressions matter: it has no
      # multi/`developer` system-message merging, and it RAISES on
      # reasoning_effort='high' (it accepts only xhigh/medium/low), which OpenAI
      # clients send routinely -> a 400 on every such request. So we mount
      # unsloth's patched template explicitly. Both default reasoning_effort to
      # 'xhigh' and emit the same <tool_call><function=...> format, so
      # --tool-call-parser qwen3_coder is unchanged and correct (qwen3_coder and
      # qwen3_xml both resolve to the same Qwen3EngineToolParser class).
      #
      # Sampling is NOT overridden on purpose: generation_config.json ships
      # temperature 1.0 / top_p 0.95 / top_k 20, which is exactly Qwen's
      # documented THINKING-mode recipe, and neither pi nor OpenClaw overrides
      # it. Do not "fix" this to 0.6/0.7 -- 0.7/top_p 0.8/presence_penalty 1.5 is
      # the NON-thinking recipe and would be wrong here.
      #
      # DOCKER_NAME changed (was vllm_qwen_3_8) ON PURPOSE: a stopped container
      # is RESTARTED, not recreated, so reusing the old name would silently
      # relaunch the old checkpoint with the old flags.
      MODEL_ID="QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4"
      QUANTIZATION="compressed-tensors"
      CONTEXT_SIZE=110000
      INPUT_MODES="text,image"
      DOCKER_NAME="vllm_qwen_3_8_qat"
      DOCKER_IMAGE="vllm-custom:latest"
      CPU_OFFLOAD=0
      GPU_MEM_UTIL=0.982
      EXTRA=(--enable-prefix-caching --kv-cache-dtype bfloat16 --max-num-seqs 2 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --chat-template /etc/vllm/chat-templates/qwen3_8-patched.jinja --speculative-config '{"method":"mtp","num_speculative_tokens":4}')
      ;;
    1c)
      # Same as option 1 but with maximized context (228K of the 256K native).
      # Native MTP + 0.982 util ceiling as in option 1 (see notes there).
      MODEL_ID="unsloth/Qwen3.8-27B-NVFP4"
      QUANTIZATION="compressed-tensors"
      CONTEXT_SIZE=228000
      INPUT_MODES="text,image"
      DOCKER_NAME="vllm_qwen_3_8_228k"
      DOCKER_IMAGE="vllm-custom:latest"
      CPU_OFFLOAD=0
      GPU_MEM_UTIL=0.982
      EXTRA=(--enable-prefix-caching --kv-cache-dtype fp8 --max-num-seqs 2 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --speculative-config '{"method":"mtp","num_speculative_tokens":4}')
      ;;
    1k)
      # Option 1 with an UNQUANTIZED (bf16) KV cache — a quality experiment, not a
      # capacity one. Weight quantization error is fixed and calibrated; KV
      # quantization error instead ACCUMULATES along the sequence and lands
      # directly on attention scores, so at long context it's plausibly the larger
      # quality lever on this checkpoint (whose bulk is already only 4-bit in the
      # MLPs — attention, lm_head and layers 56-63 are FP8, vision/MTP bf16).
      # `--kv-cache-dtype bfloat16` MUST be passed explicitly, and `auto` is a TRAP
      # here. This checkpoint bakes its own KV quantization into config.json
      # (`kv_cache_scheme`: num_bits 8, type float, strategy tensor, static
      # calibrated scales); CompressedTensorsKVCacheMethod applies that whatever
      # the CLI says, and logs NO warning. Verified on the 2026-08-15 boot of this
      # very config with `auto`: FlashInfer reported kv_cache_dtype=
      # torch.float8_e4m3fn, i.e. it was silently identical to option 1. Grep the
      # boot log for "FlashInfer resolved query dtypes" to confirm what you got —
      # the engine-config line echoes the CLI value and will lie to you.
      #
      # VERIFIED 2026-08-15: metrics report cache_dtype="bfloat16", so the explicit
      # flag DOES beat the checkpoint scheme. Note the "FlashInfer resolved query
      # dtypes" line disappears on this path — it only logs for the fp8 KV backend,
      # so check `curl -s localhost:8000/metrics | grep cache_config_info` instead.
      #
      # Sizing: the KV pool is ~5.98 GiB REGARDLESS of CONTEXT_SIZE — it's whatever
      # survives weights (22.9 GiB) + peak activation (1.91) + cudagraphs (0.04).
      # Lowering ctx frees NO memory; it only changes the concurrency ratio.
      # Measured: 74,556 tokens @ bf16 (~86 B/token) vs 130,121 @ fp8 (~49 B/token)
      # — a 1.75x ratio, NOT the 2.0x the dtype width suggests, because on this
      # hybrid the linear_attn/Mamba state is sized separately (mamba_block_size
      # 16, mamba_ssm_cache_dtype float32, both untouched by --kv-cache-dtype) and
      # ~6.25% is lost to "Add 3 padding layers". Don't predict this ratio; read it.
      # 64000 measured 1.16x, so CONTEXT_SIZE was raised to 74000: FINAL MEASURED
      # STATE is 76,715 pool tokens @ 1.04x, the ~1.0x single-user target the other
      # configs use. The pool moved (74,556 -> 76,715) even though block_size (816)
      # and num_gpu_blocks (113) did NOT — so re-read cache_config_info and the
      # concurrency line after any ctx change instead of assuming either holds.
      #
      # Native MTP is retained and costs no quality — rejection sampling makes
      # speculative decoding distribution-preserving. Dropping fp8 KV also moves
      # AWAY from the FlashInfer fp8-prefill + MTP crash noted on option 2, so this
      # is the safer path there, not the riskier one. GPU_MEM_UTIL stays at the
      # 0.982 ceiling for the same reason as option 1 (the flashinfer fp4 autotune
      # warmup transient OOM-crash-loops at 0.983) — that limit is about the weight
      # path and is unaffected by the KV dtype. CONTEXT_SIZE is part of the
      # torch.compile cache key, so the first boot pays a full recompile.
      MODEL_ID="unsloth/Qwen3.8-27B-NVFP4"
      QUANTIZATION="compressed-tensors"
      CONTEXT_SIZE=74000
      INPUT_MODES="text,image"
      DOCKER_NAME="vllm_qwen_3_8_bf16kv"
      DOCKER_IMAGE="vllm-custom:latest"
      CPU_OFFLOAD=0
      GPU_MEM_UTIL=0.982
      EXTRA=(--enable-prefix-caching --kv-cache-dtype bfloat16 --max-num-seqs 2 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --speculative-config '{"method":"mtp","num_speculative_tokens":4}')
      ;;
    2)
      # IFM K2-Horizon-MoVA-36B-A4B, cyankiwi AWQ INT4 (compressed-tensors
      # pack-quantized, W4A16, group 32, asymmetric). Replaced Qwen3.6-35B-A3B-
      # NVFP4 on 2026-09-28 (recover from git history if needed).
      #
      # NEEDS THE IMAGE PATCHES (b24+). MoVA = Mixture of Value experts: V comes from 64
      # routed experts (top-4) per layer instead of a v_proj. Upstream only
      # handles bf16 value experts; any INT4 checkpoint dies in profile_run with
      # "'ColumnParallelLinear' object has no attribute 'weight'". The Dockerfile
      # applies cyankiwi fork commit 46485cf572 (k2_horizon_mova_int4.patch),
      # which keeps the experts packed 4-bit on the existing wna16 fused-MoE
      # kernel, plus our k2_horizon_mova_int4_compile.patch (the fork alone dies
      # under torch.compile; see the Dockerfile). UNMERGED fork code, not
      # upstream-CI'd. Its author checked it against a bf16 reference (4.9e-4)
      # on a different group-32 checkpoint. Here it has only been spot-checked on
      # sm_120 (correct arithmetic, tool call, 42k needle), with no eval run.
      #
      # Quant contents: embeddings, lm_head, shared experts, both routers and
      # layers 0-2 stay bf16 (~5.3 bits/param overall). Attention q/k/o ARE int4.
      # Calibrated on STEM + agentic data.
      #
      # ---- SIZING (MEASURED b24, 2026-09-28, cold boot) ----
      # Weights 22.7 GiB resident (checkpoint 22.6 GiB on disk; the fork's
      # "26.58 GiB" figure was weights + non-torch and did NOT carry over here).
      # All 48 layers are full attention, 8 KV heads x 128: fp8 KV = 96 KiB/token
      # (vs ~3 KiB for config 5). No sliding window and no hybrid state, so the
      # pool does NOT move with max_model_len (the same 4.09 GiB = 44,720 tokens
      # at 24576 and at 44000 ctx). CONTEXT_SIZE 44000 = 1.02x. Idle 30.7 GiB used.
      # Stress: a 33,041-token and then a 42,721-token needle prompt were answered
      # correctly in 5.0s / 7.0s, with 0 restarts and 31.0 of 32.6 GiB used at
      # peak. Decode ~131 tok/s (single stream, no spec-decode).
      # Raising context means raising util (capped by laya, below) or
      # --num-gpu-blocks-override. Lowering CONTEXT_SIZE frees nothing.
      # GPU_MEM_UTIL 0.96: the laya container's ~500 MiB resident caps this
      # at ~0.967 (see config 3). No --num-gpu-blocks-override: the profiled
      # pool is what was stress-tested above.
      # expandable_segments: reclaims fragmentation slack (as in config 3). No
      # KV connector here, so nothing forbids it (unlike config 4).
      #
      # ---- FLAGS ----
      # --reasoning-parser/--tool-call-parser k2_horizon: registered upstream.
      #   The template's tool format (json/xml/xml_typed, default xml) is chosen
      #   per request via chat_template_kwargs; the parser handles all three.
      # --override-generation-config: generation_config.json ships NO sampling
      #   defaults, but the card's results all use temperature 1.0 / top_p 0.95.
      #   Without this, vLLM would fall back to top_p 1.0.
      # ---- CLIENT REQUIREMENTS (hit by pi on 2026-09-28) ----
      # (1) reasoning_effort: only high/medium/low. Anything else ('minimal',
      #   'xhigh', ...) is a 400 from the reasoning parser. Only use 'high'
      #   (the template default and the card's setting). At medium/low the model
      #   sometimes closes with the high-level </ifm|think> instead of its own
      #   </ifm|think_fast[er]>. The parser then files the whole answer as
      #   reasoning and the client shows an empty reply, with no error.
      # (2) FIXED SERVER-SIDE by --chat-template k2_horizon-patched.jinja
      #   (~/scripts/chat-templates). The stock template raises "Assistant
      #   message is missing a thinking field" whenever a replayed assistant
      #   turn has no string reasoning field, which the OpenAI API does not
      #   require. That was a 400 on turn 2 of every pi and OpenClaw tool loop.
      #   The patch renders an empty think block instead; it is the only change
      #   vs the checkpoint's chat_template.jinja. Re-diff if the checkpoint's
      #   template is updated.
      # (1) has NO server-side fix (the parser picks its end tag from the
      #   request), so each client maps its thinking levels to "high":
      #   thinkingLevelMap on this model's entry in ~/.pi/agent/models.json and
      #   ~/.openclaw/openclaw.json. update-agent-models.sh preserves those fields.
      # --max-num-batched-tokens 4096: caps the prefill activation peak, because
      #   headroom is the scarce resource here.
      # No MTP/drafter: the checkpoint ships none.
      # --trust-remote-code: REQUIRED. Neither transformers 5.12.1 nor vLLM's
      #   config registry knows model_type k2_horizon, so ModelConfig refuses to
      #   start without it. It only executes the repo's configuration_k2_horizon.py
      #   (a plain PreTrainedConfig dataclass, no I/O; audited 2026-09-28). The
      #   model itself still runs on vLLM's native K2HorizonForCausalLM.
      # --chat-template-content-format string: REQUIRED. vLLM auto-detects
      #   'openai' (list-of-parts) for this template, but the template renders
      #   message.content ONLY if it is a string and silently substitutes ''
      #   otherwise. Every user message came through empty (prompt_tokens 10,
      #   the model answering "??"), with no error. 'string' makes vLLM flatten
      #   the parts first. Verify with /tokenize + messages after any change.
      MODEL_ID="cyankiwi/K2-Horizon-MoVA-36B-A4B-AWQ-INT4"
      QUANTIZATION="compressed-tensors"
      CONTEXT_SIZE=44000
      INPUT_MODES="text"
      DOCKER_NAME="vllm_k2_horizon_mova_36b_int4"
      DOCKER_IMAGE="vllm-custom:latest"
      CPU_OFFLOAD=0
      GPU_MEM_UTIL=0.96
      ENV_ARGS=(--env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
      EXTRA=(--trust-remote-code --chat-template /etc/vllm/chat-templates/k2_horizon-patched.jinja --chat-template-content-format string --enable-prefix-caching --kv-cache-dtype fp8 --max-num-seqs 2 --max-num-batched-tokens 4096 --enable-auto-tool-choice --tool-call-parser k2_horizon --reasoning-parser k2_horizon --override-generation-config '{"temperature":1.0,"top_p":0.95}')
      ;;
    3|3n)
      # DiffusionGemma 26B-A4B — Google's BLOCK-DIFFUSION Gemma 4 MoE (25.2B total,
      # 3.8B active, 8 of 128 experts + 1 shared). Replaced the DeepSeek-R1-Distill-32B-AWQ
      # config on 2026-09-22 (recover it from git history if ever needed).
      #
      # NOT autoregressive. An encoder pass caches the prompt causally, then the
      # decoder denoises a whole 256-token CANVAS with bidirectional attention,
      # iterating up to max_denoising_steps (48) until the canvas is stable, then
      # commits it and starts the next one. So "tokens/s" comes from committing
      # ~15-20 tokens per forward pass, not from one-token-per-step + spec decode.
      # Both variants share every flag; only the checkpoint/quant/ctx/util differ:
      #   3   RedHatAI FP8-dynamic (compressed-tensors) — the variant upstream has an
      #       end-to-end eval recipe for (tests/evals/gsm8k/configs/DiffusionGemma-*.yaml)
      #   3n  nvidia NVFP4 (modelopt) — ~8 GiB more headroom, see sizing below
      #
      # ---- SIZING (weights dominate, but KV and warmup both matter — see STATUS) ----
      # FP8:   24.48B fp8 + 1.34B bf16 params ~= 25.3 GiB of weights.
      # NVFP4: modelopt quantizes ONLY the MoE experts — its hf_quant_config.json
      #        ignores lm_head, *mlp*, *router*, *self_attn*, *self_conditioning*,
      #        *vision_tower*, *embed_vision* — so 11.4 GB packed fp4 + 1.4 GB scales
      #        + 3.0B bf16 params ~= 17.5 GiB. Do NOT expect a 4x shrink from "NVFP4".
      # KV is NOT cheap, despite what the layer_types list suggests. 25 of the 30
      # layers are sliding (window 1024) and only 5 are full attention with 2 KV
      # heads x 512, so the arithmetic says "bounded per sequence" — but the
      # sliding layers draw from the SAME pool, and the measured cost is 68-95
      # KiB/token (it moves with max_model_len; see the STATUS block). Predicting
      # it from the config is how this config got sized wrong twice on 2026-09-22.
      # CONTEXT_SIZE below is a STARTING
      # POINT — boot cold, read "GPU KV cache size: <N> tokens", and tune from the
      # measured pool (same discipline as configs 4/4m). Record what you measure here.
      #
      # ---- FLAGS, and why ----
      # --attention-backend TRITON_ATTN: this model passes a per-request causal FLAG
      #   TENSOR (mixed causal encoder + bidirectional decoder in one batch). Only
      #   TRITON_ATTN and FA4 handle that; FlashInfer is hard-rejected by vLLM
      #   (models/config.py), and FA defaults to FA2 on sm_120 (fa_utils.py: FA4 is
      #   auto-selected only for capability major==10), which raises NotImplementedError
      #   on the tensor-causal path. vLLM's Gemma4 hook already forces TRITON_ATTN here
      #   ("FA4 not available"), so this is belt-and-braces — but deliberately explicit:
      #   if a future image ever ships FA4 for sm_120, we would be silently promoted
      #   onto a path upstream only validates on SM90 (tests/kernels/attention/
      #   test_mixed_causal_attn.py), which is exactly how the flashinfer q_cu_seq_lens
      #   skew bit us at runtime in b19.
      # --max-num-seqs 1: the diffusion sampler materializes [seqs, canvas, vocab]
      #   fp32 transients — 1 x 256 x 262144 x 4B = 268 MiB PER SEQUENCE per buffer,
      #   and there are several. That is the warmup allocation that OOMed this config
      #   three times on 2026-09-22 (the failing request was for exactly 256 MiB).
      #   vLLM also CLOBBERS this value to 8 unless it is BELOW its generic default,
      #   so it must stay small and explicit. 1 is not a throughput compromise worth
      #   arguing about on a single-user box; `3n` (17.5 GiB of weights) could afford
      #   2, but shares this line — raise it there only with a cold-boot re-test.
      #   If more room is ever needed, the next lever is the canvas itself:
      #   -dc '{"canvas_length":128}' halves every one of those buffers, at the cost
      #   of committing fewer tokens per step (and deviating from how it was trained).
      # --enforce-eager: what upstream's only end-to-end recipe uses. FULL-cudagraph
      #   support is coded (diffusion_gemma.py prepare_attn) but unexercised, and the
      #   scheduler refuses to pad decode batches for diffusion. TUNING STEP 2: drop it
      #   and measure; keep it if capture OOMs or regresses.
      # --max-num-batched-tokens 2048: a decode step budgets 1+canvas_length (=257)
      #   tokens per running sequence, so this only has to clear 257 — the rest is
      #   prefill chunk size, and a smaller chunk is less activation memory on a
      #   config with none to spare.
      # No --chat-template: the checkpoint's bundled template is used as-is and has
      #   enable_thinking=false by default (fast path). Callers opt in per request with
      #   chat_template_kwargs {"enable_thinking": true}. Do NOT point this at
      #   gemma4-force-think.jinja — that is Gemma 4's template, not this model's (this
      #   one carries its own tool-loop/turn-closure fixes).
      # No --kv-cache-dtype: default bf16 for now. fp8 is a later lever (TRITON_ATTN
      #   supports it, and the NVFP4 checkpoint even ships fp8 KV scales).
      # No --speculative-config, EVER: canvas_length IS num_speculative_tokens for this
      #   model (config/vllm.py), and a spec config silently overrides it.
      # The gemma4 parsers apply because the chat template is Gemma 4-shaped
      #   (<|channel>thought...<channel|>, <|tool_call>call:...<tool_call|>).
      #
      # ---- REQUEST-SHAPE CONSTRAINTS (the real sharp edge of this config) ----
      # vLLM 400s any request with temperature != 1.0, seed, min_p, min_tokens,
      # logit_bias, bad_words or allowed_token_ids (sampling_params.py
      # _validate_diffusion) — the denoiser has its own fixed temperature schedule
      # (0.8 -> 0.4). Structured outputs / response_format are 400'd too, which also
      # rules out tool_choice "required" or a named function; --enable-auto-tool-choice
      # is fine because that parser is pure text. Penalties are accepted but ignored
      # with a warning. `vllm test` and bench-inference.ts were fixed to stop sending
      # an explicit temperature; any OTHER client that sends one will fail here.
      # ---- STATUS: `3` VALIDATED on b22, 2026-09-22 ----
      # util 0.92 / --max-num-seqs 1 / 20480 ctx: KV 2.37 GiB = 35,477 tokens
      # (1.73x), idle 31,663 of 32,607 MiB, 0 restarts. Stress: 18,925-token
      # prompt 1.5s, evicted by a second 18,225-token prompt, then re-prefilled
      # 0.6s — no OOM. Bench (bench-inference.ts): 385 tok/s mean / 627 peak
      # end-to-end, 8.0 denoising steps per canvas, 32 tokens committed per step.
      # The pool is 1.73x the context, so CONTEXT_SIZE could go higher — but the
      # pool is NOT stable across max_model_len on this hybrid (measured:
      # 2.85 GiB -> 39,122 tokens at 32k ctx, but 2.36 GiB -> 25,955 at 24k, i.e.
      # 76 vs 95 KiB/token), so re-measure rather than extrapolating. 20480 is
      # the value that was actually stress-tested.
      #
      # ---- MEASURED, b22, 2026-09-22 (cold boot, FP8 variant) ----
      # weights 25.86 GiB | profiled activation peak ~1.13 GiB | KV 76.4 KiB/token
      # (2.85 GiB -> 39,122 tokens). KV is NOT cheap on this model: the 25 sliding
      # layers draw from the same pool as the 5 full-attention ones, so the window
      # does not bound it the way the layer_types list suggests.
      # At util 0.95 + 32768 ctx it BOOTED THE KV POOL AND THEN DIED: the profiler
      # handed 2.85 GiB to KV, leaving 52 MiB free, and torch inductor's autotune
      # (benchmark_all_configs, which --enforce-eager does NOT skip) OOMed asking
      # for 96 MiB. --restart unless-stopped turned that into a crash loop.
      # Fix was headroom, not a smaller context: util 0.95 -> 0.93 hands ~630 MiB
      # back, and expandable_segments reclaims the ~210 MiB the OOM report showed
      # as "reserved but unallocated". Context came down with it because the pool
      # must stay >= max_model_len or the engine refuses to start.
      # NOTE the pool is whatever survives weights + peak, so LOWERING CONTEXT
      # ALONE FREES NOTHING — util (or --num-gpu-blocks-override) is the lever.
      MODEL_ID="RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic"
      QUANTIZATION="compressed-tensors"
      CONTEXT_SIZE=20480
      INPUT_MODES="text,image"
      DOCKER_NAME="vllm_diffusiongemma_26b_fp8"
      DOCKER_IMAGE="vllm-custom:latest"
      CPU_OFFLOAD=0
      # No KV connector here (unlike config 4), so expandable_segments is allowed
      # — and this config needs it: see the fragmentation note above.
      ENV_ARGS=(--env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
      # 0.92, NOT the 0.982 the Gemma 4 configs use. Two separate reasons, and
      # both are real — do not raise this without re-running a cold boot.
      # (1) STARTUP CHECK. vLLM refuses to start when
      # FREE memory < util x TOTAL, and this box has a PERMANENT ~500 MiB GPU
      # resident: the `laya` container (uvicorn on :8100). Measured 2026-09-22:
      # free 30.39 of 31.41 GiB with every vLLM container stopped, so anything
      # above ~0.967 dies instantly with "Free memory on device cuda:0 ... is
      # less than desired GPU memory utilization" — and --restart unless-stopped
      # turns that into a crash loop, not a visible failure. (The 4/4m configs
      # boot at 0.982 only because laya's allocation is lazy — it was not
      # resident when they last started.)
      # (2) HEADROOM FOR A TRANSIENT THE BUDGET DOES NOT COVER. The KV pool always
      # expands to fill util x TOTAL, so what protects warmup is the gap between
      # that budget and the memory actually free: 30.39 - util x 31.41. Warmup then
      # has to fit in that gap. Measured with --max-num-seqs 2: the process peaked
      # at 30.64 GiB, i.e. ~1.4 GiB ABOVE a 29.21 GiB budget, so 0.93 (gap 1.18 GiB)
      # still OOMed. 0.92 leaves a ~1.49 GiB gap, and --max-num-seqs 1 roughly
      # halves what has to fit in it. Both were needed; neither alone sufficed.
      GPU_MEM_UTIL=0.92
      # ---- STATUS: `3n` VALIDATED on b22, 2026-09-22 ----
      # Same util/flags, 131072 ctx: weights 17.96 GiB (vs 25.86 FP8), KV 10.14 GiB
      # = 763,555 tokens (5.83x), idle 31,583 MiB, 0 restarts. 45,925-token prompt
      # answered in 3.9s. Bench: 444 tok/s mean / 726 peak, 9.8 denoising steps per
      # canvas, 26.3 tokens per step — faster than `3` AND 6x the context, which is
      # what the extra ~8 GiB buys. Prefer this variant unless you specifically
      # want the FP8 checkpoint upstream evaluates.
      # NOTE the per-token KV cost is ~14 KiB here vs 68-95 KiB on `3`: the sliding
      # layers' per-sequence cost amortizes over a much longer context. It is the
      # same model — so never carry a token/GiB figure between these two configs.
      # The pool is 5.83x the context, so 262144 (the checkpoint's
      # max_position_embeddings) is probably reachable; UNTESTED, and the pool does
      # not scale linearly with max_model_len, so measure before trusting it.
      if [ "$model_choice" = "3n" ]; then
        MODEL_ID="nvidia/diffusiongemma-26B-A4B-it-NVFP4"
        QUANTIZATION="modelopt_fp4"
        CONTEXT_SIZE=131072
        DOCKER_NAME="vllm_diffusiongemma_26b_nvfp4"
      fi
      EXTRA=(--enable-prefix-caching --attention-backend TRITON_ATTN --max-num-seqs 1 --max-num-batched-tokens 2048 --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --enforce-eager)
      ;;
    4)
      # 4m's stack (RedHatAI NVFP4 + Google MTP drafter + fp8 KV + a pinned block
      # pool + autotune skip) at 90k ctx, plus a 24 GiB host-RAM prefix-cache
      # tier. Everything in 4m's comment block applies here too — in particular,
      # re-derive --num-gpu-blocks-override from a COLD boot after ANY image bump.
      # This override is DELIBERATELY LOWER than 4m's (6400 vs 6780) and must NOT
      # be synced to it: this config runs without expandable_segments (see below).
      # (2026-09-10: replaced the old nvidia/Gemma-4-31B-IT-NVFP4 CPU_OFFLOAD=4
      # config, whose weights barely fit the card — recover it from git history.)
      #
      # STATUS: VALIDATED at 6400 blocks (b21, 2026-09-10). Cold boot 83s,
      # 101,088 tokens (1.12x @ 90k), idle 30,784 MiB, 19,660 CPU blocks / 24 GB
      # pinned. Three stress passes (~87k-token prompt + 2 concurrent ~38k
      # prompts + reload) peaked at 31,870 MiB with 0 OOM / 0 restarts. RAM-tier
      # reload of an evicted 87,724-token prompt: 87,680 tokens served from RAM,
      # 0.3s vs 47.6s re-prefill. Correctness: needle test (~60k prompt; answer
      # from computed KV vs from RAM-reloaded KV) matched exactly.
      #
      # History — at 6780 (4m's value) it FAILED: booted fine (idle 31,260 MiB)
      # but the first ~86k-token prompt killed the engine with the SAME 168 MiB
      # prefill-buffer OOM as 4m's 2026-09-09 crash
      # (buf7 = empty_strided_cuda((4096, 21504)), 142.88 MiB free), which
      # --restart unless-stopped then masked. The OOM report showed 508 MiB
      # "reserved by PyTorch but unallocated" — fragmentation expandable_segments
      # would reclaim, and which idle nvidia-smi does NOT reveal. Dropping
      # expandable_segments (forced, see below) costs ~500 MiB of real headroom;
      # 6780 -> 6400 hands back 380 blocks x ~1.22 MiB ~= 476 MiB measured (6780
      # was only ~25 MiB short, but fragmentation without expandable_segments
      # isn't deterministic, so the margin is deliberately wide).
      #
      # Re-validate after ANY change with the long-prompt stress test (cold boot,
      # ~88k prompt, 2 concurrent ~38k prompts, RAM reload, RestartCount) — idle
      # headroom alone proved nothing here. If trimming ever stops being enough:
      # patch the image so the Gemma 4 drafter config keeps
      # enable_cumem_allocator, then restore expandable_segments +
      # --enable-cumem-allocator.
      #
      # CONTEXT_SIZE 90000 (vs 4m's 105000) for 2nd-sequence headroom. MEASURED
      # (b21, 2026-09-10): 6400 blocks -> 101,088 tokens, 1.12x at 90k (4m: 1.10x
      # at 105k). The same 6780 blocks reported 107,090 tokens here vs 115,437 in
      # 4m — for this hybrid sliding-window model the token figure depends on
      # max_model_len, so ratios don't carry over between configs. Read the
      # "GPU KV cache size" boot line.
      # CONTEXT_SIZE (and the kv-transfer config) are in the torch.compile cache
      # key -> the first boot recompiles.
      #
      # RAM TIER (SimpleCPUOffloadConnector, 24 GiB). A second prefix-cache tier,
      # NOT extra context: attention only reads GPU KV, so max context is still
      # bound by the GPU pool. GPU-evicted blocks land in pinned host RAM, and a
      # later request with the same prefix reloads over PCIe instead of
      # re-prefilling. Pays off when rotating between long conversations (one 90k
      # conversation fills most of the pool) and on preemption recovery. Eager
      # mode (default): blocks are copied D2H as they're computed, so they're
      # already in RAM when evicted. GPU cost ~0 — the connector adds two CUDA
      # streams and views over the existing KV tensors; the 24 GiB is a CPU
      # tensor pinned via cudaHostRegister. (The 6780 -> 6400 retune below comes
      # from losing expandable_segments, not from the connector itself.)
      #
      # CAVEAT — a prefix reaches RAM only from its 2nd sighting. On this model a
      # prompt's blocks only become reusable the SECOND time it is seen (sends 1
      # and 2 miss, send 3 hits — identical on 4m with no connector, so this is
      # upstream Gemma 4 hybrid-cache behaviour, not this config). The connector
      # skips unhashed GPU blocks (simple_kv_offload/manager.py
      # `block_hash is None`), so a prefix seen only once never reaches RAM.
      # Multi-turn chat re-sends the history every turn, so earlier turns
      # qualify quickly; a one-off long prompt never will.
      #
      # SIZING: MEASURED 19,660 CPU blocks for 24 GiB — bytes / bytes-per-block,
      # so independent of the GPU override — ~3.1x the 6400-block GPU pool
      # (~310k tokens, ~3.4 full 90k contexts). Host RAM "used" went 5 -> 34 GB
      # at boot. Pinned memory is unswappable and invisible to the page cache;
      # 24 of 91 GB leaves ~55 GB for the OS, the vLLM process, and page cache
      # for the ~20 GB of weight files (fast restarts). 16 GiB ~= 2x the pool,
      # 32 GiB ~= 4x. Figure: "SimpleCPUOffloadScheduler: Allocating N offload
      # blocks" boot line.
      #
      # Two deliberate deviations from 4m, both forced by the connector:
      # * NO PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. vLLM REJECTS any KV
      #   connector while it is set (config/vllm.py _verify_kv_transfer_compat:
      #   VMM remaps would invalidate pinned KV). The documented exemption,
      #   --enable-cumem-allocator, does NOT work with Gemma 4 MTP: the drafter's
      #   VllmConfig is rebuilt from the draft ModelConfig
      #   (v1/worker/gpu/spec_decode/gemma4/speculator.py _create_draft_vllm_config),
      #   which drops enable_cumem_allocator, so the check re-fires and engine init
      #   dies with "KV connector SimpleCPUOffloadConnector is incompatible with
      #   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" (tried 2026-09-10;
      #   still unfixed on upstream main). expandable_segments was 4m's 2026-07-22
      #   lever for the (since-disabled) flashinfer autotune transient; 4m's
      #   headroom was measured WITH it, so this config's headroom is measured
      #   separately (see STATUS). If upstream fixes the drafter config, cumem +
      #   expandable_segments becomes an option again.
      # * --ulimit memlock=-1:-1 (in ENV_ARGS): the container's default memlock
      #   soft limit is 8 MB, and this connector's pin_tensor() RAISES on a
      #   cudaHostRegister failure (no unpinned fallback). Whether the driver
      #   enforces RLIMIT_MEMLOCK there is driver-dependent; this rules it out.
      MODEL_ID="RedHatAI/gemma-4-31B-it-NVFP4"
      QUANTIZATION="compressed-tensors"
      CONTEXT_SIZE=90000
      INPUT_MODES="text,image"
      DOCKER_NAME="vllm_gemma4-31b-nvfp4-mtp-ram"
      DOCKER_IMAGE="vllm-custom:latest"
      CPU_OFFLOAD=0
      GPU_MEM_UTIL=0.982
      ENV_ARGS=(--ulimit memlock=-1:-1)
      EXTRA=(--enable-prefix-caching --kv-cache-dtype fp8 --max-num-batched-tokens 4096 --max-num-seqs 2 --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --chat-template /etc/vllm/chat-templates/gemma4-force-think.jinja --speculative-config '{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":4}' --kernel-config '{"enable_flashinfer_autotune":false}' --num-gpu-blocks-override 6400 --kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":25769803776}}')
      ;;
    4m)
      # RedHatAI Gemma 4 31B NVFP4 + MTP speculative decoding via Google's drafter
      # (option 4 is now this same stack at 90k ctx + a host-RAM KV tier)
      # google/gemma-4-31B-it-assistant (~0.5B, model_type gemma4_assistant which
      # vLLM rewrites to gemma4_mtp; n_predict forced to 1). Drafter shares the
      # target KV cache, so ~1 GB extra weights only.
      #
      # REQUIRES transformers >= 5.12.1 in the image (gemma4_assistant isn't in
      # vLLM's registry; only transformers >=5.12.x recognizes it). Stock
      # vllm-custom:latest ships 5.5.4 -> crashes at config load until rebuilt.
      #
      # CONTEXT_SIZE is KV-bound and drifts with every image bump — re-derive it
      # from the "GPU KV cache size: N tokens" log line after any rebuild.
      # Mind GPU_MEM_UTIL headroom: the flashinfer fp4 autotune warmup needs a
      # ~336 MiB transient that OOM-crash-looped at 0.983 (2026-07-20) and, it
      # turned out, still at 0.982. Nudging GPU_MEM_UTIL is futile here: vLLM
      # sizes the KV pool to fill the whole budget (profiled ~114k tokens), so
      # lowering util just shrinks KV to refill the same sliver, and the
      # flashinfer fp4 autotune warmup (NOT counted in vLLM's mem profiling)
      # OOMs on the leftover. The real lever is PYTORCH_CUDA_ALLOC_CONF=
      # expandable_segments:True (see ENV_ARGS below): reclaims the ~185 MiB
      # PyTorch keeps "reserved but unallocated", covering the transient with no
      # context cost. (2026-07-22)
      #
      # 2026-08-02 image rebuild (0.26.1rc1.dev251+g0033211c0) ate that headroom
      # again and crash-looped: V2 Model Runner + TRITON_ATTN (FA4 unavailable for
      # Gemma 4's heterogeneous head dims) + cudagraph FULL_AND_PIECEWISE. The
      # autotune transient missed by ~9 MiB (needed 168, had 158.88 free) — and
      # note flashinfer swallows per-tactic OOMs ("falling back to default
      # tactic") but the fallback's own OOM escapes and kills engine init.
      # expandable_segments was already on and doing its job (only 129 MiB
      # reserved-unallocated), so the fix this time is to skip the autotune
      # outright via --kernel-config (see EXTRA): the cache file is populated and
      # hits ("Config cache hit for fp4_gemm"), so tuning buys ~nothing here.
      #
      # --num-gpu-blocks-override 6780 PINS the KV pool, and DELIBERATELY takes
      # more than vLLM's profiler offers. Two separate reasons:
      #
      # 1. Profiling UNDER-estimates. vLLM compiles BEFORE the profiling pass
      #    that sizes KV, so the compile working set is resident while the peak
      #    is measured — memory it is about to release. A cold boot therefore
      #    reports only 6.87 GiB available (-> 5629 blocks / 87,388 tokens) and
      #    leaves ~2.2 GiB of the budget unused at steady state. Overriding
      #    claims most of it back (see the 2026-09-09 retune below for the
      #    current block count / token count / concurrency).
      #
      # 2. It closes the restart path, which is what actually crash-looped. The
      #    torch.compile cache lives in the container's writable layer and
      #    survives `docker restart` (only `docker rm` clears it). So boot 1
      #    profiles cold at 6.87 GiB but every --restart unless-stopped restart
      #    after it profiles WARM at 9.14 GiB, handing 2.27 GiB more to KV and
      #    starving what runs AFTER profiling. Autotune OOM'd on the leftover
      #    158 MiB; cudagraph capture (0.21 GiB) would have too. Pinning makes
      #    every boot identical regardless of cache state.
      #
      # tokens/block is NOT linear — 5629 -> 15.5 tok/blk but 7000 -> 17.0.
      # Do not extrapolate; boot it and read vllm:cache_config_info.
      #
      # HEADROOM: this leaves only a thin sliver free, which is fine but is NOT
      # a budget to spend further. It is stable rather than a countdown because
      # chunked prefill caps activations at --max-num-batched-tokens (4096) per
      # step, so neither prompt length nor concurrency moves the high-water
      # mark. Verified 2026-08-03 at a steady 117 MiB across: 100,027-token
      # prompt, 2 concurrent 40k prompts, repeat requests, and an image request
      # (896x896). Re-derive from a COLD boot after any image bump — and note
      # CONTEXT_SIZE is part of the torch.compile cache key, so changing it
      # forces a full recompile. (2026-08-03)
      #
      # 2026-09-09: 7000 -> 6780. That "re-derive after any image bump" note was
      # NOT honoured across b18/b19/b20/b21, and b21's larger non-KV footprint
      # ate the 117 MiB. Symptom was NOT a tight crash loop (which is why it sat
      # unnoticed): the engine served fine for 38 min / 4h13m, then a prefill
      # chunk hit the fp8-KV MLP buffer
      #   buf7 = empty_strided_cuda((s59=4096, 21504), torch.bfloat16) = 168 MiB
      # against 154.88 MiB free, died with EngineDeadError, 500'd the in-flight
      # request, and --restart unless-stopped silently brought it back. Both
      # crashes reported the SAME 154.88 MiB free — a hard deterministic
      # ceiling, not fragmentation drift, so it recurs on request shape alone.
      # 7000 blocks claimed 8.37 GiB while the 0.982 budget only affords 8.11
      # GiB (vLLM says so directly: "Replace gpu_memory_utilization config with
      # --kv-cache-memory=8710538839"); 6780 gives that ~266 MiB back, covering
      # the 168 MiB alloc with margin.
      #
      # MEASURED on the cold boot (b21, 2026-09-09), not extrapolated:
      #   6780 blocks -> 115,437 tokens, 1.10x at 105,000 ctx (17.03 tok/blk)
      #   device usage 31542 -> 31262 MiB idle, i.e. 280 MiB actually reclaimed
      # Verified serving: a 90,022-token prompt (many 4096 chunks — the exact
      # path that OOM'd) and 2 concurrent 38,521-token prompts, 0 OOM, 0 500s,
      # 0 restarts. Device usage plateaued at 31964 MiB after the FIRST large
      # prefill and did NOT move for the concurrent pair — re-confirming the
      # high-water mark is pinned by --max-num-batched-tokens, not by prompt
      # length or concurrency. NOTE the gpu_worker.py:871 line still prints
      # "kv cache memory in use is 8.37 GiB" under the override; it reports the
      # profiled figure, not the pinned pool. Trust nvidia-smi + the
      # kv_cache_utils token count instead.
      MODEL_ID="RedHatAI/gemma-4-31B-it-NVFP4"
      QUANTIZATION="compressed-tensors"
      CONTEXT_SIZE=105000
      INPUT_MODES="text,image"
      DOCKER_NAME="vllm_gemma4-31b-nvfp4-mtp"
      DOCKER_IMAGE="vllm-custom:latest"
      CPU_OFFLOAD=0
      GPU_MEM_UTIL=0.982
      ENV_ARGS=(--env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
      EXTRA=(--enable-prefix-caching --kv-cache-dtype fp8 --max-num-batched-tokens 4096 --max-num-seqs 2 --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --chat-template /etc/vllm/chat-templates/gemma4-force-think.jinja --speculative-config '{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":4}' --kernel-config '{"enable_flashinfer_autotune":false}' --num-gpu-blocks-override 6780)
      ;;
    4mc)
      # High-context turbo: LilaRest turbo (modelopt_fp4, TEXT-ONLY), NOT the
      # RedHatAI compressed-tensors of 4m. Single-user config that shrinks the
      # prefill reservation to buy KV: 4m's defaults reserve ~3 GiB of
      # activation/NVFP4 matmul workspace (straight out of the KV pool);
      # --max-num-batched-tokens 2560 --max-num-seqs 1 frees most of it back into
      # KV for more context. (The old 2560 hard floor was Gemma 4's bidirectional
      # image attention needing one 2496-token image item per batch — it no longer
      # binds now that this is text-only, so 2560 can go lower to buy more KV.)
      # The google MTP drafter is kept but UNVERIFIED on this quant; drop
      # --speculative-config if engine init balks.
      # Tune CONTEXT_SIZE to ~1.0x KV concurrency (re-derive from the log line).
      MODEL_ID="LilaRest/gemma-4-31B-it-NVFP4-turbo"
      QUANTIZATION="modelopt_fp4"
      CONTEXT_SIZE=100000
      INPUT_MODES="text"
      DOCKER_NAME="vllm_gemma4-31b-nvfp4-mtp-hc"
      DOCKER_IMAGE="vllm-custom:latest"
      CPU_OFFLOAD=0
      GPU_MEM_UTIL=0.982
      EXTRA=(--enable-prefix-caching --kv-cache-dtype fp8 --max-num-batched-tokens 2560 --max-num-seqs 1 --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --chat-template /etc/vllm/chat-templates/gemma4-force-think.jinja --speculative-config '{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":4}')
      ;;
    4f)
      # 4m but with NVFP4 KV cache. DOES NOT BOOT on the RTX 5090: NVFP4 KV needs
      # FlashInfer's trtllm-gen kernel, gated to SM100 (datacenter Blackwell);
      # the 5090 is SM120, so engine init dies "kv_cache_dtype not supported".
      # SM100-only placeholder — use 4m (fp8 KV) on this box.
      MODEL_ID="RedHatAI/gemma-4-31B-it-NVFP4"
      QUANTIZATION="compressed-tensors"
      CONTEXT_SIZE=128000
      INPUT_MODES="text,image"
      DOCKER_NAME="vllm_gemma4-31b-nvfp4-mtp-fp4kv"
      DOCKER_IMAGE="vllm-custom:latest"
      CPU_OFFLOAD=0
      GPU_MEM_UTIL=0.983
      EXTRA=(--enable-prefix-caching --kv-cache-dtype nvfp4 --kv-cache-dtype-skip-layers sliding_window --max-num-batched-tokens 4096 --max-num-seqs 2 --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --chat-template /etc/vllm/chat-templates/gemma4-force-think.jinja --speculative-config '{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":1}')
      ;;
    5)
      # NVIDIA Nemotron 3.5 Lightning: Mamba-2/MoE hybrid, 3B active of 30B.
      # Only 6 of 52 layers are attention (indices 5/12/19/26/33/42) at 2 KV
      # heads x 128 head_dim, so fp8 KV costs ~3 KiB/token — an order of
      # magnitude cheaper than the dense configs above — so KV is NOT the
      # binding constraint here; weights (~18 GiB) are. MEASURED on the first
      # cold boot (2026-08-13): the pool came out at 1,671,967 tokens, i.e.
      # 6.38x concurrency at a 262,144 ctx — six times more headroom than the
      # ~1.0x single-user target the other configs are tuned to. CONTEXT_SIZE is
      # therefore the full native 1,048,576 (max_position_embeddings) — the real
      # ceiling is the position embeddings, not VRAM.
      #
      # Counter-intuitively the pool got BIGGER when ctx was raised: 262,144 ctx
      # -> 1,671,967 tokens (6.38x), but 1,048,576 ctx -> 2,046,288 tokens
      # (1.95x). vLLM derives block_size from max_model_len (4240 at 262k), so
      # raising ctx re-blocks the pool rather than eating it. Do NOT assume a
      # bigger ctx costs KV here — measure both ends.
      #
      # Unlike every other config in this file, the lever to watch is the
      # weights, not the KV pool: 0.95 util leaves ~1.9 GiB free at steady
      # state, so there is genuine slack (contrast 4m's 117 MiB).
      #
      # QUANTIZATION=modelopt_mixed, NOT modelopt_fp4. The checkpoint is
      # modelopt-produced but quant_algo is MIXED_PRECISION: routed/shared
      # experts are W4A16_NVFP4 (group_size 16) while the Mamba in_proj/out_proj
      # are FP8. vLLM maps that to ModelOptMixedPrecisionConfig — its
      # override_quantization_method() returns "modelopt_mixed" on seeing
      # MIXED_PRECISION, and modelopt_fp4 would mis-handle the FP8 layers.
      #
      # MTP is native: config.json carries num_nextn_predict_layers=1 and vLLM
      # registers NemotronHMTPModel, rewriting model_type nemotron_h ->
      # nemotron_h_mtp automatically. So --speculative-config takes no "model"
      # key (same as options 1/2, unlike gemma 4m/4f). The predictor asserts
      # exactly 1 MTP layer; num_speculative_tokens 3 reuses that head
      # autoregressively, which is the count NVIDIA's card recommends for
      # low-concurrency single-user serving.
      #
      # Mamba-specific flags, both off-default and both needed:
      #   --mamba-backend flashinfer  (default is TRITON)
      #   --mamba-cache-mode align    (default "none"; prefix caching on a hybrid
      #                                model requires "align" — it caches the
      #                                Mamba state of each scheduler step's last
      #                                token and pads to attention page size)
      # --moe-backend marlin matches the target's W4A16_NVFP4 expert kernels,
      # but it is a GLOBAL setting and the MTP draft layer's MoE is UNQUANTIZED
      # (the mtp_layers_block_type layers are absent from `quantized_layers`).
      # Marlin is quantized-only, so the drafter's UnquantizedFusedMoEMethod
      # rejects it at load: "moe_backend='marlin' is not supported for
      # unquantized MoE" — engine init dies AFTER the target model has already
      # loaded, which makes it look like a target-side failure. Hence
      # "moe_backend":"triton" inside --speculative-config: SpeculativeConfig
      # takes a per-drafter override for exactly this case (its docstring cites
      # "quantized generator with unquantized drafter"). Do NOT "simplify" by
      # collapsing the two backends — they are deliberately different. (2026-08-13)
      #
      # GPU_MEM_UTIL is a deliberately conservative 0.95 for the first boot:
      # weights leave ~14 GiB free, so unlike the gemma configs there is no need
      # to scrape the last 2% — the slack absorbs the Marlin/FlashInfer autotune
      # transients that crash-looped options 4m/1. Push it up only after a clean
      # cold boot, and re-derive from the "GPU KV cache size" log line.
      #
      # NOTE: the model card claims vLLM >= v0.27.1. vllm-custom:latest reports
      # 0.26.1rc1.devNNN, but that is just main's version string lagging: the
      # build (commit edbc4969a76) already registers NemotronHForCausalLM,
      # NemotronHMTPModel, the nemotron_v3 reasoning parser, modelopt_mixed, and
      # the mamba/moe backend flags. Verified against ~/src/vllm/vllm.
      #
      # BOOTED AND VERIFIED 2026-08-13: healthy in 242s (warm weights; the first
      # run also downloads ~21.6 GB), tool/reasoning parsing OK, MTP accepting
      # 848/1569 draft tokens (~54%, ~2.6 tok/step; still 39% at position 2,
      # which is what justifies num_speculative_tokens 3).
      #
      # Heads up when testing: this is a HEAVY reasoner and the nemotron_v3
      # parser puts the think block in `reasoning`, leaving `content` NULL until
      # it finishes. A 300-token cap returned content=None mid-thought; it needed
      # ~570 completion tokens for a one-sentence answer. `vllm test` caps at 500
      # and prints only `content`, so it can print "None" on a working server —
      # that is the harness being terse, not a broken model.
      MODEL_ID="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
      QUANTIZATION="modelopt_mixed"
      CONTEXT_SIZE=1048576
      INPUT_MODES="text"
      DOCKER_NAME="vllm_nemotron35_lightning_30b_a3b"
      DOCKER_IMAGE="vllm-custom:latest"
      CPU_OFFLOAD=0
      GPU_MEM_UTIL=0.95
      EXTRA=(--enable-prefix-caching --kv-cache-dtype fp8 --mamba-backend flashinfer --mamba-cache-mode align --moe-backend marlin --max-num-batched-tokens 4096 --max-num-seqs 2 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}')
      ;;
    logs|l)
      RUNNING_NAMES=$(docker ps --filter "name=vllm_" --format "{{.Names}}")
      if [ -n "$RUNNING_NAMES" ]; then
        if [ "$(echo "$RUNNING_NAMES" | wc -l)" -gt 1 ]; then
          echo "Error: Multiple vLLM containers are running. Please stop all first ('vllm stop')."
          return 1
        fi
        echo "Showing live logs for: $RUNNING_NAMES"
        echo "Press Ctrl+C to exit logs"
        docker logs -f "$RUNNING_NAMES"
        return 0
      fi

      # None running: show logs from the most recently stopped container.
      STOPPED=$(docker ps -a --filter "name=vllm_" --filter "status=exited" --format "{{.Names}}")
      if [ -z "$STOPPED" ]; then
        echo "Error: No vLLM containers found (running or stopped)."
        return 1
      fi

      MOST_RECENT=$(echo "$STOPPED" | while read -r name; do
        printf '%s\t%s\n' "$(docker inspect --format='{{.State.FinishedAt}}' "$name")" "$name"
      done | sort -r | head -n1 | awk '{print $2}')

      echo "No running vLLM container. Showing logs from most recently stopped: $MOST_RECENT"
      docker logs "$MOST_RECENT"
      return 0
      ;;
    test|t)
      one_running_vllm > /dev/null || return 1

      echo "Enter your prompt (type END on a new line when done, or Ctrl+C to cancel):"
      USER_PROMPT=""
      while IFS= read -r line; do
        [ "$line" = "END" ] && break
        if [ -z "$USER_PROMPT" ]; then
          USER_PROMPT="$line"
        else
          USER_PROMPT+=$'\n'"$line"
        fi
      done

      if [ -z "$USER_PROMPT" ]; then
        echo "Error: Empty prompt provided."
        return 1
      fi

      echo ""
      echo "Sending request to model..."
      echo ""

      # Build JSON request using Python to properly escape everything.
      # NO explicit temperature: vLLM then applies each model's own
      # generation_config.json defaults (what every checkpoint here recommends
      # anyway), and diffusion configs (3/3n) stay usable — they 400 on any
      # temperature != 1.0, since the denoiser owns its own schedule.
      RESPONSE=$(python3 -c "
import json
import sys
import urllib.request

model_id = '$(get_vllm_model_id)'
prompt = '''$USER_PROMPT'''

data = {
    'model': model_id,
    'messages': [{'role': 'user', 'content': prompt}],
    'max_tokens': 500
}

req = urllib.request.Request(
    'http://localhost:8000/v1/chat/completions',
    data=json.dumps(data).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)

try:
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))
        print(result['choices'][0]['message']['content'])
except Exception as e:
    sys.exit(1)
" 2>/dev/null)

      if [ $? -ne 0 ]; then
        echo "Error: Failed to get response from model. Is it still loading?"
        return 1
      fi

      echo "$RESPONSE"

      echo ""
      return 0
      ;;
    status|st|ps)
      # Shorthand for `docker ps` scoped to this script's containers. Show
      # stopped ones too (-a) so a crashed/exited server is visible, not silently
      # absent. Same `name=vllm_` filter the rest of the script keys off.
      docker ps -a --filter "name=vllm_" \
        --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
      return 0
      ;;
    ctx)
      RUNNING_NAMES=$(one_running_vllm) || return 1

      MODEL_INFO=$(curl -s http://localhost:8000/v1/models)
      if [ -z "$MODEL_INFO" ]; then
        echo "Error: Cannot reach vLLM API at http://localhost:8000 (still loading?)"
        return 1
      fi

      MODEL_ID=$(echo "$MODEL_INFO" | python3 -c "import sys, json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
      CTX_LEN=$(echo "$MODEL_INFO" | python3 -c "import sys, json; print(json.load(sys.stdin)['data'][0].get('max_model_len', '?'))" 2>/dev/null)

      echo -e "${BOLD}Container:${RESET}      $RUNNING_NAMES"
      echo -e "${BOLD}Model:${RESET}          $MODEL_ID"
      echo -e "${BOLD}Context length:${RESET} $CTX_LEN"
      log_vllm_concurrency "$RUNNING_NAMES"
      return 0
      ;;
    stop|s|kill|k)
      IDS=$(docker ps -q --filter "name=vllm_")
      if [ -z "$IDS" ]; then
        echo "No running vLLM containers found."
        return 0
      fi
      sample_token_usage
      echo "$IDS" | xargs -r docker stop
      return 0
      ;;
    *)
      return 0
      ;;
  esac

  # Bare `vllm rec` inferred the running config, so confirm before tearing it
  # down (unless -y / piped `yes`; the prompt reads a 'y' from stdin). Explicit
  # `vllm rec <cfg>` skips this — the caller already named the target.
  if [ -n "$REC_INFERRED" ]; then
    echo -e "${YELLOW}Recreate ${BOLD}$DOCKER_NAME${RESET}${YELLOW} (model: $MODEL_ID, ctx: $CONTEXT_SIZE)?${RESET}"
    if [ -n "$REC_YES" ]; then
      echo -e "${GRAY}(-y) Proceeding without confirmation.${RESET}"
    else
      echo -ne "${BOLD}Stop, remove, and start fresh? [y/N]:${RESET} "
      read -r REC_REPLY
      case "$REC_REPLY" in
        y|Y|yes|YES) ;;
        *) echo "Aborted."; return 0 ;;
      esac
    fi
  fi

  # Handle rm mode: stop (if running) then remove the container for the selected model
  if [ -n "$RM_MODE" ]; then
    if [ -n "$(docker ps -q -f name=^/${DOCKER_NAME}$)" ]; then
      sample_token_usage
      docker stop "$DOCKER_NAME"
    fi
    docker rm "$DOCKER_NAME"
    return 0
  fi

  # Handle rec mode: stop+rm existing container so the fresh start path below runs
  if [ -n "$REC_MODE" ]; then
    if [ -n "$(docker ps -q -f name=^/${DOCKER_NAME}$)" ]; then
      sample_token_usage
      docker stop "$DOCKER_NAME"
    fi
    if [ -n "$(docker ps -a -q -f name=^/${DOCKER_NAME}$)" ]; then
      docker rm "$DOCKER_NAME"
    fi
  fi

  # Single GPU + single port: clear out any other running vLLM server first. This
  # sits above BOTH start paths (restart + fresh run) on purpose, and after the
  # rm/rec handling above, so `rm` never touches an unrelated container.
  stop_other_vllm_containers "$DOCKER_NAME"

  # Container exists but is stopped -> just restart it (config is baked in).
  if [ "$(docker ps -a -q -f name=^/${DOCKER_NAME}$ -f status=exited)" ]; then
    echo "Starting existing Docker container: $DOCKER_NAME"
    ~/scripts/update-openclaw-wan-ip.sh
    START_TIME=$(date +%s)
    # Propagate the failure instead of falling through to wait_for_healthy, which
    # would otherwise poll a container that never started. A start that fails on
    # "port is already allocated" also bricks this container's networking for
    # good (see wait_for_port_release) — hence the `vllm rec` hint.
    if ! docker start "$DOCKER_NAME"; then
      echo "Error: docker start failed for $DOCKER_NAME"
      # NB: on a start that failed before the container ran, these lines are from
      # its PREVIOUS run — stale, but still the best clue when the process itself
      # is what died.
      echo "--- last 10 log lines from $DOCKER_NAME ---"
      docker logs --tail 10 "$DOCKER_NAME" 2>&1 | sed 's/^/  /'
      echo "-------------------------------------------"
      echo "If this was a port/network failure, recreate the container: vllm rec $model_choice"
      return 1
    fi
    wait_for_healthy "$DOCKER_NAME" "$START_TIME" || return 1
    sync_agent_models "$INPUT_MODES" "$CONTEXT_SIZE"
    return 0
  fi

  # Optional offload args as an array (zsh doesn't word-split unquoted expansions).
  OFFLOAD_ARGS=()
  if [ "$CPU_OFFLOAD" -gt 0 ]; then
    OFFLOAD_ARGS=(--cpu-offload-gb "$CPU_OFFLOAD")
  fi

  # HuggingFace token: read from the host's HF_TOKEN env var (never committed).
  # Only passed through when set; public models work without it.
  HF_ENV_ARGS=()
  if [ -n "${HF_TOKEN:-}" ]; then
    HF_ENV_ARGS=(--env "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}")
  fi

  # Always detached (-d) so the menu/direct-arg calls never block the shell.
  ~/scripts/update-openclaw-wan-ip.sh
  START_TIME=$(date +%s)
  # -v ~/.cache/vllm PERSISTS vLLM's startup caches across `rec` (2026-09-05).
  # Without it they live in the container's writable layer and every recreate
  # pays a COLD boot: ~28s torch.compile of backbone+eagle_head, plus a ~20s
  # FlashInfer fp4 autotune pass whose transient is what OOM-crashed config 1's
  # first b21 boot (32 MiB short, at the MTP drafter's cudagraph warmup; the
  # unless-stopped restart then succeeded warm). Contents: torch_compile_cache/
  # (~200 MB of AOT graphs), flashinfer_autotune_cache/ (the tuned
  # bf16_fp4_cute_dsl_gemm configs -- the memory-relevant one), modelinfos/.
  # Safe to share one host dir across ALL configs: every path is content-keyed
  # (flashinfer version, and hashes of vLLM version + compilation config), so
  # entries never collide and a stale one is never reused. Corollary: an image
  # rebuild is still cold ONCE (new hashes) -- expect one crash+auto-restart on
  # the first `rec` after a build. Nothing evicts: ~200 MB per (build x config),
  # so `rm -rf ~/.cache/vllm` occasionally.
  docker run -d --name $DOCKER_NAME --runtime nvidia --gpus all \
    --restart unless-stopped \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/.cache/vllm:/root/.cache/vllm \
    -v ~/scripts/chat-templates:/etc/vllm/chat-templates:ro \
    "${HF_ENV_ARGS[@]}" \
    "${ENV_ARGS[@]}" \
    -p 8000:8000 \
    --ipc=host \
    --health-cmd "curl -f http://localhost:8000/health || exit 1" \
    --health-interval 60s \
    --health-start-interval 1s \
    --health-timeout 5s \
    --health-retries 10 \
    --health-start-period 60s \
    "$DOCKER_IMAGE" \
    "${CMD_PREFIX[@]}" "$MODEL_ID" \
    --quantization "$QUANTIZATION" \
    --max-model-len "$CONTEXT_SIZE" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --dtype auto \
    "${EXTRA[@]}" \
    "${OFFLOAD_ARGS[@]}"

  wait_for_healthy "$DOCKER_NAME" "$START_TIME" || return 1
  sync_agent_models "$INPUT_MODES" "$CONTEXT_SIZE"
}

# Run by path (see ~/.bash_aliases: alias vllm, aid(), asyncservice()). Dispatch
# CLI args to vllm(), except `model-id`/`id` -> get_vllm_model_id (used by aid()).
# When sourced, `return 0` succeeds so dispatch is skipped and only the function
# definitions load (portable across bash and zsh).
if ! (return 0 2>/dev/null); then
  case "$1" in
    model-id|id|get_vllm_model_id)
      get_vllm_model_id
      ;;
    *)
      vllm "$@"
      ;;
  esac
fi
