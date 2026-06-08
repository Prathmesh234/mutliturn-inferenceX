#!/bin/bash
# Launch the KernelBook -> Triton run on ONE GB300 node (4 GPUs, 4 workers).
#
#   sbatch run.sh                       # problems 0..299
#   START=300 BATCH=300 sbatch run.sh   # next batch
#   sbatch --constraint=DH1-153-US-EAST-04B run.sh   # pin to rack 1
#
# Or interactively (foreground):  bash run.sh
#
# CLUSTER TOPOLOGY (the rules this must respect):
#   * 2 racks  : DH1-153-US-EAST-04B, DH2-058-US-EAST-04B (Slurm topology blocks)
#   * 18 servers per rack
#   * 4 GPUs (gb300) per server
# This job uses 1 server / 4 GPUs  -> well within the rules.
# To scale, run one job PER server (e.g. a Slurm array, each --nodes=1 --gres=gpu:4
# with a different START offset). Keep any single multi-node job <= 18 nodes and
# pin it to ONE rack with --constraint=<rack> so it stays in one NVLink/block domain.
#SBATCH --job-name=kb-triton
#SBATCH --partition=gb300
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
# Set --mem explicitly: the partition's DefMemPerCPU=14336 would auto-request
# 64*14GB ~= 896GB for 64 CPUs, which exceeds the node's allocatable memory
# (~868GB after MemSpecLimit) and fails with "node configuration not available".
#SBATCH --mem=800G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm-%j.out

set -euo pipefail
# Under `sbatch`, $0 is a spooled copy of this script, so locate the project via
# the submit dir (run `sbatch run.sh` from inside the project). For bash/srun,
# fall back to the script's own directory.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}"
mkdir -p runs

# --- arch-correct uv -------------------------------------------------------
# The login node is x86_64 but GB300 compute nodes are aarch64, so uv (and the
# venv) must be the aarch64 build. Bootstrap a per-arch uv inside the job.
export UV_INSTALL_DIR="$HOME/.local/uv-$(uname -m)"
if ! "$UV_INSTALL_DIR/uv" --version >/dev/null 2>&1; then
  echo ">> installing uv for $(uname -m) into $UV_INSTALL_DIR"
  curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR="$UV_INSTALL_DIR" INSTALLER_NO_MODIFY_PATH=1 sh
fi
export PATH="$UV_INSTALL_DIR:$PATH"

# --- arch-correct claude ---------------------------------------------------
# `claude` in ~/.local/bin is the x86_64 login-node binary and won't exec on the
# aarch64 GB300 nodes ("Exec format error"). Fetch the arm64 build into a per-arch
# dir (without `claude install`, so the shared x86_64 symlink/creds are untouched)
# and point the harness at it via CLAUDE_BIN. Credentials in ~/.claude are reused.
CLAUDE_DIR="$HOME/.local/claude-$(uname -m)"
if ! "$CLAUDE_DIR/claude" --version >/dev/null 2>&1; then
  echo ">> fetching arm64 claude into $CLAUDE_DIR"
  mkdir -p "$CLAUDE_DIR"
  case "$(uname -m)" in x86_64|amd64) PLAT=linux-x64 ;; arm64|aarch64) PLAT=linux-arm64 ;; esac
  VER="$(curl -fsSL https://downloads.claude.ai/claude-code-releases/latest)"
  curl -fSL "https://downloads.claude.ai/claude-code-releases/$VER/$PLAT/claude" -o "$CLAUDE_DIR/claude"
  chmod +x "$CLAUDE_DIR/claude"
fi
export CLAUDE_BIN="$CLAUDE_DIR/claude"
echo ">> claude: $("$CLAUDE_BIN" --version)"

# --- dependencies ----------------------------------------------------------
# Core deps always install (pure-python). Eval deps (torch/triton) are
# best-effort: if they don't resolve for GB300 yet, generation still runs and
# the evaluator reports "env_unavailable" instead of crashing.
uv sync --python 3.12
uv sync --python 3.12 --extra eval || echo "WARN: torch/triton not installed -> eval will report env_unavailable"

# --- run -------------------------------------------------------------------
START="${START:-0}"
BATCH="${BATCH:-300}"
OUTDIR="${OUTDIR:-runs}"          # append-only results dir; override to keep runs/ untouched
mkdir -p "$OUTDIR"
echo ">> solving problems [$START, $((START+BATCH))) on $(hostname) -> $OUTDIR"

uv run python solve.py \
  --start "$START" \
  --batch-size "$BATCH" \
  --num-workers 4 \
  --num-gpus 4 \
  --model claude-opus-4-8 \
  --min-evals "${MIN_EVALS:-8}" \
  --max-evals "${MAX_EVALS:-0}" \
  --output-dir "$OUTDIR"
