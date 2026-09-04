#!/usr/bin/env sh
# ==============================================================================
# Synapse AI-Native Programming Language - POSIX One-Liner Demo Quickstart
# Supported OS: Linux, macOS, WSL, BSD
# Usage:
#   curl -fsSL https://get.synapse-lang.org/demo.sh | sh
#   or locally: ./scripts/demo.sh [preset]
#   Presets: nanogpt (default), matmul, tour, dataloader, all
# ==============================================================================

set -e

printf "\033[90m======================================================\033[0m\n"
printf "\033[36m  Synapse AI-Native Language - Instant Demo Quickstart \033[0m\n"
printf "\033[90m======================================================\033[0m\n"

# 1. Detect Working Python 3.10+
PYTHON_BIN=""
for cand in python3 python py; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import sys" >/dev/null 2>&1; then
        PYTHON_BIN="$cand"
        break
    fi
done

# 2. Determine target preset
PRESET="${1:-${PRESET:-nanogpt}}"

# Determine source directory (if run locally from clone or repo)
SRC_DIR=""
if [ -n "$SYNAPSE_SOURCE_DIR" ]; then
    SRC_DIR="$SYNAPSE_SOURCE_DIR"
elif [ -f "$0" ] && [ -d "$(dirname "$0")/../synapse" ]; then
    SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
elif [ -d "./synapse" ]; then
    SRC_DIR="$(pwd)"
elif [ -d "$HOME/.synapse/lib/synapse" ]; then
    SRC_DIR="$HOME/.synapse/lib"
fi

if [ -z "$PYTHON_BIN" ]; then
    printf "\033[33m[!] Python runtime not found in PATH.\033[0m\n"
    # Fallback to precompiled standalone edge binary if available
    NATIVE_BIN="$SRC_DIR/examples/edge_nanogpt/nanogpt"
    if [ -f "$NATIVE_BIN" ] && [ -x "$NATIVE_BIN" ]; then
        printf "\033[32m[*] Executing native C99 standalone binary: %s\033[0m\n" "$NATIVE_BIN"
        exec "$NATIVE_BIN"
    else
        printf "\033[31mError: Python 3.10+ or compiled nanogpt binary is required.\033[0m\n" >&2
        exit 1
    fi
fi

# 3. Configure PYTHONPATH
if [ -n "$SRC_DIR" ]; then
    export PYTHONPATH="$SRC_DIR:${PYTHONPATH}"
fi

# 4. Check interactive vs non-interactive
CLI_EXTRA=""
if [ ! -t 0 ]; then
    CLI_EXTRA="--non-interactive"
fi

# 5. Run chosen preset(s)
if [ "$PRESET" = "all" ]; then
    PRESETS="nanogpt matmul tour dataloader"
else
    PRESETS="$PRESET"
fi

for p in $PRESETS; do
    printf "\n\033[90m[*] Running Synapse Showcase Preset: [%s]...\033[0m\n" "$p"
    if [ -n "$CLI_EXTRA" ]; then
        "$PYTHON_BIN" -m synapse.cli demo --preset="$p" "$CLI_EXTRA"
    else
        "$PYTHON_BIN" -m synapse.cli demo --preset="$p"
    fi
done

printf "\n\033[90m======================================================\033[0m\n"
printf "\033[32m  Synapse Quickstart Demo Finished!\033[0m\n"
printf "\033[36m  Install Full CLI: curl -fsSL https://get.synapse-lang.org/install.sh | bash\033[0m\n"
printf "\033[90m======================================================\033[0m\n"
