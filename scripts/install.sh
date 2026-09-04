#!/usr/bin/env sh
# ==============================================================================
# Synapse AI-Native Programming Language - POSIX One-Click Installer
# Supported OS: Linux, macOS, WSL, BSD
# Usage:
#   curl -fsSL https://get.synapse-lang.org/install.sh | bash
#   or locally: ./scripts/install.sh
# ==============================================================================

set -e

printf "\033[90m======================================================\033[0m\n"
printf "\033[36m  Synapse AI-Native Language - POSIX Installer        \033[0m\n"
printf "\033[90m======================================================\033[0m\n"

# 1. Detect Python 3.10+
PYTHON_BIN=""
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

if [ -z "$PYTHON_BIN" ]; then
    printf "\033[31mError: Python 3.10+ is required but not found in PATH.\033[0m\n" >&2
    exit 1
fi

printf "\033[32m[1/5] Python runtime detected: %s\033[0m\n" "$("$PYTHON_BIN" --version 2>&1)"

# 2. Determine target directories
INSTALL_DIR="${SYNAPSE_HOME:-$HOME/.synapse}"
BIN_DIR="$INSTALL_DIR/bin"
LIB_DIR="$INSTALL_DIR/lib"
DEST_PKG_DIR="$LIB_DIR/synapse"

# Determine source directory (if run locally from clone)
SRC_DIR=""
if [ -n "$SYNAPSE_SOURCE_DIR" ]; then
    SRC_DIR="$SYNAPSE_SOURCE_DIR"
elif [ -f "$0" ] && [ -d "$(dirname "$0")/../synapse" ]; then
    SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
fi

printf "\033[36m[2/5] Preparing target directories: %s\033[0m\n" "$INSTALL_DIR"
mkdir -p "$BIN_DIR"
mkdir -p "$LIB_DIR"

# 3. Install/Copy Synapse package files
printf "\033[36m[3/5] Installing Synapse core libraries...\033[0m\n"
if [ -n "$SRC_DIR" ] && [ -d "$SRC_DIR/synapse" ]; then
    rm -rf "$DEST_PKG_DIR"
    cp -R "$SRC_DIR/synapse" "$DEST_PKG_DIR"
else
    # Remote fallback: if piped from curl, install via pip or download bundle
    if "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
        "$PYTHON_BIN" -m pip install --target "$LIB_DIR" synapse-lang >/dev/null 2>&1 || true
    fi
fi

# Create executable runner in ~/.synapse/bin/synapse
LAUNCHER="$BIN_DIR/synapse"
cat << 'EOF' > "$LAUNCHER"
#!/usr/bin/env sh
SYNAPSE_HOME="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$SYNAPSE_HOME/lib:${PYTHONPATH}"

if command -v python3 >/dev/null 2>&1; then
    exec python3 -m synapse.cli "$@"
elif command -v python >/dev/null 2>&1; then
    exec python -m synapse.cli "$@"
else
    echo "Error: Python 3 not found. Please install Python 3.10+." >&2
    exit 1
fi
EOF

chmod +x "$LAUNCHER"
printf "\033[32m  -> Launcher created: %s\033[0m\n" "$LAUNCHER"

# 4. PATH export configuration in shell profiles
printf "\033[36m[4/5] Configuring shell environment PATH...\033[0m\n"
PATH_LINE='export PATH="$HOME/.synapse/bin:$PATH"'
CONFIGURED=0

for RC_FILE in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.profile"; do
    if [ -f "$RC_FILE" ] || [ "$(basename "$RC_FILE")" = ".profile" ]; then
        if [ ! -f "$RC_FILE" ]; then
            touch "$RC_FILE"
        fi
        if grep -qF "$PATH_LINE" "$RC_FILE" 2>/dev/null; then
            printf "\033[90m  -> %s already contains Synapse PATH export.\033[0m\n" "$RC_FILE"
        else
            printf "\n# Synapse AI-Native Language CLI\n%s\n" "$PATH_LINE" >> "$RC_FILE"
            printf "\033[32m  -> Added Synapse PATH export to %s\033[0m\n" "$RC_FILE"
            CONFIGURED=1
        fi
    fi
done

# Export for current subshell
export PATH="$BIN_DIR:$PATH"

# 5. Version verification and CLI test
if [ "$SYNAPSE_SKIP_TEST" != "1" ]; then
    printf "\033[36m[5/5] Verifying installation and version...\033[0m\n"
    VER_OUT="$("$LAUNCHER" --version 2>&1 || true)"
    case "$VER_OUT" in
        *Synapse*)
            printf "\033[32m  -> Verification passed: %s\033[0m\n" "$VER_OUT"
            ;;
        *)
            printf "\033[33m  -> Notice: '%s' produced output: %s\033[0m\n" "$LAUNCHER" "$VER_OUT"
            ;;
    esac
else
    printf "\033[90m[5/5] Verification test skipped.\033[0m\n"
fi

printf "\n\033[90m======================================================\033[0m\n"
printf "\033[32m  Synapse installed successfully!\033[0m\n"
printf "\033[36m  Restart your terminal or run: source ~/.profile\033[0m\n"
printf "\033[36m  Command: synapse --version\033[0m\n"
printf "\033[90m======================================================\033[0m\n"
