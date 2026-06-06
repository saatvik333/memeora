#!/usr/bin/env bash
# Local mirror of the GitHub CI gate (.github/workflows/ci.yml).
# Run before pushing — catches everything CI checks, on your machine.
#
#   scripts/check.sh          full gate (what CI runs, all features)
#   scripts/check.sh --fast   skip the --all-features / cargo-deny passes
#                             (quick loop; does NOT fully match CI)
#
# Note: --all-features compiles the fastembed/ONNX Runtime stack. The first run
# is slow; cargo caches it afterwards.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

FAST=0
[[ "${1:-}" == "--fast" ]] && FAST=1

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

step "rustfmt --check"
cargo fmt --all --check

if [[ "$FAST" == 1 ]]; then
    step "clippy (default features)"
    cargo clippy --workspace --all-targets -- -D warnings
    step "test (default features)"
    cargo test --workspace
    printf '\n\033[1m--fast done.\033[0m CI also runs --all-features clippy/test + cargo-deny.\n'
    exit 0
fi

step "clippy --all-features -D warnings"
cargo clippy --workspace --all-targets --all-features -- -D warnings

step "test --all-features"
cargo test --workspace --all-features

if command -v cargo-deny >/dev/null 2>&1; then
    # --all-features matches the cargo-deny-action default; without it the
    # fastembed-only deps (and their licenses) aren't in the graph.
    step "cargo-deny --all-features check"
    cargo deny --all-features check
else
    printf '\n\033[33mskip:\033[0m cargo-deny not installed (CI still runs it).\n'
    printf '      install: cargo install cargo-deny --locked\n'
fi

printf '\n\033[1;32mAll checks passed.\033[0m\n'
