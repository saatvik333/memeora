#!/bin/sh
# memeora — interactive installer.
#
#   curl --proto '=https' --tlsv1.2 -fsSL \
#     https://raw.githubusercontent.com/saatvik333/memeora/main/scripts/install.sh | sh
#
# Walks you through installing the memeora binaries, the embedding model, your
# coding-tool adapters, and the daemon. Interactive by default (prompts read from
# /dev/tty so it works under `curl | sh`); fully scriptable with flags + env for CI.
#
# It NEVER uses sudo, installs only into your user directories, backs up any file
# it edits, and is safe to re-run (idempotent).
#
# shellcheck shell=dash

set -eu

# ----------------------------------------------------------------------------- #
# Constants
# ----------------------------------------------------------------------------- #
REPO="saatvik333/memeora"
DIST_INSTALLER="https://github.com/${REPO}/releases/latest/download/memeora-installer.sh"
DASHBOARD_DEFAULT_ADDR="127.0.0.1:7878"
KNOWN_ADAPTERS="claude codex antigravity opencode mcp"

# ----------------------------------------------------------------------------- #
# Defaults (overridable by flags / env)
# ----------------------------------------------------------------------------- #
ASSUME_YES="${MEMEORA_YES:-0}"
DRY_RUN="${MEMEORA_DRY_RUN:-0}"
METHOD="${MEMEORA_METHOD:-auto}"        # auto | cargo-dist | brew | npm | source
INSTALL_DIR="${MEMEORA_DIR:-}"          # empty = installer's default
OFFLINE="${MEMEORA_OFFLINE:-0}"         # 1 = don't download the model
START_DAEMON="${MEMEORA_START_DAEMON:-ask}" # ask | 1 | 0
DASHBOARD="${MEMEORA_DASHBOARD:-on}"    # on | off
WIRE_HOOKS="${MEMEORA_WIRE_HOOKS:-0}"   # 1 = also merge hook config (Claude/Codex)
ADAPTERS="${MEMEORA_ADAPTERS:-ask}"     # ask | csv of KNOWN_ADAPTERS | none

# ----------------------------------------------------------------------------- #
# Output helpers (colour only on a TTY, honouring NO_COLOR)
# ----------------------------------------------------------------------------- #
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
	C_RESET=$(printf '\033[0m'); C_BOLD=$(printf '\033[1m')
	C_BLUE=$(printf '\033[34m'); C_GREEN=$(printf '\033[32m')
	C_YELLOW=$(printf '\033[33m'); C_RED=$(printf '\033[31m'); C_DIM=$(printf '\033[2m')
else
	C_RESET=; C_BOLD=; C_BLUE=; C_GREEN=; C_YELLOW=; C_RED=; C_DIM=
fi

info()  { printf '%s\n' "${C_BLUE}::${C_RESET} $*"; }
step()  { printf '\n%s\n' "${C_BOLD}${C_BLUE}==>${C_RESET} ${C_BOLD}$*${C_RESET}"; }
ok()    { printf '%s\n' "${C_GREEN} ✓${C_RESET} $*"; }
warn()  { printf '%s\n' "${C_YELLOW} !${C_RESET} $*" >&2; }
err()   { printf '%s\n' "${C_RED}error:${C_RESET} $*" >&2; }
dim()   { printf '%s\n' "${C_DIM}$*${C_RESET}"; }
die()   { err "$@"; exit 1; }

# Show a command, and run it unless --dry-run.
run() {
	printf '%s\n' "${C_DIM}\$ $*${C_RESET}"
	[ "$DRY_RUN" = 1 ] && return 0
	"$@"
}

have() { command -v "$1" >/dev/null 2>&1; }

# ----------------------------------------------------------------------------- #
# Interaction (reads from /dev/tty so prompts work under `curl | sh`)
# ----------------------------------------------------------------------------- #
INTERACTIVE=0
if [ "$ASSUME_YES" != 1 ] && [ -r /dev/tty ] && [ -t 1 ]; then INTERACTIVE=1; fi

# ask_yes_no <prompt> <default 0|1> -> returns 0 for yes, 1 for no
ask_yes_no() {
	_prompt=$1; _default=$2
	if [ "$INTERACTIVE" != 1 ]; then return "$([ "$_default" = 1 ] && echo 0 || echo 1)"; fi
	if [ "$_default" = 1 ]; then _hint="[Y/n]"; else _hint="[y/N]"; fi
	printf '%s %s ' "${C_BOLD}?${C_RESET} $_prompt" "$_hint" >/dev/tty
	IFS= read -r _ans </dev/tty || _ans=
	case "$_ans" in
		[Yy]*) return 0 ;;
		[Nn]*) return 1 ;;
		*) return "$([ "$_default" = 1 ] && echo 0 || echo 1)" ;;
	esac
}

# ----------------------------------------------------------------------------- #
# Usage
# ----------------------------------------------------------------------------- #
usage() {
	cat <<EOF
${C_BOLD}memeora installer${C_RESET} — local-first memory for your AI coding tools.

Usage: install.sh [options]   (interactive by default)

Options:
  --yes                 non-interactive; accept defaults (CI)
  --dry-run             print what would happen; change nothing
  --method <m>          binary install: auto|cargo-dist|brew|npm|source
  --dir <path>          custom install dir for the binaries
  --offline             do NOT download the embedding model (set one up later)
  --adapters <csv>      tools to wire, of: ${KNOWN_ADAPTERS} (or 'none')
  --wire-hooks          also merge auto-capture hook config (Claude/Codex)
  --[no-]daemon         start (or don't) the daemon at the end
  --dashboard <on|off>  daemon dashboard on 127.0.0.1:7878 (default on)
  -h, --help            this help

Everything is also settable via MEMEORA_* env (MEMEORA_YES, MEMEORA_METHOD, …).
EOF
}

# ----------------------------------------------------------------------------- #
# Arg parsing
# ----------------------------------------------------------------------------- #
while [ $# -gt 0 ]; do
	case "$1" in
		--yes|-y) ASSUME_YES=1 ;;
		--dry-run) DRY_RUN=1 ;;
		--method) METHOD=${2:?--method needs a value}; shift ;;
		--method=*) METHOD=${1#*=} ;;
		--dir) INSTALL_DIR=${2:?--dir needs a value}; shift ;;
		--dir=*) INSTALL_DIR=${1#*=} ;;
		--offline) OFFLINE=1 ;;
		--adapters) ADAPTERS=${2:?--adapters needs a value}; shift ;;
		--adapters=*) ADAPTERS=${1#*=} ;;
		--wire-hooks) WIRE_HOOKS=1 ;;
		--daemon) START_DAEMON=1 ;;
		--no-daemon) START_DAEMON=0 ;;
		--dashboard) DASHBOARD=${2:?--dashboard needs a value}; shift ;;
		--dashboard=*) DASHBOARD=${1#*=} ;;
		-h|--help) usage; exit 0 ;;
		*) die "unknown option: $1 (try --help)" ;;
	esac
	# If --yes was set, drop interactivity for the rest of the run.
	if [ "$ASSUME_YES" = 1 ]; then INTERACTIVE=0; fi
	shift
done

# ----------------------------------------------------------------------------- #
# Download helper (TLS-pinned, retrying; curl or wget)
# ----------------------------------------------------------------------------- #
fetch() { # fetch <url> -> stdout
	if have curl; then
		curl --proto '=https' --tlsv1.2 -fsSL --retry 3 "$1"
	elif have wget; then
		wget --https-only -qO- "$1"
	else
		die "need curl or wget to download files"
	fi
}

# ----------------------------------------------------------------------------- #
# Step 1 — binaries
# ----------------------------------------------------------------------------- #
install_binaries() {
	step "Install the memeora binaries"
	if have memeora && have memeora-daemon && have memeora-mcp && have memeora-hook; then
		ok "binaries already on PATH ($(memeora --version 2>/dev/null || echo memeora))"
		if ! ask_yes_no "Reinstall anyway?" 0; then return 0; fi
	fi

	_method=$METHOD
	if [ "$_method" = auto ]; then
		if [ "$INTERACTIVE" = 1 ]; then
			info "Install method:"
			dim "  1) cargo-dist installer (recommended; prebuilt, checksum-verified)"
			dim "  2) Homebrew$(have brew || printf ' (brew not found)')"
			dim "  3) npm/bun (@memeora/memeora)"
			dim "  4) build from source (needs Rust/cargo)"
			printf '%s ' "${C_BOLD}?${C_RESET} choice [1]:" >/dev/tty
			IFS= read -r _c </dev/tty || _c=1
			case "${_c:-1}" in
				2) _method=brew ;; 3) _method=npm ;; 4) _method=source ;; *) _method=cargo-dist ;;
			esac
		else
			_method=cargo-dist
		fi
	fi

	case "$_method" in
		brew)
			have brew || die "Homebrew not found; choose another --method"
			run brew install "${REPO%/*}/tap/memeora"
			;;
		npm)
			if have bun; then run bun add -g @memeora/memeora
			elif have npm; then run npm install -g @memeora/memeora
			else die "neither bun nor npm found; choose another --method"; fi
			;;
		source)
			have cargo || die "cargo not found; install Rust from https://rustup.rs first"
			run cargo install --git "https://github.com/${REPO}" memeora
			;;
		cargo-dist|*)
			_args=""
			[ -n "$INSTALL_DIR" ] && _args="--install-dir $INSTALL_DIR"
			info "fetching the cargo-dist installer"
			_tmp=$(mktemp)
			if ! fetch "$DIST_INSTALLER" >"$_tmp" || [ ! -s "$_tmp" ]; then
				rm -f "$_tmp"
				err "could not download the release installer (is a version published yet?)"
				if have cargo && ask_yes_no "Build from source with cargo instead?" 1; then
					run cargo install --git "https://github.com/${REPO}" memeora
				else
					die "no binaries installed; see https://github.com/${REPO}#install"
				fi
			else
				# shellcheck disable=SC2086
				run sh "$_tmp" $_args
				rm -f "$_tmp"
			fi
			;;
	esac

	if [ "$DRY_RUN" != 1 ] && ! have memeora-daemon; then
		warn "memeora-daemon is not on PATH yet — open a new shell or add the install dir to PATH,"
		warn "then re-run this installer (it is idempotent)."
	else
		ok "binaries installed"
	fi
}

# ----------------------------------------------------------------------------- #
# Step 2 — embedding model (consent)
# ----------------------------------------------------------------------------- #
ALLOW_DOWNLOAD=0
choose_model() {
	step "Embedding model"
	if [ "$OFFLINE" = 1 ]; then
		ALLOW_DOWNLOAD=0
	elif [ "$INTERACTIVE" = 1 ]; then
		dim "memeora runs a small local embedding model (~130 MB, BGE-small). It is offline"
		dim "by default and won't download without consent."
		if ask_yes_no "Download the model now from HuggingFace?" 1; then ALLOW_DOWNLOAD=1; else ALLOW_DOWNLOAD=0; fi
	else
		ALLOW_DOWNLOAD=1 # non-interactive default: opt in (use --offline to decline)
	fi

	if [ "$ALLOW_DOWNLOAD" = 1 ]; then
		ok "model will download on first daemon start (MEMEORA_ALLOW_MODEL_DOWNLOAD=1)"
	else
		warn "staying offline — the daemon will refuse to start until a model is present."
		dim "   Provide an offline bundle in ~/.memeora/models (or set MEMEORA_MODELS_DIR),"
		dim "   then verify with: memeora models verify"
	fi
}

# ----------------------------------------------------------------------------- #
# Step 3 — adapters
# ----------------------------------------------------------------------------- #
SELECTED_ADAPTERS=""
choose_adapters() {
	step "Wire memeora into your coding tools"
	if [ "$ADAPTERS" = none ]; then SELECTED_ADAPTERS=""; return 0; fi
	if [ "$ADAPTERS" != ask ]; then
		SELECTED_ADAPTERS=$(printf '%s' "$ADAPTERS" | tr ',' ' ')
		return 0
	fi
	if [ "$INTERACTIVE" != 1 ]; then SELECTED_ADAPTERS=""; return 0; fi

	dim "memeora connects via MCP (recall/remember/context/list) — that part is wired"
	dim "automatically and safely (backed up, never overwritten). Pick your tools:"
	dim "  1) Claude Code   2) Codex   3) Antigravity   4) OpenCode"
	dim "  5) Other MCP tool (print the snippet)   0) none"
	printf '%s ' "${C_BOLD}?${C_RESET} space-separated numbers [1]:" >/dev/tty
	IFS= read -r _sel </dev/tty || _sel=1
	[ -z "$_sel" ] && _sel=1
	for _n in $_sel; do
		case "$_n" in
			1) SELECTED_ADAPTERS="$SELECTED_ADAPTERS claude" ;;
			2) SELECTED_ADAPTERS="$SELECTED_ADAPTERS codex" ;;
			3) SELECTED_ADAPTERS="$SELECTED_ADAPTERS antigravity" ;;
			4) SELECTED_ADAPTERS="$SELECTED_ADAPTERS opencode" ;;
			5) SELECTED_ADAPTERS="$SELECTED_ADAPTERS mcp" ;;
			0) SELECTED_ADAPTERS="" ; break ;;
		esac
	done
}

backup_file() { # backup_file <path>
	[ -f "$1" ] || return 0
	_bak="$1.memeora.bak.$(date +%Y%m%d%H%M%S 2>/dev/null || echo bak)"
	run cp "$1" "$_bak"
	[ "$DRY_RUN" = 1 ] || dim "   backed up $1 -> $_bak"
}

# Merge {"mcpServers":{"memeora":{"command":"memeora-mcp"}}} into a JSON config.
merge_mcp_json() { # merge_mcp_json <file>
	_f=$1
	if ! have jq; then
		warn "jq not found — add this entry to $_f manually:"
		printf '   %s\n' '{"mcpServers":{"memeora":{"command":"memeora-mcp"}}}'
		return 0
	fi
	if [ -f "$_f" ] && jq -e '.mcpServers.memeora' "$_f" >/dev/null 2>&1; then
		ok "already configured in $_f"; return 0
	fi
	[ -f "$_f" ] && backup_file "$_f"
	if [ "$DRY_RUN" = 1 ]; then dim "   would merge memeora MCP entry into $_f"; return 0; fi
	mkdir -p "$(dirname "$_f")" 2>/dev/null || true
	_tmp=$(mktemp)
	if [ -f "$_f" ]; then
		jq '.mcpServers.memeora = {"command":"memeora-mcp"}' "$_f" >"$_tmp"
	else
		jq -n '{mcpServers:{memeora:{command:"memeora-mcp"}}}' >"$_tmp"
	fi
	mv "$_tmp" "$_f"
	ok "wired MCP in $_f"
}

wire_claude() {
	info "Claude Code — MCP"
	if have claude && run claude mcp add -s user memeora -- memeora-mcp 2>/dev/null; then
		ok "registered via 'claude mcp add'"
	else
		merge_mcp_json "$HOME/.claude.json"
	fi
	if [ "$WIRE_HOOKS" = 1 ]; then wire_claude_hooks; else hooks_hint claude; fi
}

wire_claude_hooks() {
	_f="$HOME/.claude/settings.json"
	have jq || { warn "jq needed to merge Claude hooks; skipping (printing steps)"; hooks_hint claude; return 0; }
	mkdir -p "$HOME/.claude" 2>/dev/null || true
	[ -f "$_f" ] || { [ "$DRY_RUN" = 1 ] || printf '{}\n' >"$_f"; }
	backup_file "$_f"
	if [ "$DRY_RUN" = 1 ]; then dim "   would merge SessionStart/Stop/PreCompact hooks into $_f"; return 0; fi
	_tmp=$(mktemp)
	jq '
	  .hooks.SessionStart = ((.hooks.SessionStart // []) + [{"hooks":[{"type":"command","command":"memeora-hook --host claude --event session-start"}]}])
	  | .hooks.Stop = ((.hooks.Stop // []) + [{"hooks":[{"type":"command","command":"memeora-hook --host claude --event stop"}]}])
	  | .hooks.PreCompact = ((.hooks.PreCompact // []) + [{"hooks":[{"type":"command","command":"memeora-hook --host claude --event pre-compact"}]}])
	' "$_f" >"$_tmp" && mv "$_tmp" "$_f"
	ok "wired Claude auto-capture hooks in $_f"
}

wire_codex() {
	info "Codex — MCP"
	_f="$HOME/.codex/config.toml"
	mkdir -p "$HOME/.codex" 2>/dev/null || true
	if [ -f "$_f" ] && grep -q '^\[mcp_servers\.memeora\]' "$_f" 2>/dev/null; then
		ok "already configured in $_f"
	else
		backup_file "$_f"
		if [ "$DRY_RUN" = 1 ]; then
			dim "   would append [mcp_servers.memeora] to $_f"
		else
			{ printf '\n[mcp_servers.memeora]\ncommand = "memeora-mcp"\nenabled = true\n'; } >>"$_f"
			ok "wired MCP in $_f"
		fi
	fi
	if [ "$WIRE_HOOKS" = 1 ]; then
		dim "   Codex hooks: see adapters/codex/hooks/hooks.json — merge into ~/.codex/hooks.json"
	fi
	hooks_hint codex
}

wire_antigravity() {
	info "Antigravity — MCP (manual: its config path varies by IDE vs CLI install)"
	dim "   Add to Antigravity's mcp_config.json (mcpServers):"
	printf '   %s\n' '{"mcpServers":{"memeora":{"command":"memeora-mcp","disabled":false}}}'
	dim "   Plugin bundle + hooks: https://github.com/${REPO}/tree/main/adapters/antigravity"
}

wire_opencode() {
	info "OpenCode — plugin (npm)"
	dim "   Install the plugin and add it to ~/.config/opencode/opencode.jsonc:"
	if have bun; then dim "     bun add -g @memeora/opencode"
	else dim "     npm install -g @memeora/opencode"; fi
	dim '     "plugin": ["@memeora/opencode"]'
	dim "   (or use the MCP entry directly — OpenCode is MCP-capable)"
}

wire_mcp_generic() {
	info "Any MCP-capable tool"
	dim "   Register this stdio server with your tool (key 'memeora'):"
	printf '   %s\n' '{"command":"memeora-mcp"}'
}

hooks_hint() { # hooks_hint <claude|codex>
	[ "$WIRE_HOOKS" = 1 ] && return 0
	dim "   auto-capture hooks (optional): re-run with --wire-hooks, or copy"
	dim "   adapters/$1/hooks/hooks.json into $1's hook config."
}

apply_adapters() {
	[ -z "$SELECTED_ADAPTERS" ] && { info "no adapters selected"; return 0; }
	for _a in $SELECTED_ADAPTERS; do
		case "$_a" in
			claude) wire_claude ;;
			codex) wire_codex ;;
			antigravity) wire_antigravity ;;
			opencode) wire_opencode ;;
			mcp) wire_mcp_generic ;;
			*) warn "unknown adapter '$_a' (known: $KNOWN_ADAPTERS)" ;;
		esac
	done
}

# ----------------------------------------------------------------------------- #
# Step 4 — daemon
# ----------------------------------------------------------------------------- #
maybe_start_daemon() {
	step "Daemon"
	_start=$START_DAEMON
	if [ "$_start" = ask ]; then
		if [ "$INTERACTIVE" = 1 ]; then
			if ask_yes_no "Start memeora-daemon now (in the background)?" 1; then _start=1; else _start=0; fi
		else
			_start=0
		fi
	fi
	if [ "$_start" != 1 ]; then
		info "not starting the daemon. Start it later with:"
		if [ "$ALLOW_DOWNLOAD" = 1 ]; then
			dim "   MEMEORA_ALLOW_MODEL_DOWNLOAD=1 memeora-daemon &"
		else
			dim "   memeora-daemon &"
		fi
		return 0
	fi
	if [ "$ALLOW_DOWNLOAD" != 1 ] && [ "$OFFLINE" = 1 ]; then
		warn "offline mode: the daemon will refuse to start without a model — skipping start."
		return 0
	fi
	if ! have memeora-daemon && [ "$DRY_RUN" != 1 ]; then
		warn "memeora-daemon not on PATH yet — start it from a new shell once PATH is updated."
		return 0
	fi
	_env=""
	[ "$ALLOW_DOWNLOAD" = 1 ] && _env="MEMEORA_ALLOW_MODEL_DOWNLOAD=1 "
	[ "$DASHBOARD" = off ] && _env="${_env}MEMEORA_DASHBOARD_ADDR=off "
	info "starting the daemon in the background (logs: ~/.memeora/daemon.log)"
	printf '%s\n' "${C_DIM}\$ ${_env}nohup memeora-daemon >~/.memeora/daemon.log 2>&1 &${C_RESET}"
	if [ "$DRY_RUN" != 1 ]; then
		mkdir -p "$HOME/.memeora" 2>/dev/null || true
		# shellcheck disable=SC2086
		env ${_env} nohup memeora-daemon >"$HOME/.memeora/daemon.log" 2>&1 &
		sleep 2
		if have memeora && memeora doctor >/dev/null 2>&1; then
			ok "daemon is up — 'memeora doctor' reports healthy"
		else
			warn "daemon started but not reachable yet; check ~/.memeora/daemon.log"
		fi
	fi
	[ "$DASHBOARD" != off ] && dim "   dashboard: http://${DASHBOARD_DEFAULT_ADDR}  (memeora dashboard)"
}

# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
main() {
	printf '%s\n' "${C_BOLD}memeora${C_RESET} — local-first memory for your AI coding tools"
	[ "$DRY_RUN" = 1 ] && warn "dry-run: no changes will be made"
	[ "$INTERACTIVE" = 1 ] || dim "(non-interactive mode — using flags/env + defaults)"

	install_binaries
	choose_model
	choose_adapters
	apply_adapters
	maybe_start_daemon

	step "Done"
	ok "memeora is set up."
	dim "Next:"
	dim "  • memeora doctor                 # check the daemon + model"
	dim "  • memeora dashboard              # open the local graph UI"
	dim "  • restart your coding tool       # so it picks up the new MCP server"
	[ -n "$SELECTED_ADAPTERS" ] && dim "  wired: $SELECTED_ADAPTERS"
}

main
