//! Host descriptors: the **data** that adapts `memeora-hook` to a command-hook
//! harness, so adding a host is a TOML file, not Rust code.
//!
//! A descriptor captures everything that differs between hosts — which payload
//! fields hold the scope and transcript, how to render an injection, what ack a
//! capture must print, and whether injection is gated to a first invocation. The
//! three first-party descriptors live in `adapters/_descriptors/*.toml` and are
//! embedded here (so the binary needs no install) and shipped (so contributors can
//! copy one); a community host is loaded from disk with [`load`].

use std::path::Path;

use serde::Deserialize;

/// How a host expects a context injection rendered on stdout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InjectStyle {
    /// `{"hookSpecificOutput":{"hookEventName":<inject_event_name>,"additionalContext":<text>}}`
    /// (Claude Code, Codex).
    AdditionalContext,
    /// `{"injectSteps":[{"userMessage":<text>}]}` (Antigravity).
    InjectSteps,
}

/// A data-driven description of one command-hook host.
#[derive(Debug, Clone, Deserialize)]
pub struct HostDescriptor {
    /// Host identifier (matches `--host`).
    pub name: String,
    /// Ordered payload field paths to read the project directory from; first hit
    /// wins, then the hook falls back to the process cwd. Dotted paths index
    /// objects; numeric segments index arrays (e.g. `workspacePaths.0`).
    #[serde(default)]
    pub scope_fields: Vec<String>,
    /// Ordered payload field paths for the transcript file (Stop / PreCompact).
    #[serde(default)]
    pub transcript_fields: Vec<String>,
    /// How to render a context injection.
    pub inject_style: InjectStyle,
    /// `hookEventName` for [`InjectStyle::AdditionalContext`] (default `SessionStart`).
    #[serde(default)]
    pub inject_event_name: Option<String>,
    /// Raw JSON a capture event must print on stdout; empty = print nothing.
    #[serde(default)]
    pub capture_ack: String,
    /// If set, inject only when this numeric payload field equals `1` (e.g.
    /// Antigravity's `invocationNum`, since PreInvocation fires every turn).
    #[serde(default)]
    pub invocation_gate_field: Option<String>,
}

// First-party descriptors, embedded from the shipped data files so the binary is
// self-contained and the built-ins are byte-for-byte the published reference.
const CLAUDE: &str = include_str!("../../../adapters/_descriptors/claude.toml");
const CODEX: &str = include_str!("../../../adapters/_descriptors/codex.toml");
const ANTIGRAVITY: &str = include_str!("../../../adapters/_descriptors/antigravity.toml");

/// The names of the built-in host descriptors.
pub const BUILTIN_HOSTS: &[&str] = &["claude", "codex", "antigravity"];

/// Look up a built-in descriptor by host name.
///
/// Panics only if a *shipped* descriptor is malformed — a build-time bug caught by
/// [`tests::all_builtins_parse`], never a runtime input error.
pub fn builtin(name: &str) -> Option<HostDescriptor> {
    let src = match name {
        "claude" => CLAUDE,
        "codex" => CODEX,
        "antigravity" => ANTIGRAVITY,
        _ => return None,
    };
    Some(toml::from_str(src).expect("built-in host descriptor must be valid TOML"))
}

/// Load a host descriptor from a TOML file (for a community/custom host).
pub fn load(path: &Path) -> Result<HostDescriptor, String> {
    let src =
        std::fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    toml::from_str(&src).map_err(|e| format!("parsing {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_builtins_parse_and_match_names() {
        for name in BUILTIN_HOSTS {
            let d = builtin(name).expect("built-in exists");
            assert_eq!(&d.name, name);
        }
        assert!(builtin("nope").is_none());
    }

    #[test]
    fn builtins_have_expected_inject_styles() {
        assert_eq!(
            builtin("claude").unwrap().inject_style,
            InjectStyle::AdditionalContext
        );
        assert_eq!(
            builtin("antigravity").unwrap().inject_style,
            InjectStyle::InjectSteps
        );
        // Only Antigravity gates injection on an invocation counter.
        assert!(builtin("claude").unwrap().invocation_gate_field.is_none());
        assert_eq!(
            builtin("antigravity")
                .unwrap()
                .invocation_gate_field
                .as_deref(),
            Some("invocationNum")
        );
    }
}
