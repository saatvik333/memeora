//! Container-tag scoping.
//!
//! Three scopes, matching the design in `docs/ARCHITECTURE.md`:
//! - `memeora_user_{sha16(git email)}`  — cross-project personal memory
//! - `memeora_project_{sha16(path)}`    — private, per-checkout project memory
//! - `repo_{sanitize(repo name)}`       — team-shareable (name-based, not hashed)

use std::fmt::Write;

use sha2::{Digest, Sha256};

/// First 8 bytes of SHA-256, hex-encoded (16 chars).
///
/// Sized for low-cardinality inputs (a git email, a project path). For
/// content-addressed ids over unbounded input, use [`sha32`] instead.
pub fn sha16(input: &str) -> String {
    hex_prefix(input, 8)
}

/// First 16 bytes of SHA-256, hex-encoded (32 chars).
///
/// Used for content-addressed memory ids: 128 bits keeps the birthday-bound
/// collision probability negligible even for very large per-tag stores, where the
/// 64-bit [`sha16`] would not.
pub fn sha32(input: &str) -> String {
    hex_prefix(input, 16)
}

/// Hex-encode the first `bytes` bytes of the SHA-256 of `input`.
fn hex_prefix(input: &str, bytes: usize) -> String {
    let digest = Sha256::digest(input.as_bytes());
    let mut s = String::with_capacity(bytes * 2);
    for b in &digest[..bytes] {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Cross-project personal scope, derived from the user's git email.
pub fn user_tag(git_email: &str) -> String {
    format!("memeora_user_{}", sha16(git_email))
}

/// Private per-checkout project scope, derived from the project root path.
pub fn project_tag(path: &str) -> String {
    format!("memeora_project_{}", sha16(path))
}

/// Team-shareable scope, derived from the repository name (sanitized, not hashed),
/// so teammates working on the same repo converge on the same tag.
pub fn repo_tag(repo_name: &str) -> String {
    let name = sanitize(repo_name);
    if name.is_empty() {
        // An empty or punctuation-only name has no readable form; hash it so distinct
        // unusual names stay distinct instead of all collapsing to the bare "repo_"
        // bucket and silently merging unrelated repos' memory.
        format!("repo_{}", sha16(repo_name))
    } else {
        format!("repo_{name}")
    }
}

/// Lowercase, collapse non-alphanumeric runs to single `_`, trim leading/trailing `_`.
fn sanitize(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut prev_underscore = false;
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            prev_underscore = false;
        } else if !prev_underscore {
            out.push('_');
            prev_underscore = true;
        }
    }
    out.trim_matches('_').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha16_is_deterministic_and_16_chars() {
        let a = sha16("hello@example.com");
        let b = sha16("hello@example.com");
        assert_eq!(a, b);
        assert_eq!(a.len(), 16);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(sha16("a@b.com"), sha16("c@d.com"));
    }

    #[test]
    fn sha32_is_deterministic_and_32_chars() {
        let a = sha32("some memory content");
        assert_eq!(a, sha32("some memory content"));
        assert_eq!(a.len(), 32);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(sha32("content a"), sha32("content b"));
        // Wider than sha16 (no shared prefix length assumption beyond the first 16).
        assert!(sha32("x").starts_with(&sha16("x")));
    }

    #[test]
    fn tag_formats() {
        assert!(user_tag("x@y.com").starts_with("memeora_user_"));
        assert!(project_tag("/home/me/proj").starts_with("memeora_project_"));
    }

    #[test]
    fn repo_tag_sanitizes() {
        assert_eq!(repo_tag("My Repo!"), "repo_my_repo");
        assert_eq!(repo_tag("memeora"), "repo_memeora");
        assert_eq!(repo_tag("a--b__c"), "repo_a_b_c");
    }

    #[test]
    fn repo_tag_punctuation_only_names_do_not_collide() {
        // Names with no alphanumerics used to all sanitize to "" → the same "repo_"
        // bucket. They are now hashed, so they stay distinct.
        assert_ne!(repo_tag("!!!"), repo_tag("???"));
        assert!(repo_tag("!!!").starts_with("repo_"));
        assert_ne!(repo_tag("!!!"), "repo_");
        assert_eq!(repo_tag("@@@"), repo_tag("@@@"), "still deterministic");
    }
}
