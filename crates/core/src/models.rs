//! Model-asset integrity: SHA-256 manifests for the offline model bundle and
//! first-run download verification.
//!
//! memeora does **not** embed the (tens-of-MB) ONNX weights in the binary (Risk F
//! in `docs/ARCHITECTURE.md`): they are downloaded on first run or shipped as an
//! offline bundle. Either way the bytes are untrusted until verified, so the bundle
//! carries a `SHA256SUMS` manifest and memeora checks the on-disk files against it.
//!
//! The manifest uses the ubiquitous `sha256sum` text format — `<64-hex>␠␠<relpath>`
//! per line — so it interoperates with the standard `sha256sum -c` tool and needs
//! no bespoke parser dependency.

use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Conventional manifest filename memeora reads/writes inside a model directory.
pub const MANIFEST_NAME: &str = "SHA256SUMS";

/// Streaming SHA-256 of a file, hashed in chunks so a large model file is never
/// loaded into memory all at once.
pub fn sha256_file(path: &Path) -> io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Resolve memeora's model cache directory:
/// `$MEMEORA_MODELS_DIR`, else `$MEMEORA_HOME/models`, else `~/.memeora/models`.
///
/// `MEMEORA_MODELS_DIR` lets an offline bundle be pre-placed and used as-is; the
/// daemon points its embedder cache here so the resolution is identical everywhere.
pub fn resolve_dir() -> PathBuf {
    resolve_dir_from(
        std::env::var_os("MEMEORA_MODELS_DIR").map(PathBuf::from),
        std::env::var_os("MEMEORA_HOME").map(PathBuf::from),
        dirs::home_dir(),
    )
}

/// Pure precedence logic behind [`resolve_dir`] (separated so it's testable without
/// mutating process-global env): `models_dir` > `home/models` > `~/.memeora/models`.
fn resolve_dir_from(
    models_dir: Option<PathBuf>,
    home: Option<PathBuf>,
    home_dir: Option<PathBuf>,
) -> PathBuf {
    if let Some(dir) = models_dir {
        return dir;
    }
    if let Some(home) = home {
        return home.join("models");
    }
    home_dir
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".memeora")
        .join("models")
}

/// One file's verification outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssetStatus {
    /// The on-disk file matches the manifest hash.
    Ok,
    /// The file exists but its hash differs (corruption or tampering).
    Mismatch { expected: String, actual: String },
    /// The manifest lists the file but it is absent from the directory.
    Missing,
}

/// A single manifest entry's path and verification status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetResult {
    pub path: String,
    pub status: AssetStatus,
}

/// The result of verifying a directory against a manifest.
#[derive(Debug, Default, Clone)]
pub struct VerifyReport {
    pub results: Vec<AssetResult>,
}

impl VerifyReport {
    /// Whether every listed asset verified.
    pub fn ok(&self) -> bool {
        self.results.iter().all(|r| r.status == AssetStatus::Ok)
    }

    /// `(ok, mismatched, missing)` counts.
    pub fn counts(&self) -> (usize, usize, usize) {
        let mut ok = 0;
        let mut mismatch = 0;
        let mut missing = 0;
        for r in &self.results {
            match r.status {
                AssetStatus::Ok => ok += 1,
                AssetStatus::Mismatch { .. } => mismatch += 1,
                AssetStatus::Missing => missing += 1,
            }
        }
        (ok, mismatch, missing)
    }
}

/// Parse a `SHA256SUMS` manifest into `(relative_path, expected_hex)` pairs.
///
/// Tolerant of the `sha256sum` variants: the binary-mode `*` marker before the
/// path and either two-space or single-space separators; blank lines are skipped.
pub fn parse_manifest(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((hash, rest)) = line.split_once(char::is_whitespace) else {
            continue;
        };
        let path = rest.trim_start().trim_start_matches('*').trim();
        if hash.len() == 64 && !path.is_empty() {
            out.push((path.to_string(), hash.to_ascii_lowercase()));
        }
    }
    out
}

/// Verify every entry of `manifest_text` against files under `dir`.
pub fn verify(dir: &Path, manifest_text: &str) -> io::Result<VerifyReport> {
    let mut report = VerifyReport::default();
    for (rel, expected) in parse_manifest(manifest_text) {
        let path = dir.join(&rel);
        let status = if !path.exists() {
            AssetStatus::Missing
        } else {
            let actual = sha256_file(&path)?;
            if actual == expected {
                AssetStatus::Ok
            } else {
                AssetStatus::Mismatch { expected, actual }
            }
        };
        report.results.push(AssetResult { path: rel, status });
    }
    Ok(report)
}

/// Read `dir/SHA256SUMS` and verify it. `Ok(None)` if no manifest is present
/// (so callers can treat "no manifest" distinctly from "verification failed").
pub fn verify_dir(dir: &Path) -> io::Result<Option<VerifyReport>> {
    let manifest = dir.join(MANIFEST_NAME);
    if !manifest.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(&manifest)?;
    Ok(Some(verify(dir, &text)?))
}

/// Generate a `SHA256SUMS` manifest covering every regular file under `dir`
/// (recursively), with paths relative to `dir` and sorted for determinism. An
/// existing top-level manifest is excluded so re-running is idempotent.
///
/// Used by the release tooling to stamp an offline model bundle with checksums.
pub fn generate_manifest(dir: &Path) -> io::Result<String> {
    let mut files = Vec::new();
    collect_files(dir, dir, &mut files)?;
    files.sort();
    let mut out = String::new();
    for rel in files {
        if rel == MANIFEST_NAME {
            continue;
        }
        let hash = sha256_file(&dir.join(&rel))?;
        out.push_str(&hash);
        out.push_str("  ");
        out.push_str(&rel);
        out.push('\n');
    }
    Ok(out)
}

/// Recursively collect file paths under `dir`, relative to `root`, using `/` as
/// the separator so manifests are portable across platforms.
fn collect_files(root: &Path, dir: &Path, out: &mut Vec<String>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let ty = entry.file_type()?;
        if ty.is_dir() {
            collect_files(root, &path, out)?;
        } else if ty.is_file()
            && let Ok(rel) = path.strip_prefix(root)
        {
            let rel = rel
                .components()
                .map(|c| c.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("/");
            out.push(rel);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A clean, uniquely-named scratch dir under the system temp dir.
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("memeora-models-test-{name}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn generate_then_verify_roundtrips() {
        let dir = scratch("roundtrip");
        fs::write(dir.join("model.onnx"), b"fake onnx weights").unwrap();
        fs::create_dir_all(dir.join("sub")).unwrap();
        fs::write(dir.join("sub/tokenizer.json"), b"{\"vocab\":1}").unwrap();

        let manifest = generate_manifest(&dir).unwrap();
        // Paths are relative and use `/`; the manifest itself isn't listed.
        assert!(manifest.contains("  model.onnx\n"));
        assert!(manifest.contains("  sub/tokenizer.json\n"));
        assert!(!manifest.contains(MANIFEST_NAME));

        let report = verify(&dir, &manifest).unwrap();
        assert!(report.ok());
        assert_eq!(report.counts(), (2, 0, 0));

        // Persisting it and re-running `generate` is idempotent (manifest excluded).
        fs::write(dir.join(MANIFEST_NAME), &manifest).unwrap();
        assert_eq!(generate_manifest(&dir).unwrap(), manifest);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn detects_corruption_and_missing_files() {
        let dir = scratch("tamper");
        fs::write(dir.join("a.bin"), b"original").unwrap();
        fs::write(dir.join("b.bin"), b"present").unwrap();
        let manifest = generate_manifest(&dir).unwrap();

        // Tamper with one file and delete another.
        fs::write(dir.join("a.bin"), b"tampered!").unwrap();
        fs::remove_file(dir.join("b.bin")).unwrap();

        let report = verify(&dir, &manifest).unwrap();
        assert!(!report.ok());
        assert_eq!(report.counts(), (0, 1, 1));
        assert!(
            report
                .results
                .iter()
                .any(|r| r.path == "a.bin" && matches!(r.status, AssetStatus::Mismatch { .. }))
        );
        assert!(
            report
                .results
                .iter()
                .any(|r| r.path == "b.bin" && r.status == AssetStatus::Missing)
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_dir_distinguishes_absent_manifest() {
        let dir = scratch("nomanifest");
        fs::write(dir.join("x.bin"), b"x").unwrap();
        assert!(verify_dir(&dir).unwrap().is_none());
        let manifest = generate_manifest(&dir).unwrap();
        fs::write(dir.join(MANIFEST_NAME), &manifest).unwrap();
        assert!(verify_dir(&dir).unwrap().unwrap().ok());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_tolerates_sha256sum_variants() {
        let text = "\
# a comment\n\
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  spaced.bin\n\
ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad *binary.bin\n\
\n\
shorthash  ignored.bin\n";
        let parsed = parse_manifest(text);
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].0, "spaced.bin");
        assert_eq!(parsed[1].0, "binary.bin");
    }

    #[test]
    fn resolve_dir_precedence() {
        // MEMEORA_MODELS_DIR wins outright.
        assert_eq!(
            resolve_dir_from(
                Some(PathBuf::from("/bundle")),
                Some(PathBuf::from("/home/x/.memeora")),
                Some(PathBuf::from("/home/x")),
            ),
            PathBuf::from("/bundle")
        );
        // Else MEMEORA_HOME/models.
        assert_eq!(
            resolve_dir_from(
                None,
                Some(PathBuf::from("/data/m")),
                Some(PathBuf::from("/h"))
            ),
            PathBuf::from("/data/m/models")
        );
        // Else ~/.memeora/models.
        assert_eq!(
            resolve_dir_from(None, None, Some(PathBuf::from("/home/u"))),
            PathBuf::from("/home/u/.memeora/models")
        );
    }
}
