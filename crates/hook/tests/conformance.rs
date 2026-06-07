//! Adapter conformance kit.
//!
//! Replays real per-host stdin payloads (`tests/fixtures/<host>/<case>.json`)
//! through the descriptor-driven hook logic and asserts the host contract: the
//! resolved scope directory, whether injection fires, the injection JSON shape,
//! the transcript path, and the capture ack.
//!
//! A third-party host self-verifies by adding its descriptor (`adapters/
//! _descriptors/<host>.toml`) and a few fixtures here — the same runner validates
//! them, so "does my adapter behave?" has a runnable answer.

use std::fs;
use std::path::{Path, PathBuf};

use memeora_core::container_tag::project_tag;
use memeora_hook::{
    capture_ack, descriptor, render_inject, resolve_scope, should_inject, transcript_path,
};
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct Fixture {
    /// Built-in host this fixture targets (must resolve via `descriptor::builtin`).
    host: String,
    /// Lifecycle event (documentation only; the assertions drive behavior).
    #[allow(dead_code)]
    event: String,
    /// The raw stdin payload the host would send.
    payload: Value,
    /// Expected outcomes (each optional — assert only what's specified).
    expect: Expect,
}

#[derive(Deserialize)]
struct Expect {
    /// Directory the scope should resolve to (compared via `project_tag`).
    scope_dir: Option<String>,
    /// Whether an inject event should fire.
    should_inject: Option<bool>,
    /// Substring the rendered injection JSON must contain (e.g. the style key).
    inject_contains: Option<String>,
    /// Transcript path the descriptor should extract.
    transcript_path: Option<String>,
    /// The capture ack, as parsed JSON.
    ack_json: Option<Value>,
}

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn check(path: &Path) {
    let raw = fs::read_to_string(path).unwrap();
    let fx: Fixture = serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("{}: bad fixture: {e}", path.display()));
    let desc = descriptor::builtin(&fx.host)
        .unwrap_or_else(|| panic!("{}: unknown host {:?}", path.display(), fx.host));

    if let Some(dir) = &fx.expect.scope_dir {
        assert_eq!(
            resolve_scope(&desc, &fx.payload),
            project_tag(dir),
            "{}: scope",
            path.display()
        );
    }
    if let Some(want) = fx.expect.should_inject {
        assert_eq!(
            should_inject(&desc, &fx.payload),
            want,
            "{}: should_inject",
            path.display()
        );
    }
    if let Some(sub) = &fx.expect.inject_contains {
        let out = render_inject(&desc, "PROFILE_TEXT");
        // Always valid JSON, carries the profile, and uses the host's style key.
        let _: Value = serde_json::from_str(&out)
            .unwrap_or_else(|e| panic!("{}: inject not JSON: {e}", path.display()));
        assert!(
            out.contains("PROFILE_TEXT"),
            "{}: inject missing text",
            path.display()
        );
        assert!(
            out.contains(sub.as_str()),
            "{}: inject missing {sub:?} in {out}",
            path.display()
        );
    }
    if let Some(tp) = &fx.expect.transcript_path {
        assert_eq!(
            transcript_path(&desc, &fx.payload).as_deref(),
            Some(tp.as_str()),
            "{}: transcript_path",
            path.display()
        );
    }
    if let Some(ack) = &fx.expect.ack_json {
        let got =
            capture_ack(&desc).unwrap_or_else(|| panic!("{}: expected an ack", path.display()));
        let got_json: Value = serde_json::from_str(&got)
            .unwrap_or_else(|e| panic!("{}: ack not JSON: {e}", path.display()));
        assert_eq!(&got_json, ack, "{}: ack", path.display());
    }
}

#[test]
fn all_fixtures_conform() {
    let dir = fixtures_dir();
    let mut count = 0;
    for host_entry in fs::read_dir(&dir).expect("fixtures dir exists") {
        let host_dir = host_entry.unwrap().path();
        if !host_dir.is_dir() {
            continue;
        }
        for case in fs::read_dir(&host_dir).unwrap() {
            let p = case.unwrap().path();
            if p.extension().and_then(|e| e.to_str()) == Some("json") {
                check(&p);
                count += 1;
            }
        }
    }
    assert!(count >= 5, "expected several fixtures, found {count}");
}
