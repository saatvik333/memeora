//! Guarantees the dashboard asset folder (`dashboard/dist`) exists at compile
//! time so `rust_embed` always builds — including in CI, where the Svelte
//! frontend isn't built (no Node toolchain). A real `pnpm --dir dashboard build`
//! overwrites this placeholder with the actual app, which then gets embedded.

use std::fs;
use std::path::Path;

const PLACEHOLDER: &str = r#"<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>memeora dashboard</title>
<style>body{font:15px system-ui,sans-serif;margin:3rem auto;max-width:42rem;padding:0 1rem;color:#222}code{background:#f0f0f0;padding:.1em .3em;border-radius:3px}</style>
</head>
<body>
<h1>memeora dashboard</h1>
<p>The dashboard UI hasn't been built into this binary. The JSON API is live at
<code>/api/scopes</code>, <code>/api/graph</code>, <code>/api/search</code>,
<code>/api/context</code>, <code>/api/list</code> and the live stream at
<code>/api/events</code>.</p>
<p>To build the graph UI, run:</p>
<pre><code>pnpm --dir dashboard install
pnpm --dir dashboard build</code></pre>
<p>then rebuild the daemon (<code>cargo build --release -p memeora-daemon</code>).</p>
</body>
</html>
"#;

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by cargo");
    let dist = Path::new(&manifest).join("../../dashboard/dist");
    let index = dist.join("index.html");
    if !index.exists() {
        // Best-effort: if we can't create it the rust_embed expansion will report
        // the missing folder, which is the clearer error to surface.
        let _ = fs::create_dir_all(&dist);
        let _ = fs::write(&index, PLACEHOLDER);
    }
    // Re-run if the built assets change, so a fresh `pnpm build` is picked up.
    println!("cargo:rerun-if-changed={}", dist.display());
}
