//! `memeora-hook` — the multi-host command-hook adapter binary.
//!
//! A thin wrapper: all logic lives in [`memeora_hook::run`] so every memeora
//! binary ships from the one `memeora` package (a single `dist` installer).

fn main() -> Result<(), Box<dyn std::error::Error>> {
    memeora_hook::run()
}
