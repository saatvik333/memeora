//! `memeora-daemon` — the long-lived engine process (models + DB + IPC server).
//!
//! A thin wrapper: all logic lives in [`memeora_daemon::run`] so every memeora
//! binary ships from the one `memeora` package (a single `dist` installer).

fn main() -> Result<(), Box<dyn std::error::Error>> {
    memeora_daemon::run()
}
