//! memeora CLI (scaffold).
//!
//! User-facing entrypoint: `install <host>`, `serve`, `doctor`, `index`, `dashboard`,
//! `adapter new`. Thin client over the daemon. Subcommands land per the build order.

fn main() {
    println!(
        "memeora {} (client protocol v{})",
        env!("CARGO_PKG_VERSION"),
        memeora_proto::PROTOCOL_VERSION,
    );
}
