//! Forgetting & reinforcement dynamics (VISION "Evolves", MemPalace `dynamics.py`).
//!
//! Pure-heuristic, neuroscience-grounded, no LLM:
//! - **Hebbian potentiation** on access — strength grows, capped at [`STRENGTH_MAX`].
//! - **Ebbinghaus decay** — effective strength falls off as `exp(-Δdays / stability)`
//!   from the last access, floored at [`STRENGTH_FLOOR`] so nothing is ever lost.
//! - **Cepeda spacing** — `stability` (durability) grows only on *spaced* repetition
//!   (accesses ≥ [`SPACING_SECS`] apart), so rapid bursts don't build lasting memory.
//!
//! Decay is applied **lazily at read time** ([`decayed_strength`]) rather than by a
//! background tick: the stored `strength` is the potentiation level at last access,
//! and reads discount it by idle time. This fits the sole-writer daemon (no mutation
//! on the read path) and keeps history intact — salience decays, data never vanishes.
//! The potentiation + spacing half is applied on write by the store's reinforce path.
//!
//! Graph **edges** obey the same laws (Phase E): [`decayed_edge_strength`] discounts an
//! edge's strength by idle time so a long-idle relationship contributes less to the recall
//! graph channel, and the store's edge-reinforce path potentiates + Cepeda-spaces edges
//! co-activated in recall ([`EDGE_POTENTIATION_DELTA`] / [`EDGE_STABILITY_DELTA`]).
//!
//! Constants are MemPalace's published starting points — tune against real data.

use crate::store::Memory;

/// Upper bound on stored strength (Hebbian cap) so a hot memory can't grow without limit.
pub const STRENGTH_MAX: f32 = 10.0;
/// Floor on effective strength — decay approaches but never reaches zero (nothing lost).
pub const STRENGTH_FLOOR: f32 = 0.05;
/// Minimum spacing between accesses (seconds) for a reinforcement to build durability.
pub const SPACING_SECS: i64 = 86_400;
/// Stability gained per spaced reinforcement.
pub const STABILITY_DELTA: f32 = 1.0;

/// Strength added to a graph edge each time its endpoints are co-activated in recall
/// (Hebbian potentiation), capped at [`STRENGTH_MAX`]. MemPalace-derived starting point.
pub const EDGE_POTENTIATION_DELTA: f32 = 0.5;
/// Edge durability gained per Cepeda-spaced co-activation (accesses ≥ [`SPACING_SECS`]
/// apart); higher stability ⇒ the edge's strength decays slower, so a repeatedly and
/// *spaced* co-activated relationship endures. MemPalace-derived starting point.
pub const EDGE_STABILITY_DELTA: f32 = 1.0;

/// Decay ratio at/above which a memory still reads as well-retained ("stable").
const FRESH_STABLE_RATIO: f32 = 0.7;
/// Decay ratio at/above which a memory reads as fading but present ("weakening");
/// below it, "stale".
const FRESH_WEAK_RATIO: f32 = 0.25;

const SECONDS_PER_DAY: f32 = 86_400.0;

/// Effective strength of `m` at `now` (Unix seconds): Ebbinghaus decay of the stored
/// strength by idle days, slowed by `stability`, floored at [`STRENGTH_FLOOR`].
pub fn decayed_strength(m: &Memory, now: i64) -> f32 {
    let idle_days = (now - m.last_accessed_at).max(0) as f32 / SECONDS_PER_DAY;
    let stability = m.stability.max(0.001); // guard div-by-zero; higher ⇒ slower decay
    (m.strength * (-idle_days / stability).exp()).max(STRENGTH_FLOOR)
}

/// Effective strength of a graph edge given its stored `strength`, `stability`, and idle
/// time (`now - last_activated`, seconds), floored at [`STRENGTH_FLOOR`]. Used to weight
/// the recall graph channel so a long-idle relationship activates less than a fresh one.
///
/// Edges use a **hyperbolic** decay `strength / (1 + idle_days/stability)` rather than the
/// memory model's exponential: it is the same monotone, deterministic, stability-slowed
/// curve, but expressible in pure SQL arithmetic — so [`crate::store`]'s `graph_search`
/// can fold decay straight into its ranking aggregation without the optional SQLite math
/// extension (`exp()`), which this codebase deliberately avoids. This Rust form is the
/// reference the SQL mirrors exactly.
pub fn decayed_edge_strength(strength: f32, stability: f32, idle_secs: i64) -> f32 {
    let idle_days = idle_secs.max(0) as f32 / SECONDS_PER_DAY;
    let stability = stability.max(0.001); // guard div-by-zero; higher ⇒ slower decay
    (strength / (1.0 + idle_days / stability)).max(STRENGTH_FLOOR)
}

/// Coarse freshness/trend label for `m` at `now` (VISION "freshness trends"), derived
/// purely from in-hand fields — no extra query:
/// - `new` — just learned from a single source.
/// - `strengthening` — corroborated by ≥2 sources and reinforced recently.
/// - `stable` — strength has held up (little decay since last access).
/// - `weakening` — noticeably decayed but still well above the floor.
/// - `stale` — decayed toward the floor (long idle).
///
/// This reads decay (Ebbinghaus) × distinct-source proof together, which is enough to
/// trend a belief at read time.
// ponytail: in-hand signals only (decay ratio + proof_count + age). The fuller VISION
// model compares recent-vs-older distinct-source density from `evidence.occurred_at`;
// add that windowed query if a label needs to distinguish *when* corroboration landed.
pub fn freshness(m: &Memory, now: i64) -> &'static str {
    let recently_reinforced = (now - m.last_accessed_at) < SPACING_SECS;
    if m.proof_count <= 1 && (now - m.created_at) < SPACING_SECS {
        return "new";
    }
    if m.proof_count > 1 && recently_reinforced {
        return "strengthening";
    }
    let ratio = decayed_strength(m, now) / m.strength.max(STRENGTH_FLOOR);
    if ratio >= FRESH_STABLE_RATIO {
        "stable"
    } else if ratio >= FRESH_WEAK_RATIO {
        "weakening"
    } else {
        "stale"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::MemoryKind;

    fn mem(strength: f32, stability: f32, last_accessed_at: i64) -> Memory {
        let mut m = Memory::new("i", "c", MemoryKind::Fact, "t", vec![]);
        m.strength = strength;
        m.stability = stability;
        m.last_accessed_at = last_accessed_at;
        m
    }

    #[test]
    fn no_decay_when_freshly_accessed() {
        let now = 1_000_000_000;
        assert!((decayed_strength(&mem(5.0, 1.0, now), now) - 5.0).abs() < 1e-4);
    }

    #[test]
    fn decays_with_idle_time_but_never_below_floor() {
        let now = 1_000_000_000;
        // 1 day idle, stability 1 ⇒ strength · e^-1 ≈ 0.37.
        let d = decayed_strength(&mem(1.0, 1.0, now - 86_400), now);
        assert!(d < 1.0 && d > 0.3, "{d}");
        // Ancient: floored, not zero — data is never truly lost.
        assert_eq!(
            decayed_strength(&mem(1.0, 1.0, now - 86_400 * 365), now),
            STRENGTH_FLOOR
        );
    }

    #[test]
    fn edge_decay_mirrors_memory_shape() {
        // Fresh (no idle) reads back the stored strength; idle time discounts it; higher
        // stability decays slower; and the floor is never breached.
        assert!((decayed_edge_strength(1.0, 1.0, 0) - 1.0).abs() < 1e-6);
        let day = 86_400;
        let fresh = decayed_edge_strength(1.0, 1.0, 0);
        let idle = decayed_edge_strength(1.0, 1.0, 5 * day);
        assert!(idle < fresh, "idle edge decays: {idle} < {fresh}");
        assert!(
            decayed_edge_strength(1.0, 10.0, 5 * day) > decayed_edge_strength(1.0, 1.0, 5 * day),
            "higher stability decays slower"
        );
        // Ancient edge floors, never zero — the relationship is dimmed, not deleted.
        assert_eq!(
            decayed_edge_strength(1.0, 1.0, day * 365 * 100),
            STRENGTH_FLOOR
        );
    }

    #[test]
    fn higher_stability_decays_slower() {
        let now = 1_000_000_000;
        let idle = now - 86_400 * 5;
        assert!(
            decayed_strength(&mem(1.0, 10.0, idle), now)
                > decayed_strength(&mem(1.0, 1.0, idle), now)
        );
    }

    #[test]
    fn freshness_buckets_classify_each_trend() {
        let now = 1_000_000_000;
        let day = 86_400;
        // Builder: strength, stability, last_accessed, created, proof_count.
        let m = |strength, stability, last, created, proof: u32| {
            let mut m = mem(strength, stability, last);
            m.created_at = created;
            m.proof_count = proof;
            m
        };

        // Just learned, one source.
        assert_eq!(freshness(&m(1.0, 1.0, now, now, 1), now), "new");
        // Corroborated and reinforced recently (created long ago ⇒ not "new").
        assert_eq!(
            freshness(&m(1.0, 1.0, now, now - day * 10, 2), now),
            "strengthening"
        );
        // Old, idle a day but high stability ⇒ barely decayed.
        assert_eq!(
            freshness(&m(1.0, 10.0, now - day, now - day * 10, 2), now),
            "stable"
        );
        // Idle a day, stability 1 ⇒ e^-1 ≈ 0.37 of strength left.
        assert_eq!(
            freshness(&m(1.0, 1.0, now - day, now - day * 10, 1), now),
            "weakening"
        );
        // Idle a month ⇒ decayed to the floor.
        assert_eq!(
            freshness(&m(1.0, 1.0, now - day * 30, now - day * 60, 1), now),
            "stale"
        );
    }
}
