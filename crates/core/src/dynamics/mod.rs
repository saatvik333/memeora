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
//! Constants are MemPalace's published starting points — tune against real data.
// ponytail: edge decay (VISION applies this to graph edges too) is deferred — the
// relationships table has no strength column yet; add it when the graph channel needs it.

use crate::store::Memory;

/// Upper bound on stored strength (Hebbian cap) so a hot memory can't grow without limit.
pub const STRENGTH_MAX: f32 = 10.0;
/// Floor on effective strength — decay approaches but never reaches zero (nothing lost).
pub const STRENGTH_FLOOR: f32 = 0.05;
/// Minimum spacing between accesses (seconds) for a reinforcement to build durability.
pub const SPACING_SECS: i64 = 86_400;
/// Stability gained per spaced reinforcement.
pub const STABILITY_DELTA: f32 = 1.0;

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
