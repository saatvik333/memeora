//! Heuristic valid-time parsing (VISION "Evolves" — bi-temporal occurred-time).
//!
//! Pure and offline: pull a coarse *occurred* interval out of text — "yesterday",
//! "3 days ago", "last week", an ISO `YYYY-MM-DD` — so a memory can record *when the
//! event happened* (valid-time) distinct from `created_at` (when it was learned).
//! Conservative: recognise a small, high-signal set and return `None` for everything
//! else, so the `occurred_*` columns stay unset (costing nothing) unless a real cue
//! is present. The same parser resolves a recall query's date window (the temporal
//! channel), so stored and queried time are read off one implementation.
// ponytail: naive marker / "<n> <unit> ago" / ISO parser over UTC days — no timezone
// or locale handling, which is fine for coarse time-bucketed recall. The opt-in LLM
// tier is the upgrade path for nuanced temporal expressions.

const DAY: i64 = 86_400;

/// A parsed occurred interval `(start, end)` in Unix seconds, mapping directly onto a
/// memory's `(occurred_start, occurred_end)`. `end` is `None` for a bare point in
/// time and `Some` for a bounded range ("last week", a calendar day).
pub type Span = (i64, Option<i64>);

/// Floor `t` to the start of its UTC day.
fn day_start(t: i64) -> i64 {
    t - t.rem_euclid(DAY)
}

/// Parse a coarse occurred interval from `text` relative to `now` (Unix seconds), or
/// `None` when no recognised temporal cue is present. An explicit ISO date wins over a
/// relative phrase (it is the most specific signal).
pub fn parse(text: &str, now: i64) -> Option<Span> {
    let lower = text.to_ascii_lowercase();

    // Most specific: an explicit calendar date.
    if let Some(day) = first_iso_date(&lower) {
        return Some((day, Some(day + DAY)));
    }
    // "<n> <day|week|month|hour>(s) ago".
    if let Some(span) = n_units_ago(&lower, now) {
        return Some(span);
    }

    // Coarse relative anchors. Order matters: a longer phrase that *contains* a
    // shorter one ("day before yesterday" ⊃ "yesterday") must be tested first.
    let today = day_start(now);
    if lower.contains("day before yesterday") {
        let d = today - 2 * DAY;
        return Some((d, Some(d + DAY)));
    }
    if lower.contains("yesterday") || lower.contains("last night") {
        let d = today - DAY;
        return Some((d, Some(d + DAY)));
    }
    if lower.contains("last week") {
        return Some((today - 7 * DAY, Some(today + DAY)));
    }
    if lower.contains("last month") {
        return Some((today - 30 * DAY, Some(today + DAY)));
    }
    if [
        "today",
        "this morning",
        "this afternoon",
        "tonight",
        "right now",
    ]
    .iter()
    .any(|m| lower.contains(m))
    {
        return Some((today, Some(today + DAY)));
    }
    None
}

/// Parse a "<n> <unit>(s) ago" phrase into a one-day interval `<n> units` before now.
fn n_units_ago(lower: &str, now: i64) -> Option<Span> {
    let idx = lower.find(" ago")?;
    // The two whitespace tokens immediately before " ago" are "<n> <unit>".
    let mut it = lower[..idx].split_whitespace().rev();
    let unit = it.next()?;
    let n: i64 = it.next()?.parse().ok()?;
    if n <= 0 {
        return None;
    }
    let secs = match unit.trim_end_matches('s') {
        "day" => n * DAY,
        "week" => n * 7 * DAY,
        "month" => n * 30 * DAY,
        "hour" => n * 3600,
        _ => return None,
    };
    let start = day_start(now - secs);
    Some((start, Some(start + DAY)))
}

/// The first `YYYY-MM-DD` (or `YYYY/MM/DD`) in `text` as that day's UTC start, or
/// `None`. Ignores a date embedded in a longer digit run (e.g. version strings).
fn first_iso_date(text: &str) -> Option<i64> {
    let bytes = text.as_bytes();
    for i in 0..bytes.len() {
        // Don't start a match in the middle of a number (avoid "12025-01-01").
        if i > 0 && bytes[i - 1].is_ascii_digit() {
            continue;
        }
        if let Some(day) = iso_at(&text[i..]) {
            return Some(day);
        }
    }
    None
}

/// Parse a leading `YYYY[-/]MM[-/]DD` from `s` into that day's UTC start (Unix s).
fn iso_at(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    if b.len() < 10 {
        return None;
    }
    let digit = |c: u8| c.is_ascii_digit();
    let sep = |c: u8| c == b'-' || c == b'/';
    let shaped = digit(b[0])
        && digit(b[1])
        && digit(b[2])
        && digit(b[3])
        && sep(b[4])
        && digit(b[5])
        && digit(b[6])
        && sep(b[7])
        && digit(b[8])
        && digit(b[9]);
    // Reject a trailing digit so "2026-06-211" doesn't read as a valid date.
    if !shaped || b.get(10).is_some_and(|c| c.is_ascii_digit()) {
        return None;
    }
    let year: i64 = s[0..4].parse().ok()?;
    let month: i64 = s[5..7].parse().ok()?;
    let day: i64 = s[8..10].parse().ok()?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    Some(days_from_civil(year, month, day) * DAY)
}

/// Days since the Unix epoch for a proleptic-Gregorian `Y-M-D` (Howard Hinnant's
/// branch-free `days_from_civil`). Avoids a calendar dependency for one date math.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400; // [0, 399]
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe - 719468 // 719468 = days from 0000-03-01 to 1970-01-01
}

#[cfg(test)]
mod tests {
    use super::*;

    // A fixed "now": 2026-06-21 12:00:00 UTC (well after that day's start).
    const NOW: i64 = 1_781_000_000;

    #[test]
    fn epoch_and_known_dates() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(1970, 1, 2), 1);
        assert_eq!(days_from_civil(1969, 12, 31), -1);
        // 2000-03-01 is 11017 days after the epoch (a stable Hinnant check point).
        assert_eq!(days_from_civil(2000, 3, 1), 11017);
    }

    #[test]
    fn iso_date_wins_and_is_day_floored() {
        let (start, end) = parse("we shipped it on 2026-06-12, big day", NOW).unwrap();
        assert_eq!(start, days_from_civil(2026, 6, 12) * DAY);
        assert_eq!(end, Some(start + DAY));
        assert_eq!(start % DAY, 0, "start is a UTC day boundary");
        // Slash form parses identically.
        assert_eq!(parse("on 2026/06/12", NOW).unwrap().0, start);
    }

    #[test]
    fn iso_ignored_inside_longer_number() {
        // A date glued to extra digits is not a date.
        assert!(parse("build 12026-06-12x", NOW).is_none());
    }

    #[test]
    fn relative_anchors() {
        let today = day_start(NOW);
        assert_eq!(
            parse("I did it today", NOW),
            Some((today, Some(today + DAY)))
        );
        assert_eq!(
            parse("met Alex yesterday", NOW),
            Some((today - DAY, Some(today)))
        );
        assert_eq!(
            parse("the day before yesterday we met", NOW),
            Some((today - 2 * DAY, Some(today - DAY))),
            "longer phrase must beat the contained 'yesterday'"
        );
        assert_eq!(
            parse("shipped last week", NOW),
            Some((today - 7 * DAY, Some(today + DAY)))
        );
    }

    #[test]
    fn n_units_ago_parses() {
        let today = day_start(NOW);
        assert_eq!(
            parse("we decided 3 days ago", NOW),
            Some((today - 3 * DAY, Some(today - 2 * DAY)))
        );
        assert_eq!(parse("2 weeks ago", NOW).unwrap().0, today - 14 * DAY);
        // A non-numeric or zero count doesn't parse.
        assert!(parse("a while ago", NOW).is_none());
        assert!(parse("0 days ago", NOW).is_none());
    }

    #[test]
    fn no_temporal_cue_is_none() {
        assert!(parse("I prefer dark mode", NOW).is_none());
        assert!(parse("we use SQLite for storage", NOW).is_none());
    }
}
