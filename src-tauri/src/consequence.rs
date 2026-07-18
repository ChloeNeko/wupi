//! Consequence queue data model — Phase 2 stub (Games app Seam 4).
//!
//! Models the delayed/probabilistic/chained consequence system borrowed from
//! UIE's `consequenceEngine.js` + `causalityEngine.js` (see
//! docs/games-app-design.md §4). The MVP ships types only — Phase 2 wires
//! the queue into the game clock's tick (the master tick that will also
//! drive schedules, ambient events, gossip spread).
//!
//! **Nothing here is wired into the game loop yet.** The types are
//! forward-compat scaffolding so Phase 2's queue + ripple engine have a
//! place to land.

use crate::schema::SchemaDelta;

/// One queued consequence — a delayed, probabilistic effect triggered by an
/// in-world event. UIE pattern (`consequenceEngine.js:34`): each consequence
/// has a trigger, a delay window (sampled when scheduled), a probability of
/// actually firing, and a ripple of effects.
#[derive(Debug, Clone)]
pub struct Consequence {
    /// The event id that schedules this consequence (e.g. "crime_witnessed",
    /// "promise_broken", "lie_told"). Matching against recent world events
    /// is Phase 2's job.
    pub trigger: String,
    /// Delay window in in-game minutes. When scheduled, a delay is sampled
    /// uniformly from this Vec (UIE supports multiple candidate delays;
    /// single-element Vecs are fixed-delay). At least one entry required.
    pub delay_minutes: Vec<u32>,
    /// 0.0-1.0. Probability the consequence actually fires when its delay
    /// elapses. Below 1.0 = probabilistic; 0.0 = never (drop on schedule).
    pub probability: f32,
    /// The cross-domain ripple effects (UIE `causalityEngine.js:93`). Each
    /// fires if the probability check passes. Multiple rules = chained.
    pub ripple: Vec<RippleRule>,
}

/// One ripple effect — a cross-domain state change caused by a consequence
/// firing. UIE pattern: e.g. `crime_witnessed` → reputation drop, gossip
/// spread, NPC suspicion bump, journal quest update.
#[derive(Debug, Clone)]
pub struct RippleRule {
    /// A short id for this rule (logging + the debug panel).
    pub event: String,
    /// The state delta to apply if this rule fires.
    pub effect: SchemaDelta,
}

/// The (Phase 2) consequence queue. Holds scheduled consequences + drains
/// them on the game clock's tick, rolling probability + firing ripple rules
/// whose delays have elapsed. UIE uses a 15s wall-clock interval; we'll
/// drive it off the in-game time advance (so pausing the game pauses the
/// queue — a real-time queue would fire while the player is reading).
///
/// Phase 2 will add: schedule(), drain(minutes_advanced), probability rolls,
/// logging for the debug panel.
#[derive(Debug, Clone, Default)]
pub struct ConsequenceQueue {
    /// Scheduled-but-not-yet-fired consequences, in scheduled-at order.
    /// Each entry carries the absolute in-game minute it fires at (set when
    /// scheduled = now + sampled delay).
    pub scheduled: Vec<ScheduledConsequence>,
}

/// A consequence with a concrete fire-at time (post-sampling).
#[derive(Debug, Clone)]
pub struct ScheduledConsequence {
    /// The originating consequence template.
    pub template: Consequence,
    /// Absolute in-game minute this fires at.
    pub fire_at_minute: u64,
    /// When it was scheduled (for the debug panel's "pending" view).
    pub scheduled_at_minute: u64,
}

impl ConsequenceQueue {
    /// Phase 2 will implement: schedule a consequence template by sampling
    /// its delay + pushing a ScheduledConsequence. Stub for now.
    pub fn schedule(&mut self, _template: Consequence, _now_minute: u64) {
        // TODO Phase 2: sample delay from template.delay_minutes, roll nothing
        // here (the probability roll happens at fire time, not schedule time).
    }

    /// Phase 2 will implement: drain the queue, firing any consequence whose
    /// fire_at_minute <= now. Returns the deltas to apply. Stub for now.
    pub fn drain(&mut self, _now_minute: u64) -> Vec<SchemaDelta> {
        // TODO Phase 2: filter scheduled by fire_at <= now, roll probability,
        // collect ripple effects' deltas, drop fired entries.
        Vec::new()
    }
}
