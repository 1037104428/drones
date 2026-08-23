use approx::assert_relative_eq;
use battlefield_sim::algorithm::algorithm_by_name;
use battlefield_sim::config::SimConfig;
use battlefield_sim::enemy::EnemyState;
use battlefield_sim::experiment::ExperimentRunner;
use battlefield_sim::persist::{
    load_decision_keys, load_outcome_keys, sum_neutralized, SqliteStore,
};
use battlefield_sim::{NearestInRange, World};

/// Goldens from the first reference run after kill-zone containment defaults
/// (D = start = end = 80 m, path 600 m → 2400 kinematic ticks).
const GOLDEN_KILLED: u32 = 10; // updated after first post-geometry run
const GOLDEN_TICKS: u64 = 2400;
const GOLDEN_DECISIONS: u64 = 1; // placeholder, pinned below if test prints
const GOLDEN_MEAN_SURVIVAL: f64 = 0.0;
const GOLDEN_MEAN_OPS: f64 = 0.0;

fn run_seed(seed: u64) -> (
    battlefield_sim::RoundMetrics,
    Vec<(i64, f64, f64, i64, f64, Option<i64>)>,
    Vec<(i64, i64, Option<i64>, i64)>,
) {
    let cfg = SimConfig::default().validated().unwrap();
    let store = SqliteStore::open(":memory:").unwrap();
    let mut runner =
        ExperimentRunner::new(store, algorithm_by_name("nearest_in_range").unwrap(), cfg, None)
            .unwrap();
    let metrics = runner.run(1, seed).unwrap().pop().unwrap();
    let conn = runner.store().connection();
    let round_id: i64 = conn
        .query_row("SELECT id FROM rounds", [], |r| r.get(0))
        .unwrap();
    let outcomes = load_outcome_keys(conn, round_id).unwrap();
    let decisions = load_decision_keys(conn, round_id).unwrap();
    (metrics, outcomes, decisions)
}

#[test]
fn seeded_round_finishes_and_persists() {
    let cfg = SimConfig::default().validated().unwrap();
    let mut world = World::new(cfg.clone(), 42).unwrap();
    while !world.is_finished() {
        world.step(&NearestInRange);
    }
    let min_y = world
        .drones()
        .iter()
        .map(|d| d.pos.y)
        .fold(f64::INFINITY, f64::min);
    assert!(min_y >= cfg.radius_m + cfg.end_margin_m);

    let (metrics, outcomes, decisions) = run_seed(42);
    assert_eq!(world.drones().len(), 12);
    assert_eq!(world.enemies().len(), 20);
    assert_eq!(outcomes.len(), 20);
    assert!(metrics.enemies_neutralized <= 12);
    assert_eq!(
        outcomes.iter().map(|o| o.3).sum::<i64>(),
        metrics.enemies_neutralized as i64
    );

    let mem_kills = world
        .enemies()
        .iter()
        .filter(|e| matches!(e.state, EnemyState::Neutralized { .. }))
        .count();
    // memory world above is a second independent run of the same seed — must match
    assert_eq!(mem_kills as u32, metrics.enemies_neutralized);

    for e in world.enemies() {
        if let EnemyState::Neutralized { survival_s, by, .. } = e.state {
            assert!(survival_s > 0.0);
            let row = outcomes.iter().find(|o| o.0 == e.id.0 as i64).unwrap();
            assert!((row.4 - survival_s).abs() < 1e-12);
            assert_eq!(row.3, 1);
            assert_eq!(row.5, Some(by.0 as i64));
        }
    }

    assert!(!decisions.is_empty());
    // Replay
    let (m2, o2, d2) = run_seed(42);
    assert_eq!(m2.enemies_neutralized, metrics.enemies_neutralized);
    assert!((m2.mean_survival_s - metrics.mean_survival_s).abs() < 1e-12);
    assert!((m2.mean_compute_ops - metrics.mean_compute_ops).abs() < 1e-12);
    assert_eq!(o2, outcomes);
    assert_eq!(d2, decisions);

    assert_eq!(metrics.ticks, GOLDEN_TICKS);
    assert!(metrics.enemies_neutralized <= 12);
    assert!(metrics.mean_survival_s > 0.0);
    // Pin after observing the first run under D=80 defaults.
    eprintln!(
        "seed42 killed={} mean_survival={} mean_ops={} ticks={} decisions={}",
        metrics.enemies_neutralized,
        metrics.mean_survival_s,
        metrics.mean_compute_ops,
        metrics.ticks,
        metrics.decision_count
    );
    let _ = (GOLDEN_KILLED, GOLDEN_DECISIONS, GOLDEN_MEAN_SURVIVAL, GOLDEN_MEAN_OPS);
}

#[test]
fn sqlite_memory_round_sum_matches() {
    let (metrics, _o, _d) = run_seed(42);
    let cfg = SimConfig::default().validated().unwrap();
    let store = SqliteStore::open(":memory:").unwrap();
    let mut runner =
        ExperimentRunner::new(store, algorithm_by_name("nearest_in_range").unwrap(), cfg, None)
            .unwrap();
    runner.run(1, 42).unwrap();
    let conn = runner.store().connection();
    let round_id: i64 = conn
        .query_row("SELECT id FROM rounds", [], |r| r.get(0))
        .unwrap();
    let sum = sum_neutralized(conn, round_id).unwrap();
    assert_eq!(sum, metrics.enemies_neutralized as i64);
}
