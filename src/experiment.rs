use std::path::PathBuf;

use crate::algorithm::TargetingAlgorithm;
use crate::config::SimConfig;
use crate::drone::DroneState;
use crate::metrics::{compute_round_metrics, RoundMetrics};
use crate::persist::{RunError, SqliteStore};
use crate::world::{SimEvent, World};

pub struct RunSpec {
    pub rounds: u32,
    pub base_seed: u64,
    pub algorithm: String,
    pub db_path: PathBuf,
    pub config: SimConfig,
    pub log_every_ticks: Option<u64>,
}

pub struct ExperimentRunner {
    store: SqliteStore,
    algo: Box<dyn TargetingAlgorithm>,
    config: SimConfig,
    log_every_ticks: Option<u64>,
}

impl ExperimentRunner {
    pub fn new(
        store: SqliteStore,
        algo: Box<dyn TargetingAlgorithm>,
        config: SimConfig,
        log_every_ticks: Option<u64>,
    ) -> Result<Self, RunError> {
        let config = config.validated()?;
        Ok(Self {
            store,
            algo,
            config,
            log_every_ticks,
        })
    }

    /// round i 使用 seed = base_seed.wrapping_add(i as u64)，round_index 从 0 起。
    pub fn run(
        &mut self,
        rounds: u32,
        base_seed: u64,
    ) -> Result<Vec<RoundMetrics>, RunError> {
        let algo = self.algo.as_ref();
        let config = &self.config;
        let log = self.log_every_ticks;
        run_session(&mut self.store, algo, config, rounds, base_seed, log)
    }

    pub fn store(&self) -> &SqliteStore {
        &self.store
    }
}

/// Run `rounds` starting at `base_seed` with a borrowed algorithm (for training loops).
pub fn run_session(
    store: &mut SqliteStore,
    algo: &dyn TargetingAlgorithm,
    config: &SimConfig,
    rounds: u32,
    base_seed: u64,
    log_every_ticks: Option<u64>,
) -> Result<Vec<RoundMetrics>, RunError> {
    if rounds < 1 {
        return Err(RunError::Invariant("rounds must be >= 1".into()));
    }
    let mut out = Vec::with_capacity(rounds as usize);
    for i in 0..rounds {
        let seed = base_seed.wrapping_add(i as u64);
        out.push(run_one(store, algo, config, i, seed, log_every_ticks)?);
    }
    Ok(out)
}

pub fn run_one(
    store: &mut SqliteStore,
    algo: &dyn TargetingAlgorithm,
    config: &SimConfig,
    round_index: u32,
    seed: u64,
    log_every_ticks: Option<u64>,
) -> Result<RoundMetrics, RunError> {
        let mut world = World::new(config.clone(), seed)?;
        let algo_name = algo.name();
        let round_id = store.begin_round(round_index, seed, algo_name, config)?;

        tracing::info!(
            round_index,
            seed,
            algo = algo_name,
            R = config.radius_m,
            D = config.detection_range_m,
            v = config.speed_m_s,
            dt = config.dt_s,
            expend_on_kill = config.expend_on_kill,
            "round start"
        );
        for e in world.enemies() {
            tracing::debug!(id = e.id.0, x = e.pos.x, y = e.pos.y, "enemy placed");
        }

        let mut ops = Vec::new();
        while !world.is_finished() {
            let events = world.step(algo);
            for ev in &events {
                if let SimEvent::Decision { compute_ops, .. } = ev {
                    ops.push(*compute_ops);
                }
            }
            store.apply_events(round_id, &events)?;
            if let Some(n) = log_every_ticks {
                if n > 0 && world.tick % n == 0 {
                    log_snapshot(&world);
                }
            }
        }

        let t_end = world.sim_time_s();
        let metrics = compute_round_metrics(&world, round_index, seed, ops);
        store.finalize_round(round_id, &world, t_end, &metrics)?;
        tracing::info!(
            round_index,
            seed,
            ticks = metrics.ticks,
            sim_s = metrics.sim_duration_s,
            killed = metrics.enemies_neutralized,
            remaining = metrics.remaining_enemies,
            mean_survival_s = metrics.mean_survival_s,
            mean_compute_ops = metrics.mean_compute_ops,
            "round end"
        );
        Ok(metrics)
}

/// Kernel-only round (no SQLite). Used while training the transformer.
pub fn simulate_round(
    algo: &dyn TargetingAlgorithm,
    config: &SimConfig,
    round_index: u32,
    seed: u64,
) -> Result<RoundMetrics, RunError> {
    let mut world = World::new(config.clone(), seed)?;
    let mut ops = Vec::new();
    while !world.is_finished() {
        let events = world.step(algo);
        for ev in &events {
            if let SimEvent::Decision { compute_ops, .. } = ev {
                ops.push(*compute_ops);
            }
        }
    }
    Ok(compute_round_metrics(&world, round_index, seed, ops))
}

fn log_snapshot(world: &World) {
    let alive = world.enemies().iter().filter(|e| e.is_alive()).count();
    let locks = world
        .drones()
        .iter()
        .filter(|d| world.engagement_of(d.id).is_some())
        .count();
    for d in world.drones() {
        let state = match d.state {
            DroneState::Live => "Live",
            DroneState::Expended { .. } => "Expended",
        };
        tracing::info!(
            tick = world.tick,
            drone = d.id.0,
            x = d.pos.x,
            y = d.pos.y,
            state,
            alive_enemies = alive,
            locks,
            "tick snapshot"
        );
    }
}

impl RunSpec {
    pub fn print_summary(metrics: &RoundMetrics, algo: &str, db: &std::path::Path, enemies_total: u32) {
        println!(
            "round={} seed={} algo={} ticks={} sim_s={:.4}",
            metrics.round_index, metrics.seed, algo, metrics.ticks, metrics.sim_duration_s
        );
        let killed_s = metrics
            .mean_survival_killed_s
            .map(|v| format!("{v:.6}"))
            .unwrap_or_else(|| "null".into());
        println!(
            "killed={}/{} remaining={} mean_survival_s={:.6} mean_survival_killed_s={}",
            metrics.enemies_neutralized,
            enemies_total,
            metrics.remaining_enemies,
            metrics.mean_survival_s,
            killed_s
        );
        println!(
            "compute_ops mean/p50/p99={:.4}/{}/{} decisions={}",
            metrics.mean_compute_ops,
            metrics.p50_compute_ops,
            metrics.p99_compute_ops,
            metrics.decision_count
        );
        println!("db={} round_index={}", db.display(), metrics.round_index);
    }
}
