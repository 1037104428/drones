use crate::enemy::EnemyState;
use crate::world::World;

#[derive(Clone, Debug, PartialEq)]
pub struct RoundMetrics {
    pub round_index: u32,
    pub seed: u64,
    pub ticks: u64,
    pub sim_duration_s: f64,
    pub enemies_neutralized: u32,
    pub remaining_enemies: u32,
    pub mean_survival_s: f64,
    pub mean_survival_killed_s: Option<f64>,
    pub mean_compute_ops: f64,
    pub p50_compute_ops: u64,
    pub p99_compute_ops: u64,
    pub max_compute_ops: u64,
    pub decision_count: u64,
}

/// p50: index `len/2` (higher side of the two middles when even).
/// p99: index `floor(0.99 * (n-1))`.
pub fn p50_compute_ops(sorted: &[u64]) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    sorted[sorted.len() / 2]
}

pub fn p99_compute_ops(sorted: &[u64]) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let n = sorted.len();
    let idx = (0.99 * (n as f64 - 1.0)).floor() as usize;
    sorted[idx.min(n - 1)]
}

pub fn compute_round_metrics(
    world: &World,
    round_index: u32,
    seed: u64,
    mut compute_ops: Vec<u64>,
) -> RoundMetrics {
    let t_end = world.sim_time_s();
    let mut survivals = Vec::new();
    let mut killed_survivals = Vec::new();
    let mut neutralized = 0u32;
    for e in world.enemies() {
        match e.state {
            EnemyState::Neutralized { survival_s, .. } => {
                survivals.push(survival_s);
                killed_survivals.push(survival_s);
                neutralized += 1;
            }
            EnemyState::Alive => {
                survivals.push(t_end);
            }
        }
    }
    let remaining = world.enemies().len() as u32 - neutralized;
    let mean_survival_s = if survivals.is_empty() {
        0.0
    } else {
        survivals.iter().sum::<f64>() / survivals.len() as f64
    };
    let mean_survival_killed_s = if killed_survivals.is_empty() {
        None
    } else {
        Some(killed_survivals.iter().sum::<f64>() / killed_survivals.len() as f64)
    };

    compute_ops.sort_unstable();
    let decision_count = compute_ops.len() as u64;
    let mean_compute_ops = if compute_ops.is_empty() {
        0.0
    } else {
        compute_ops.iter().sum::<u64>() as f64 / compute_ops.len() as f64
    };
    let max_compute_ops = compute_ops.last().copied().unwrap_or(0);

    RoundMetrics {
        round_index,
        seed,
        ticks: world.tick,
        sim_duration_s: t_end,
        enemies_neutralized: neutralized,
        remaining_enemies: remaining,
        mean_survival_s,
        mean_survival_killed_s,
        mean_compute_ops,
        p50_compute_ops: p50_compute_ops(&compute_ops),
        p99_compute_ops: p99_compute_ops(&compute_ops),
        max_compute_ops,
        decision_count,
    }
}

#[cfg(test)]
mod tests {
    use super::{p50_compute_ops, p99_compute_ops};

    #[test]
    fn p50_uses_len_div_2() {
        let v = [1u64, 2, 3, 4];
        assert_eq!(p50_compute_ops(&v), 3);
        let v = [1u64, 2, 3];
        assert_eq!(p50_compute_ops(&v), 2);
    }

    #[test]
    fn p99_uses_floor_formula() {
        let v: Vec<u64> = (0..100).collect();
        // floor(0.99 * 99) = floor(98.01) = 98
        assert_eq!(p99_compute_ops(&v), 98);
    }
}
