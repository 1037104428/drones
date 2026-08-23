use std::collections::BTreeMap;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::algorithm::{SelectResult, TargetingAlgorithm, TargetingInput};
use crate::config::{ConfigError, SimConfig};
use crate::contact::{keep_nearest_drones, keep_nearest_targets, DroneContact, TargetContact};
use crate::drone::{Drone, DroneState};
use crate::enemy::{Enemy, EnemyState};
use crate::geom::{row_x_positions, sample_point_in_disk, Position};
use crate::ids::{DroneId, EnemyId};

#[derive(Clone, Copy, Debug)]
pub struct Engagement {
    pub drone_id: DroneId,
    pub enemy_id: EnemyId,
    pub started_tick: u64,
}

pub struct World {
    pub config: SimConfig,
    pub seed: u64,
    pub tick: u64,
    drones: Vec<Drone>,
    enemies: Vec<Enemy>,
    /// 锁定唯一真相源。BTreeMap：progress 顺序确定。
    engagements: BTreeMap<EnemyId, Engagement>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SimEvent {
    Decision {
        tick: u64,
        drone: DroneId,
        target: Option<EnemyId>,
        compute_ops: u64,
    },
    Neutralized {
        tick: u64,
        enemy: EnemyId,
        by: DroneId,
        survival_s: f64,
        pos: Position,
    },
    EngagementAborted {
        tick: u64,
        drone: DroneId,
        enemy: EnemyId,
        reason: AbortReason,
    },
}

/// 飞离探测圈。排他锁下「目标已死仍占锁」是实现 bug：
/// progress 里 `debug_assert!` + 当作 OutOfRange 清锁，不另开 TargetGone。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbortReason {
    OutOfRange,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NeutralizeError {
    TargetNotFound,
    TargetAlreadyDead,
    OutOfRange,
    LockedByOther { owner: DroneId },
    DroneBusy { current: EnemyId },
    DroneExpended,
    DroneNotFound,
}

impl World {
    pub fn new(config: SimConfig, seed: u64) -> Result<Self, ConfigError> {
        let config = config.validated()?;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut positions = Vec::with_capacity(config.enemy_count as usize);
        for _ in 0..config.enemy_count {
            positions.push(sample_point_in_disk(&mut rng, config.radius_m));
        }
        Ok(Self::from_positions(config, seed, positions)?)
    }

    /// 测试/夹具：跳过圆盘采样，按给定坐标放置敌人。
    /// `positions.len()` 必须等于 `config.enemy_count`。
    /// **不**要求 `in_disk`。
    pub fn with_enemy_positions(
        config: SimConfig,
        positions: Vec<Position>,
    ) -> Result<Self, ConfigError> {
        let config = config.validated()?;
        if positions.len() != config.enemy_count as usize {
            return Err(ConfigError::InvalidFormation {
                reason: format!(
                    "positions.len()={} != enemy_count={}",
                    positions.len(),
                    config.enemy_count
                ),
            });
        }
        Self::from_positions(config, 0, positions)
    }

    fn from_positions(
        config: SimConfig,
        seed: u64,
        positions: Vec<Position>,
    ) -> Result<Self, ConfigError> {
        let enemies: Vec<Enemy> = positions
            .into_iter()
            .enumerate()
            .map(|(i, pos)| Enemy::new(EnemyId(i as u32), pos))
            .collect();

        let xs = row_x_positions(config.radius_m, config.drones_per_row);
        let y_lead = -config.radius_m - config.start_margin_m;
        let mut drones = Vec::new();
        let mut next_id = 0u32;
        for row in 0..config.drone_rows {
            let y = y_lead - (row as f64) * config.row_gap_m;
            for &x in &xs {
                drones.push(Drone::new(
                    DroneId(next_id),
                    Position { x, y },
                    config.detection_range_m,
                ));
                next_id += 1;
            }
        }

        Ok(Self {
            config,
            seed,
            tick: 0,
            drones,
            enemies,
            engagements: BTreeMap::new(),
        })
    }

    pub fn sim_time_s(&self) -> f64 {
        self.tick as f64 * self.config.dt_s
    }

    pub fn drones(&self) -> &[Drone] {
        &self.drones
    }

    pub fn enemies(&self) -> &[Enemy] {
        &self.enemies
    }

    pub fn drone(&self, id: DroneId) -> Option<&Drone> {
        self.drones.iter().find(|d| d.id == id)
    }

    fn drone_mut(&mut self, id: DroneId) -> Option<&mut Drone> {
        self.drones.iter_mut().find(|d| d.id == id)
    }

    fn enemy(&self, id: EnemyId) -> Option<&Enemy> {
        self.enemies.iter().find(|e| e.id == id)
    }

    fn enemy_mut(&mut self, id: EnemyId) -> Option<&mut Enemy> {
        self.enemies.iter_mut().find(|e| e.id == id)
    }

    pub fn engagement_of(&self, id: DroneId) -> Option<EnemyId> {
        self.engagements
            .values()
            .find(|e| e.drone_id == id)
            .map(|e| e.enemy_id)
    }

    pub fn is_finished(&self) -> bool {
        if self.tick >= self.config.max_ticks {
            return true;
        }
        let min_y = self
            .drones
            .iter()
            .map(|d| d.pos.y)
            .fold(f64::INFINITY, f64::min);
        min_y >= self.config.radius_m + self.config.end_margin_m
    }

    /// 唯一公开步进。返回本步事件；不写 DB。
    pub fn step(&mut self, algo: &dyn TargetingAlgorithm) -> Vec<SimEvent> {
        let v = self.config.speed_m_s;
        let dt = self.config.dt_s;
        let kill_ticks = self.config.kill_ticks();
        let tick = self.tick;

        for d in &mut self.drones {
            d.pos.y += v * dt;
        }
        self.sense_all();

        let mut events = Vec::new();
        let ids: Vec<DroneId> = self
            .drones
            .iter()
            .filter(|d| d.is_live())
            .map(|d| d.id)
            .collect();
        for id in ids {
            let result = self.compute_for(id, algo, tick);
            events.push(SimEvent::Decision {
                tick,
                drone: id,
                target: result.target,
                compute_ops: result.compute_ops,
            });
            if self.engagement_of(id).is_none() {
                if let Some(tid) = result.target {
                    let _ = self.try_acquire(id, tid, tick);
                }
            }
        }
        self.progress_engagements(tick, kill_ticks, &mut events);
        self.tick += 1;
        events
    }

    fn sense_all(&mut self) {
        let eng_snapshot: BTreeMap<EnemyId, DroneId> = self
            .engagements
            .iter()
            .map(|(eid, eng)| (*eid, eng.drone_id))
            .collect();

        let live: Vec<(DroneId, Position)> = self
            .drones
            .iter()
            .filter(|d| d.is_live())
            .map(|d| (d.id, d.pos))
            .collect();
        let alive: Vec<(EnemyId, Position)> = self
            .enemies
            .iter()
            .filter(|e| e.is_alive())
            .map(|e| (e.id, e.pos))
            .collect();

        for d in &mut self.drones {
            if !d.is_live() {
                continue;
            }
            let targets = keep_nearest_targets(alive.iter().map(|(eid, pos)| TargetContact {
                id: *eid,
                distance: d.pos.distance(*pos),
                pos: *pos,
                engaged_by: eng_snapshot.get(eid).copied(),
            }));
            let friends = keep_nearest_drones(live.iter().filter(|(id, _)| *id != d.id).map(
                |(id, pos)| DroneContact {
                    id: *id,
                    distance: d.pos.distance(*pos),
                    pos: *pos,
                },
            ));
            d.set_contacts(targets, friends);
        }
    }

    fn compute_for(
        &mut self,
        id: DroneId,
        algo: &dyn TargetingAlgorithm,
        tick: u64,
    ) -> SelectResult {
        let drone = self.drone(id).expect("live drone id from step");
        let self_pos = drone.pos;
        let detection_range = drone.detection_range;
        let targets: Vec<TargetContact> = drone.get_nearby_targets().to_vec();
        let drones: Vec<crate::contact::DroneContact> = drone.get_nearby_drones().to_vec();
        let current_engagement = self.engagement_of(id);
        let input = TargetingInput {
            self_id: id,
            self_pos,
            tick,
            detection_range,
            targets: &targets,
            drones: &drones,
            current_engagement,
        };
        let result = algo.select(&input);
        if let Some(d) = self.drone_mut(id) {
            d.set_last_compute_ops(result.compute_ops);
        }
        result
    }

    fn try_acquire(
        &mut self,
        drone_id: DroneId,
        enemy_id: EnemyId,
        tick: u64,
    ) -> Result<(), NeutralizeError> {
        let drone = self.drone(drone_id).ok_or(NeutralizeError::DroneNotFound)?;
        if !drone.is_live() {
            return Err(NeutralizeError::DroneExpended);
        }
        if let Some(current) = self.engagement_of(drone_id) {
            return Err(NeutralizeError::DroneBusy { current });
        }
        let d_pos = drone.pos;
        let d_range = drone.detection_range;
        let enemy = self
            .enemy(enemy_id)
            .ok_or(NeutralizeError::TargetNotFound)?;
        if !enemy.is_alive() {
            return Err(NeutralizeError::TargetAlreadyDead);
        }
        if d_pos.distance(enemy.pos) > d_range {
            return Err(NeutralizeError::OutOfRange);
        }
        if let Some(eng) = self.engagements.get(&enemy_id) {
            return Err(NeutralizeError::LockedByOther {
                owner: eng.drone_id,
            });
        }
        self.engagements.insert(
            enemy_id,
            Engagement {
                drone_id,
                enemy_id,
                started_tick: tick,
            },
        );
        tracing::debug!(
            tick,
            drone = drone_id.0,
            enemy = enemy_id.0,
            "acquired lock"
        );
        Ok(())
    }

    fn progress_engagements(&mut self, tick: u64, kill_ticks: u64, events: &mut Vec<SimEvent>) {
        let enemy_ids: Vec<EnemyId> = self.engagements.keys().copied().collect();
        for eid in enemy_ids {
            let Some(eng) = self.engagements.get(&eid).copied() else {
                continue;
            };
            let Some(drone) = self.drone(eng.drone_id) else {
                continue;
            };
            let Some(enemy) = self.enemy(eid) else {
                continue;
            };
            if !enemy.is_alive() {
                debug_assert!(
                    false,
                    "exclusive lock on a dead enemy is an implementation bug"
                );
                self.engagements.remove(&eid);
                events.push(SimEvent::EngagementAborted {
                    tick,
                    drone: eng.drone_id,
                    enemy: eid,
                    reason: AbortReason::OutOfRange,
                });
                continue;
            }
            let dist = drone.pos.distance(enemy.pos);
            if dist > drone.detection_range {
                self.engagements.remove(&eid);
                tracing::debug!(
                    tick,
                    drone = eng.drone_id.0,
                    enemy = eid.0,
                    reason = "OutOfRange",
                    "engagement aborted"
                );
                events.push(SimEvent::EngagementAborted {
                    tick,
                    drone: eng.drone_id,
                    enemy: eid,
                    reason: AbortReason::OutOfRange,
                });
                continue;
            }
            if tick.saturating_sub(eng.started_tick) >= kill_ticks {
                match self.complete_neutralization(eng.drone_id, eid) {
                    Ok(ev) => events.push(ev),
                    Err(err) => {
                        tracing::error!(?err, "complete_neutralization failed");
                    }
                }
            }
        }
    }

    pub(crate) fn complete_neutralization(
        &mut self,
        drone_id: DroneId,
        enemy_id: EnemyId,
    ) -> Result<SimEvent, NeutralizeError> {
        let tick = self.tick;
        let dt = self.config.dt_s;
        let drone = self.drone(drone_id).ok_or(NeutralizeError::DroneNotFound)?;
        if !drone.is_live() {
            return Err(NeutralizeError::DroneExpended);
        }
        let d_pos = drone.pos;
        let d_range = drone.detection_range;
        match self.engagements.get(&enemy_id) {
            Some(eng) if eng.drone_id == drone_id => {}
            Some(eng) => {
                return Err(NeutralizeError::LockedByOther {
                    owner: eng.drone_id,
                });
            }
            None => return Err(NeutralizeError::TargetNotFound),
        }
        let enemy = self
            .enemy(enemy_id)
            .ok_or(NeutralizeError::TargetNotFound)?;
        if !enemy.is_alive() {
            return Err(NeutralizeError::TargetAlreadyDead);
        }
        let e_pos = enemy.pos;
        if d_pos.distance(e_pos) > d_range {
            return Err(NeutralizeError::OutOfRange);
        }

        let survival_s = tick as f64 * dt;
        let enemy = self
            .enemy_mut(enemy_id)
            .ok_or(NeutralizeError::TargetNotFound)?;
        enemy.state = EnemyState::Neutralized {
            at_tick: tick,
            by: drone_id,
            survival_s,
        };
        self.engagements.remove(&enemy_id);
        let drone = self.drone_mut(drone_id).ok_or(NeutralizeError::DroneNotFound)?;
        drone.state = DroneState::Expended { at_tick: tick };
        drone.clear_contacts();

        tracing::info!(
            tick,
            enemy = enemy_id.0,
            by = drone_id.0,
            survival_s,
            drone_state = "Expended",
            "neutralized"
        );

        Ok(SimEvent::Neutralized {
            tick,
            enemy: enemy_id,
            by: drone_id,
            survival_s,
            pos: e_pos,
        })
    }
}
