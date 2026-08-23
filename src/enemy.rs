use crate::geom::Position;
use crate::ids::{DroneId, EnemyId};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EnemyState {
    Alive,
    Neutralized {
        at_tick: u64,
        by: DroneId,
        survival_s: f64,
    },
}

pub struct Enemy {
    pub id: EnemyId,
    pub pos: Position,
    pub spawn_tick: u64,
    pub state: EnemyState,
}

impl Enemy {
    pub(crate) fn new(id: EnemyId, pos: Position) -> Self {
        Self {
            id,
            pos,
            spawn_tick: 0,
            state: EnemyState::Alive,
        }
    }

    pub fn is_alive(&self) -> bool {
        matches!(self.state, EnemyState::Alive)
    }

    pub fn survival_s(&self, now_tick: u64, dt_s: f64) -> f64 {
        match self.state {
            EnemyState::Neutralized { survival_s, .. } => survival_s,
            EnemyState::Alive => (now_tick - self.spawn_tick) as f64 * dt_s,
        }
    }
}
