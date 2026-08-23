use arrayvec::ArrayVec;

use crate::contact::{DroneContact, TargetContact, MAX_DRONE_CONTACTS, MAX_TARGET_CONTACTS};
use crate::geom::Position;
use crate::ids::DroneId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DroneState {
    Live,
    Expended { at_tick: u64 },
}

pub struct Drone {
    pub id: DroneId,
    pub pos: Position,
    pub detection_range: f64,
    pub state: DroneState,
    targets: ArrayVec<TargetContact, MAX_TARGET_CONTACTS>,
    drones: ArrayVec<DroneContact, MAX_DRONE_CONTACTS>,
    last_compute_ops: u64,
}

impl Drone {
    pub(crate) fn new(id: DroneId, pos: Position, detection_range: f64) -> Self {
        Self {
            id,
            pos,
            detection_range,
            state: DroneState::Live,
            targets: ArrayVec::new(),
            drones: ArrayVec::new(),
            last_compute_ops: 0,
        }
    }

    pub fn is_live(&self) -> bool {
        matches!(self.state, DroneState::Live)
    }

    pub fn get_nearby_targets(&self) -> &[TargetContact] {
        &self.targets
    }

    pub fn get_nearby_drones(&self) -> &[DroneContact] {
        &self.drones
    }

    pub fn last_compute_ops(&self) -> u64 {
        self.last_compute_ops
    }

    pub(crate) fn set_contacts(
        &mut self,
        targets: ArrayVec<TargetContact, MAX_TARGET_CONTACTS>,
        drones: ArrayVec<DroneContact, MAX_DRONE_CONTACTS>,
    ) {
        self.targets = targets;
        self.drones = drones;
    }

    pub(crate) fn set_last_compute_ops(&mut self, ops: u64) {
        self.last_compute_ops = ops;
    }

    pub(crate) fn clear_contacts(&mut self) {
        self.targets.clear();
        self.drones.clear();
    }
}
