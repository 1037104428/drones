pub mod algorithm;
pub mod algorithms;
pub mod config;
pub mod contact;
pub mod drone;
pub mod enemy;
pub mod experiment;
pub mod geom;
pub mod ids;
pub mod metrics;
pub mod persist;
pub mod plot;
pub mod world;

pub use algorithm::{algorithm_by_name, SelectResult, TargetingAlgorithm, TargetingInput};
pub use algorithms::{CloserThanFriend, NearestInRange, TransformerPolicy};
pub use config::{ConfigError, SimConfig};
pub use contact::{
    keep_nearest_drones, keep_nearest_targets, DroneContact, TargetContact, MAX_DRONE_CONTACTS,
    MAX_TARGET_CONTACTS,
};
pub use drone::{Drone, DroneState};
pub use enemy::{Enemy, EnemyState};
pub use experiment::{run_one, run_session, simulate_round, ExperimentRunner, RunSpec};
pub use geom::{
    all_rail_x, disk_laterally_covered, every_disk_point_near_a_rail, formation_rows,
    rail_spacing, row_x_positions, sample_point_in_disk, Position,
};
pub use ids::{DroneId, EnemyId};
pub use metrics::RoundMetrics;
pub use persist::{RunError, SqliteStore};
pub use world::{AbortReason, Engagement, NeutralizeError, SimEvent, World};
