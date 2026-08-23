use crate::config::ConfigError;
use crate::contact::{DroneContact, TargetContact};
use crate::geom::Position;
use crate::ids::{DroneId, EnemyId};

pub struct TargetingInput<'a> {
    pub self_id: DroneId,
    pub self_pos: Position,
    pub tick: u64,
    pub detection_range: f64,
    pub targets: &'a [TargetContact],
    pub drones: &'a [DroneContact],
    /// 本机当前锁（来自 World.engagements）。锁定期间仍调用 select，但 World 忽略 target。
    pub current_engagement: Option<EnemyId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectResult {
    pub target: Option<EnemyId>,
    /// 按 v1 costing 规则累加的确定性操作数。评分唯一依据。
    pub compute_ops: u64,
}

pub trait TargetingAlgorithm: Send + Sync {
    fn name(&self) -> &'static str;
    fn select(&self, input: &TargetingInput<'_>) -> SelectResult;
}

pub fn algorithm_by_name(name: &str) -> Result<Box<dyn TargetingAlgorithm>, ConfigError> {
    match name {
        "nearest_in_range" => Ok(Box::new(crate::algorithms::NearestInRange)),
        "closer_than_friend" => Ok(Box::new(crate::algorithms::CloserThanFriend)),
        "transformer" => Ok(Box::new(
            crate::algorithms::TransformerPolicy::load_or_new(
                crate::algorithms::TransformerPolicy::default_model_path(),
                0,
            ),
        )),
        _ => Err(ConfigError::UnknownAlgorithm {
            name: name.to_string(),
        }),
    }
}
