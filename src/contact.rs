use arrayvec::ArrayVec;

use crate::geom::Position;
use crate::ids::{DroneId, EnemyId};

pub const MAX_TARGET_CONTACTS: usize = 8;
pub const MAX_DRONE_CONTACTS: usize = 8;

#[derive(Clone, Copy, Debug)]
pub struct TargetContact {
    pub id: EnemyId,
    pub distance: f64,
    pub pos: Position,
    pub engaged_by: Option<DroneId>,
}

#[derive(Clone, Copy, Debug)]
pub struct DroneContact {
    pub id: DroneId,
    pub distance: f64,
    pub pos: Position,
}

/// 保留最近 MAX_TARGET_CONTACTS 条。无 runtime cap 参数。
/// 排序键 (distance.total_cmp, id)；禁止用 < 比较 f64。
pub fn keep_nearest_targets<I>(items: I) -> ArrayVec<TargetContact, MAX_TARGET_CONTACTS>
where
    I: IntoIterator<Item = TargetContact>,
{
    let mut v: Vec<TargetContact> = items.into_iter().collect();
    v.sort_by(|a, b| a.distance.total_cmp(&b.distance).then(a.id.cmp(&b.id)));
    v.truncate(MAX_TARGET_CONTACTS);
    v.into_iter().collect()
}

pub fn keep_nearest_drones<I>(items: I) -> ArrayVec<DroneContact, MAX_DRONE_CONTACTS>
where
    I: IntoIterator<Item = DroneContact>,
{
    let mut v: Vec<DroneContact> = items.into_iter().collect();
    v.sort_by(|a, b| a.distance.total_cmp(&b.distance).then(a.id.cmp(&b.id)));
    v.truncate(MAX_DRONE_CONTACTS);
    v.into_iter().collect()
}
