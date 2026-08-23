use crate::algorithm::{SelectResult, TargetingAlgorithm, TargetingInput};
use crate::contact::TargetContact;

/// 研究算法：
/// 1. 最近友机距离
/// 2. 最近（可中和）目标距离
/// 3. 仅当最近目标 **严格近于** 最近友机时，中和该目标
pub struct CloserThanFriend;

impl TargetingAlgorithm for CloserThanFriend {
    fn name(&self) -> &'static str {
        "closer_than_friend"
    }

    fn select(&self, input: &TargetingInput<'_>) -> SelectResult {
        let mut ops: u64 = 0;

        let mut min_friend = f64::INFINITY;
        for c in input.drones {
            ops += 1;
            if c.distance < min_friend {
                ops += 1;
                min_friend = c.distance;
            }
        }

        let mut best: Option<&TargetContact> = None;
        for c in input.targets {
            ops += 1;
            let eligible = c.distance <= input.detection_range
                && match c.engaged_by {
                    None => true,
                    Some(owner) => owner == input.self_id,
                };
            if !eligible {
                continue;
            }
            let better = match best {
                None => true,
                Some(b) => c
                    .distance
                    .total_cmp(&b.distance)
                    .then(c.id.cmp(&b.id))
                    .is_lt(),
            };
            if better {
                ops += 1;
                best = Some(c);
            }
        }

        ops += 1; // compare nearest-target vs nearest-friend
        ops += 1; // return

        let target = match best {
            Some(c) if c.distance < min_friend => Some(c.id),
            _ => None,
        };
        SelectResult { target, compute_ops: ops }
    }
}
