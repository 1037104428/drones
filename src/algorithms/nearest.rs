use crate::algorithm::{SelectResult, TargetingAlgorithm, TargetingInput};
use crate::contact::TargetContact;

pub struct NearestInRange;

impl TargetingAlgorithm for NearestInRange {
    fn name(&self) -> &'static str {
        "nearest_in_range"
    }

    fn select(&self, input: &TargetingInput<'_>) -> SelectResult {
        let mut ops: u64 = 0;
        let mut best: Option<&TargetContact> = None;
        for c in input.targets {
            ops += 1; // examine
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
                ops += 1; // update current best
                best = Some(c);
            }
        }
        ops += 1; // return
        SelectResult {
            target: best.map(|c| c.id),
            compute_ops: ops,
        }
    }
}
