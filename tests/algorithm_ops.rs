use battlefield_sim::algorithm::{SelectResult, TargetingAlgorithm, TargetingInput};
use battlefield_sim::contact::{DroneContact, TargetContact};
use battlefield_sim::geom::Position;
use battlefield_sim::ids::{DroneId, EnemyId};
use battlefield_sim::{CloserThanFriend, NearestInRange};

fn contact(id: u32, distance: f64, engaged_by: Option<u32>) -> TargetContact {
    TargetContact {
        id: EnemyId(id),
        distance,
        pos: Position {
            x: distance,
            y: 0.0,
        },
        engaged_by: engaged_by.map(DroneId),
    }
}

fn input<'a>(targets: &'a [TargetContact], range: f64) -> TargetingInput<'a> {
    TargetingInput {
        self_id: DroneId(0),
        self_pos: Position { x: 0.0, y: 0.0 },
        tick: 0,
        detection_range: range,
        targets,
        drones: &[],
        current_engagement: None,
    }
}

#[test]
fn empty_targets_ops_1_none() {
    let r = NearestInRange.select(&input(&[], 50.0));
    assert_eq!(
        r,
        SelectResult {
            target: None,
            compute_ops: 1
        }
    );
}

#[test]
fn three_all_out_of_range() {
    let t = [
        contact(0, 60.0, None),
        contact(1, 70.0, None),
        contact(2, 80.0, None),
    ];
    let r = NearestInRange.select(&input(&t, 50.0));
    assert_eq!(
        r,
        SelectResult {
            target: None,
            compute_ops: 4
        }
    );
}

#[test]
fn one_eligible() {
    let t = [contact(3, 10.0, None)];
    let r = NearestInRange.select(&input(&t, 50.0));
    assert_eq!(
        r,
        SelectResult {
            target: Some(EnemyId(3)),
            compute_ops: 3
        }
    );
}

#[test]
fn two_eligible_nearer_first() {
    let t = [contact(1, 5.0, None), contact(2, 15.0, None)];
    let r = NearestInRange.select(&input(&t, 50.0));
    assert_eq!(
        r,
        SelectResult {
            target: Some(EnemyId(1)),
            compute_ops: 4
        }
    );
}

#[test]
fn two_eligible_farther_first() {
    let t = [contact(1, 15.0, None), contact(2, 5.0, None)];
    let r = NearestInRange.select(&input(&t, 50.0));
    assert_eq!(
        r,
        SelectResult {
            target: Some(EnemyId(2)),
            compute_ops: 5
        }
    );
}

#[test]
fn select_is_pure() {
    let t = [contact(1, 15.0, None), contact(2, 5.0, None)];
    let inp = input(&t, 50.0);
    let a = NearestInRange.select(&inp);
    let b = NearestInRange.select(&inp);
    assert_eq!(a, b);
}

#[test]
fn skips_engaged_by_other() {
    let t = [contact(1, 5.0, Some(9)), contact(2, 8.0, None)];
    let r = NearestInRange.select(&input(&t, 50.0));
    assert_eq!(r.target, Some(EnemyId(2)));
    // examine 2 + update 1 (only id 2) + return 1 = 4
    assert_eq!(r.compute_ops, 4);
}

fn friend(id: u32, distance: f64) -> DroneContact {
    DroneContact {
        id: DroneId(id),
        distance,
        pos: Position {
            x: distance,
            y: 0.0,
        },
    }
}

fn input_with_friends<'a>(
    targets: &'a [TargetContact],
    drones: &'a [DroneContact],
    range: f64,
) -> TargetingInput<'a> {
    TargetingInput {
        self_id: DroneId(0),
        self_pos: Position { x: 0.0, y: 0.0 },
        tick: 0,
        detection_range: range,
        targets,
        drones,
        current_engagement: None,
    }
}

#[test]
fn closer_than_friend_fires_when_target_nearer() {
    let t = [contact(1, 10.0, None)];
    let f = [friend(2, 40.0)];
    let r = CloserThanFriend.select(&input_with_friends(&t, &f, 50.0));
    assert_eq!(r.target, Some(EnemyId(1)));
    // friends: exam 1 + update 1; targets: exam 1 + update 1; compare 1; return 1 = 6
    assert_eq!(r.compute_ops, 6);
}

#[test]
fn closer_than_friend_holds_when_friend_nearer() {
    let t = [contact(1, 45.0, None)];
    let f = [friend(2, 40.0)];
    let r = CloserThanFriend.select(&input_with_friends(&t, &f, 50.0));
    assert_eq!(r.target, None);
    assert_eq!(r.compute_ops, 6);
}

#[test]
fn closer_than_friend_equal_distance_does_not_fire() {
    let t = [contact(1, 40.0, None)];
    let f = [friend(2, 40.0)];
    let r = CloserThanFriend.select(&input_with_friends(&t, &f, 50.0));
    assert_eq!(r.target, None);
}

#[test]
fn closer_than_friend_no_friends_fires() {
    let t = [contact(1, 10.0, None)];
    let r = CloserThanFriend.select(&input_with_friends(&t, &[], 50.0));
    assert_eq!(r.target, Some(EnemyId(1)));
    // targets exam+update, compare, return = 4
    assert_eq!(r.compute_ops, 4);
}
