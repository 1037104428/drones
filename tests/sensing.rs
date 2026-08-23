use battlefield_sim::config::SimConfig;
use battlefield_sim::geom::Position;
use battlefield_sim::ids::{DroneId, EnemyId};
use battlefield_sim::world::SimEvent;
use battlefield_sim::{NearestInRange, World};

fn cfg_1row(enemies: u32, drones_per_row: u32) -> SimConfig {
    let mut c = SimConfig::default();
    c.enemy_count = enemies;
    c.drone_rows = 1;
    c.drones_per_row = drones_per_row;
    c
}

#[test]
fn nearest_8_drops_9th() {
    let cfg = cfg_1row(9, 2).validated().unwrap();
    let y = -cfg.radius_m - cfg.start_margin_m + cfg.speed_m_s * cfg.dt_s;
    let x0 = -cfg.radius_m;
    let positions: Vec<Position> = (1..=9)
        .map(|k| Position {
            x: x0 + k as f64,
            y,
        })
        .collect();
    let mut world = World::with_enemy_positions(cfg, positions).unwrap();
    world.step(&NearestInRange);
    let mem = world.drone(DroneId(0)).unwrap().get_nearby_targets();
    assert_eq!(mem.len(), 8);
    let ids: Vec<u32> = mem.iter().map(|c| c.id.0).collect();
    assert_eq!(ids, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    assert!(mem.iter().all(|c| c.distance <= 8.0 + 1e-9));
}

#[test]
fn equal_distance_keeps_smaller_id() {
    let cfg = cfg_1row(9, 2).validated().unwrap();
    let y = -cfg.radius_m - cfg.start_margin_m + cfg.speed_m_s * cfg.dt_s;
    let x0 = -cfg.radius_m;
    let mut positions = Vec::new();
    for i in 1..=7 {
        positions.push(Position {
            x: x0 + i as f64,
            y,
        });
    }
    positions.push(Position { x: x0 + 10.0, y }); // id 7, dist 10
    positions.push(Position { x: x0 - 10.0, y }); // id 8, dist 10
    let mut world = World::with_enemy_positions(cfg, positions).unwrap();
    world.step(&NearestInRange);
    let mem = world.drone(DroneId(0)).unwrap().get_nearby_targets();
    let ids: Vec<u32> = mem.iter().map(|c| c.id.0).collect();
    assert_eq!(ids, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    assert!(!ids.contains(&8));
}

#[test]
fn self_not_in_friend_list() {
    let cfg = cfg_1row(1, 2).validated().unwrap();
    let positions = vec![Position { x: 0.0, y: 0.0 }];
    let mut world = World::with_enemy_positions(cfg, positions).unwrap();
    world.step(&NearestInRange);
    for d in world.drones() {
        assert!(d.get_nearby_drones().iter().all(|c| c.id != d.id));
    }
}

#[test]
fn out_of_range_target_not_in_memory() {
    let cfg = cfg_1row(1, 2).validated().unwrap();
    let positions = vec![Position { x: 0.0, y: 0.0 }];
    let mut world = World::with_enemy_positions(cfg, positions).unwrap();
    world.step(&NearestInRange);
    let mem = world.drone(DroneId(0)).unwrap().get_nearby_targets();
    assert!(
        mem.iter().all(|c| c.id != EnemyId(0)),
        "targets outside D must not occupy the 8-slot memory"
    );
}

#[test]
fn engaged_by_is_sense_snapshot_not_same_tick_acquire() {
    let mut cfg = SimConfig::default();
    cfg.radius_m = 40.0;
    cfg.enemy_count = 1;
    cfg.drone_rows = 1;
    cfg.drones_per_row = 2;
    cfg.detection_range_m = 80.0;
    cfg.start_margin_m = 50.0;
    cfg.end_margin_m = 50.0;
    let cfg = cfg.validated().unwrap();
    let y = -cfg.radius_m - cfg.start_margin_m + cfg.speed_m_s * cfg.dt_s;
    let positions = vec![Position { x: 0.0, y }];
    let mut world = World::with_enemy_positions(cfg, positions).unwrap();
    let events = world.step(&NearestInRange);
    let mem0 = world.drone(DroneId(0)).unwrap().get_nearby_targets();
    assert_eq!(mem0[0].engaged_by, None, "same-tick acquire must be stale");
    let decisions: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            SimEvent::Decision { drone, target, .. } => Some((*drone, *target)),
            _ => None,
        })
        .collect();
    assert!(decisions.iter().any(|(d, t)| *d == DroneId(0) && *t == Some(EnemyId(0))));
    assert_eq!(world.engagement_of(DroneId(0)), Some(EnemyId(0)));
    world.step(&NearestInRange);
    let mem0 = world.drone(DroneId(0)).unwrap().get_nearby_targets();
    assert_eq!(mem0[0].engaged_by, Some(DroneId(0)));
}

#[test]
fn corpse_leaves_memory_on_next_sense() {
    let mut cfg = SimConfig::default();
    cfg.enemy_count = 1;
    cfg.drone_rows = 1;
    cfg.drones_per_row = 2;
    let cfg = cfg.validated().unwrap();
    let x0 = -cfg.radius_m;
    let y = -cfg.radius_m - cfg.start_margin_m + cfg.speed_m_s * cfg.dt_s;
    let mut world =
        World::with_enemy_positions(cfg, vec![Position { x: x0, y }]).unwrap();
    let mut killed = false;
    for _ in 0..20 {
        let events = world.step(&NearestInRange);
        if events.iter().any(|e| matches!(e, SimEvent::Neutralized { .. })) {
            killed = true;
            break;
        }
    }
    assert!(killed);
    world.step(&NearestInRange);
    for d in world.drones() {
        assert!(
            d.get_nearby_targets()
                .iter()
                .all(|c| c.id != EnemyId(0)),
            "corpse still in drone {} memory",
            d.id.0
        );
    }
}

#[test]
fn expended_friend_not_in_neighbor_list() {
    let mut cfg = SimConfig::default();
    cfg.enemy_count = 1;
    cfg.drone_rows = 1;
    cfg.drones_per_row = 2;
    cfg.detection_range_m = 500.0;
    let cfg = cfg.validated().unwrap();
    let x0 = -cfg.radius_m;
    let y = -cfg.radius_m - cfg.start_margin_m + cfg.speed_m_s * cfg.dt_s;
    let mut world =
        World::with_enemy_positions(cfg, vec![Position { x: x0, y }]).unwrap();
    for _ in 0..20 {
        world.step(&NearestInRange);
        if !world.drone(DroneId(0)).unwrap().is_live() {
            break;
        }
    }
    assert!(!world.drone(DroneId(0)).unwrap().is_live());
    world.step(&NearestInRange);
    let friends = world.drone(DroneId(1)).unwrap().get_nearby_drones();
    assert!(friends.iter().all(|c| c.id != DroneId(0)));
}

#[test]
fn friend_outside_detection_range_not_in_memory() {
    let cfg = cfg_1row(1, 2).validated().unwrap();
    assert!(cfg.detection_range_m < 2.0 * cfg.radius_m);
    let mut world =
        World::with_enemy_positions(cfg.clone(), vec![Position { x: 0.0, y: 0.0 }]).unwrap();
    world.step(&NearestInRange);
    let d0 = world.drone(DroneId(0)).unwrap();
    let d1 = world.drone(DroneId(1)).unwrap();
    assert!(d0.pos.distance(d1.pos) > cfg.detection_range_m);
    assert!(d0.get_nearby_drones().is_empty());
    assert!(d1.get_nearby_drones().is_empty());
}
