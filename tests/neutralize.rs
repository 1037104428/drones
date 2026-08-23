use battlefield_sim::config::SimConfig;
use battlefield_sim::drone::DroneState;
use battlefield_sim::enemy::EnemyState;
use battlefield_sim::geom::{formation_rows, Position};
use battlefield_sim::ids::{DroneId, EnemyId};
use battlefield_sim::world::{AbortReason, SimEvent};
use battlefield_sim::{NearestInRange, World};

fn on_rail_cfg(enemies: u32) -> SimConfig {
    let mut c = SimConfig::default();
    c.enemy_count = enemies;
    c.drone_rows = 1;
    c.drones_per_row = 2;
    c
}

#[test]
fn t_kill_truth_table() {
    let cfg = on_rail_cfg(1).validated().unwrap();
    let x0 = -cfg.radius_m;
    let y = -cfg.radius_m - cfg.start_margin_m + cfg.speed_m_s * cfg.dt_s;
    let mut world = World::with_enemy_positions(cfg.clone(), vec![Position { x: x0, y }]).unwrap();
    let mut neutralized: Option<SimEvent> = None;
    for _ in 0..6 {
        let events = world.step(&NearestInRange);
        if let Some(ev) = events
            .into_iter()
            .find(|e| matches!(e, SimEvent::Neutralized { .. }))
        {
            neutralized = Some(ev);
        }
    }
    let SimEvent::Neutralized {
        tick,
        survival_s,
        enemy,
        by,
        ..
    } = neutralized.expect("should kill on 6th step")
    else {
        panic!("wrong event");
    };
    assert_eq!(tick, 5);
    assert_eq!(world.tick, 6);
    assert!((survival_s - 0.05).abs() < 1e-12);
    assert!((world.sim_time_s() - 0.06).abs() < 1e-12);
    assert_ne!(survival_s, world.sim_time_s());
    assert_eq!(enemy, EnemyId(0));
    assert_eq!(by, DroneId(0));
    match world.enemies()[0].state {
        EnemyState::Neutralized {
            at_tick,
            survival_s: s,
            ..
        } => {
            assert_eq!(at_tick, 5);
            assert!((s - 0.05).abs() < 1e-12);
        }
        other => panic!("{other:?}"),
    }
    match world.drone(DroneId(0)).unwrap().state {
        DroneState::Expended { at_tick } => assert_eq!(at_tick, 5),
        other => panic!("{other:?}"),
    }
}

#[test]
fn abort_out_of_range_with_long_t_kill() {
    let mut cfg = on_rail_cfg(1);
    cfg.t_kill_s = 10.0;
    let cfg = cfg.validated().unwrap();
    let x0 = -cfg.radius_m;
    let mut world =
        World::with_enemy_positions(cfg, vec![Position { x: x0, y: 0.0 }]).unwrap();
    let mut aborted = false;
    while !world.is_finished() {
        let events = world.step(&NearestInRange);
        if events.iter().any(|e| {
            matches!(
                e,
                SimEvent::EngagementAborted {
                    reason: AbortReason::OutOfRange,
                    ..
                }
            )
        }) {
            aborted = true;
            break;
        }
    }
    assert!(aborted);
    assert!(world.enemies()[0].is_alive());
    assert!(world.drone(DroneId(0)).unwrap().is_live());
}

#[test]
fn smaller_drone_id_wins_same_tick_conflict() {
    let mut cfg = SimConfig::default();
    cfg.radius_m = 40.0;
    cfg.enemy_count = 1;
    cfg.drone_rows = 1;
    cfg.drones_per_row = 2;
    cfg.detection_range_m = 80.0;
    let cfg = cfg.validated().unwrap();
    let y = -cfg.radius_m - cfg.start_margin_m + cfg.speed_m_s * cfg.dt_s;
    let mut world = World::with_enemy_positions(cfg, vec![Position { x: 0.0, y }]).unwrap();
    world.step(&NearestInRange);
    assert_eq!(world.engagement_of(DroneId(0)), Some(EnemyId(0)));
    assert_eq!(world.engagement_of(DroneId(1)), None);
    let mut killer = None;
    while !world.is_finished() {
        let events = world.step(&NearestInRange);
        if let Some(SimEvent::Neutralized { by, .. }) = events
            .iter()
            .find(|e| matches!(e, SimEvent::Neutralized { .. }))
        {
            killer = Some(*by);
            break;
        }
    }
    assert_eq!(killer, Some(DroneId(0)));
}

#[test]
fn expended_drone_stops_deciding_and_cannot_fire_again() {
    let cfg = on_rail_cfg(2).validated().unwrap();
    let x0 = -cfg.radius_m;
    let y = -cfg.radius_m - cfg.start_margin_m + cfg.speed_m_s * cfg.dt_s;
    let positions = vec![
        Position { x: x0, y },
        Position { x: x0, y: y + 30.0 },
    ];
    let mut world = World::with_enemy_positions(cfg, positions).unwrap();
    let mut after_expend_ticks = 0;
    loop {
        let live_before = world.drone(DroneId(0)).unwrap().is_live();
        let events = world.step(&NearestInRange);
        let decisions_0 = events
            .iter()
            .filter(|e| matches!(e, SimEvent::Decision { drone, .. } if *drone == DroneId(0)))
            .count();
        if !live_before {
            assert_eq!(decisions_0, 0);
            after_expend_ticks += 1;
            if after_expend_ticks >= 3 {
                break;
            }
        } else {
            assert_eq!(decisions_0, 1);
        }
        if world.is_finished() {
            break;
        }
    }
    assert!(!world.drone(DroneId(0)).unwrap().is_live());
    let kills_by_0 = world
        .enemies()
        .iter()
        .filter(|e| matches!(e.state, EnemyState::Neutralized { by, .. } if by == DroneId(0)))
        .count();
    assert_eq!(kills_by_0, 1);
}

#[test]
fn locked_drone_still_emits_decision() {
    let cfg = on_rail_cfg(1).validated().unwrap();
    let x0 = -cfg.radius_m;
    let y = -cfg.radius_m - cfg.start_margin_m + cfg.speed_m_s * cfg.dt_s;
    let mut world = World::with_enemy_positions(cfg, vec![Position { x: x0, y }]).unwrap();
    world.step(&NearestInRange);
    assert!(world.engagement_of(DroneId(0)).is_some());
    let lock = world.engagement_of(DroneId(0));
    let events = world.step(&NearestInRange);
    let dec = events
        .iter()
        .find_map(|e| match e {
            SimEvent::Decision {
                drone,
                target,
                compute_ops,
                ..
            } if *drone == DroneId(0) => Some((*target, *compute_ops)),
            _ => None,
        })
        .expect("decision while locked");
    assert!(dec.1 > 0);
    assert_eq!(world.engagement_of(DroneId(0)), lock);
}

fn twelve_on_six_rails() -> (SimConfig, Vec<Position>) {
    let mut cfg = SimConfig::default();
    cfg.enemy_count = 12;
    let cfg = cfg.validated().unwrap();
    let rows = formation_rows(cfg.radius_m, cfg.drones_per_row, cfg.drone_rows);
    let mut positions = Vec::new();
    for x in &rows[0] {
        positions.push(Position { x: *x, y: -20.0 });
    }
    for x in &rows[1] {
        positions.push(Position { x: *x, y: 20.0 });
    }
    (cfg, positions)
}

#[test]
fn twelve_on_six_rails_twelve_kills() {
    let (cfg, positions) = twelve_on_six_rails();
    let mut world = World::with_enemy_positions(cfg, positions).unwrap();
    let mut kills: Vec<(EnemyId, DroneId)> = Vec::new();
    while !world.is_finished() {
        for e in world.step(&NearestInRange) {
            if let SimEvent::Neutralized { enemy, by, .. } = e {
                kills.push((enemy, by));
            }
        }
    }
    assert_eq!(kills.len(), 12);
    assert!(world.drones().iter().all(|d| !d.is_live()));
    assert!(world.enemies().iter().all(|e| !e.is_alive()));
}
