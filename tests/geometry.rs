use approx::assert_relative_eq;
use battlefield_sim::config::SimConfig;
use battlefield_sim::geom::{
    disk_laterally_covered, every_disk_point_near_a_rail, formation_rows, rail_spacing,
    row_x_positions, sample_point_in_disk, Position,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn disk_membership_boundaries() {
    let r = 200.0;
    assert!(Position { x: 0.0, y: 0.0 }.in_disk(r));
    assert!(Position { x: r, y: 0.0 }.in_disk(r));
    assert!(!Position {
        x: r + 1e-9,
        y: 0.0
    }
    .in_disk(r));
}

#[test]
fn inverse_transform_sample_is_in_disk_and_spread() {
    let r = 200.0;
    let mut rng = StdRng::seed_from_u64(7);
    let pts: Vec<Position> = (0..200)
        .map(|_| sample_point_in_disk(&mut rng, r))
        .collect();
    for p in &pts {
        assert!(p.in_disk(r), "point ({}, {}) outside disk", p.x, p.y);
    }
    for i in 0..pts.len() {
        for j in (i + 1)..pts.len() {
            assert!(pts[i].distance(pts[j]) > 0.0);
        }
    }
    let mut radii: Vec<f64> = pts.iter().map(|p| p.x.hypot(p.y)).collect();
    radii.sort_by(|a, b| a.total_cmp(b));
    let median = radii[radii.len() / 2];
    assert!(median > 0.3 * r, "median r={median} not > 0.3R");
}

#[test]
fn row_x_positions_six_across_diameter() {
    let xs = row_x_positions(200.0, 6);
    let expect = [-200.0, -120.0, -40.0, 40.0, 120.0, 200.0];
    assert_eq!(xs.len(), expect.len());
    for (a, b) in xs.iter().zip(expect) {
        assert_relative_eq!(*a, b, epsilon = 1e-12);
    }
}

#[test]
fn two_rows_are_staggered_on_twelve_rails() {
    let rows = formation_rows(200.0, 6, 2);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].len(), 6);
    assert_eq!(rows[1].len(), 6);
    assert_relative_eq!(rows[0][0], -200.0, epsilon = 1e-12);
    assert_relative_eq!(rows[1][0], -200.0 + 400.0 / 11.0, epsilon = 1e-9);
    for x0 in &rows[0] {
        for x1 in &rows[1] {
            assert!((x0 - x1).abs() > 1.0, "rows should not share a rail");
        }
    }
}

#[test]
fn gaussian_ingress_places_twelve_separated_drones() {
    let mut cfg = SimConfig::default();
    cfg.ingress = "gaussian".into();
    let cfg = cfg.validated().unwrap();
    let world = battlefield_sim::World::new(cfg, 7).unwrap();
    assert_eq!(world.drones().len(), 12);
    for i in 0..12 {
        for j in (i + 1)..12 {
            assert!(world.drones()[i].pos.distance(world.drones()[j].pos) >= 8.0);
        }
    }
}

#[test]
fn aligned_rows_share_the_same_x() {
    let mut cfg = SimConfig::default();
    cfg.stagger_rows = false;
    let cfg = cfg.validated().unwrap();
    let world = battlefield_sim::World::new(cfg, 1).unwrap();
    let lead: Vec<f64> = world.drones()[..6].iter().map(|d| d.pos.x).collect();
    let trail: Vec<f64> = world.drones()[6..].iter().map(|d| d.pos.x).collect();
    for (a, b) in lead.iter().zip(&trail) {
        assert_relative_eq!(*a, *b, epsilon = 1e-12);
    }
}

#[test]
fn default_path_length_and_max_ticks() {
    let cfg = SimConfig::default().validated().unwrap();
    // 2R + start_margin + end_margin + row_gap = 400+50+50+40 = 540
    assert_relative_eq!(cfg.path_length_m(), 540.0, epsilon = 1e-12);
    assert_eq!(cfg.max_ticks, 2260);
    assert_eq!(cfg.kill_ticks(), 5);
    assert_relative_eq!(cfg.kill_rect_width_m(), 400.0, epsilon = 1e-12);
    assert_relative_eq!(cfg.rail_spacing_m(), 80.0, epsilon = 1e-12);
}

#[test]
fn default_kill_zone_perfectly_contains_disk() {
    let cfg = SimConfig::default().validated().unwrap();
    assert!(cfg.kill_zone_contains_disk());
    assert!(disk_laterally_covered(
        cfg.radius_m,
        cfg.drones_per_row,
        cfg.detection_range_m
    ));
    assert!(every_disk_point_near_a_rail(
        cfg.radius_m,
        cfg.drones_per_row,
        cfg.detection_range_m,
        4000,
        42
    ));
    // Half-spacing = 40 m; D must be at least that.
    assert!(cfg.detection_range_m >= rail_spacing(cfg.radius_m, cfg.drones_per_row) / 2.0);
}

#[test]
fn expend_on_kill_false_is_unsupported() {
    let mut cfg = SimConfig::default();
    cfg.expend_on_kill = false;
    match cfg.validated() {
        Err(battlefield_sim::ConfigError::Unsupported { flag, .. }) => {
            assert_eq!(flag, "expend_on_kill");
        }
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

#[test]
fn partial_json_uses_struct_default() {
    let cfg: SimConfig = serde_json::from_str(r#"{"row_gap_m": 20}"#).unwrap();
    assert_relative_eq!(cfg.row_gap_m, 20.0, epsilon = 1e-12);
    assert_relative_eq!(cfg.radius_m, 200.0, epsilon = 1e-12);
    assert_eq!(cfg.enemy_count, 20);
}

#[test]
fn unknown_json_field_is_rejected() {
    let err = serde_json::from_str::<SimConfig>(r#"{"nope": 1}"#).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("unknown field") || msg.contains("nope"), "{msg}");
}
