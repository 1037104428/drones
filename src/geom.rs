use rand::Rng;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Position {
    pub x: f64,
    pub y: f64,
}

impl Position {
    pub fn distance(self, other: Position) -> f64 {
        (self.x - other.x).hypot(self.y - other.y)
    }

    pub fn in_disk(self, radius: f64) -> bool {
        self.x * self.x + self.y * self.y <= radius * radius
    }
}

/// Inverse-transform uniform sampling in a closed disk of `radius`.
/// Consumes exactly two `f64` draws; no rejection sampling.
pub fn sample_point_in_disk(rng: &mut impl Rng, radius: f64) -> Position {
    let u: f64 = rng.gen();
    let theta: f64 = rng.gen::<f64>() * std::f64::consts::TAU;
    let r = radius * u.sqrt();
    Position {
        x: r * theta.cos(),
        y: r * theta.sin(),
    }
}

/// Inclusive lerp of `n` x-coordinates across `[-radius, radius]`.
pub fn row_x_positions(radius: f64, n: u32) -> Vec<f64> {
    assert!(n >= 2);
    (0..n)
        .map(|i| {
            let t = i as f64 / (n as f64 - 1.0);
            -radius + t * (2.0 * radius)
        })
        .collect()
}

/// Brick / staggered formation: `rows` ranks share `per_row * rows` distinct
/// rails on `[-R, R]`. Row 0 takes even slots (includes `-R`), row 1 the gaps.
pub fn formation_rows(radius: f64, per_row: u32, rows: u32) -> Vec<Vec<f64>> {
    assert!(per_row >= 2 && rows >= 1);
    if rows == 1 {
        return vec![row_x_positions(radius, per_row)];
    }
    let n_slots = per_row * rows;
    let all = row_x_positions(radius, n_slots);
    let mut out = vec![Vec::new(); rows as usize];
    for (i, x) in all.into_iter().enumerate() {
        out[i % rows as usize].push(x);
    }
    out
}

pub fn all_rail_x(radius: f64, per_row: u32, rows: u32) -> Vec<f64> {
    formation_rows(radius, per_row, rows)
        .into_iter()
        .flatten()
        .collect()
}

/// Inter-rail spacing for an inclusive `n`-drone row spanning the diameter.
pub fn rail_spacing(radius: f64, n: u32) -> f64 {
    2.0 * radius / (n as f64 - 1.0)
}

/// Every point of the closed disk has some rail within `detection` in x
/// (the +Y sweep then covers y). This is the lateral half of
/// "kill rectangle perfectly contains the circle".
pub fn disk_laterally_covered(radius: f64, n_rails: u32, detection: f64) -> bool {
    detection + 1e-12 >= rail_spacing(radius, n_rails) / 2.0
}

/// Sampled check: every in-disk point is within `detection` of some rail.
pub fn every_disk_point_near_a_rail(
    radius: f64,
    n_rails: u32,
    detection: f64,
    samples: usize,
    seed: u64,
) -> bool {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    if !disk_laterally_covered(radius, n_rails, detection) {
        return false;
    }
    let xs = all_rail_x(radius, n_rails, 2); // default stagger coverage uses both rows' rails
    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..samples {
        let p = sample_point_in_disk(&mut rng, radius);
        let covered = xs.iter().any(|&x| (p.x - x).abs() <= detection + 1e-9);
        if !covered {
            return false;
        }
    }
    true
}
