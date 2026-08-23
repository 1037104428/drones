use serde::{Deserialize, Serialize};

use crate::contact::{MAX_DRONE_CONTACTS, MAX_TARGET_CONTACTS};

fn default_expend_on_kill() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SimConfig {
    pub radius_m: f64,
    pub enemy_count: u32,
    pub drone_rows: u32,
    pub drones_per_row: u32,
    pub row_gap_m: f64,
    pub speed_m_s: f64,
    pub dt_s: f64,
    pub detection_range_m: f64,
    pub t_kill_s: f64,
    pub start_margin_m: f64,
    pub end_margin_m: f64,
    pub max_target_contacts: usize,
    pub max_drone_contacts: usize,
    /// v1 必须为 true；serde 缺省为 true。
    #[serde(default = "default_expend_on_kill")]
    pub expend_on_kill: bool,
    /// 若 JSON 省略或为 0，由 `validated()` 按公式填入。
    #[serde(default)]
    pub max_ticks: u64,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            radius_m: 200.0,
            enemy_count: 20,
            drone_rows: 2,
            drones_per_row: 6,
            row_gap_m: 40.0,
            speed_m_s: 25.0,
            dt_s: 0.01,
            // Spacing of 6 rails on 2R is 80 m. D = 80 m makes neighbouring
            // detection disks overlap past the mid-line, so the moving kill
            // band is a solid 2R-wide rectangle that the circle is inscribed in.
            detection_range_m: 80.0,
            t_kill_s: 0.05,
            start_margin_m: 80.0,
            end_margin_m: 80.0,
            max_target_contacts: MAX_TARGET_CONTACTS,
            max_drone_contacts: MAX_DRONE_CONTACTS,
            expend_on_kill: true,
            max_ticks: 0,
        }
    }
}

impl SimConfig {
    pub fn validated(mut self) -> Result<Self, ConfigError> {
        fn require_positive(field: &'static str, value: f64) -> Result<(), ConfigError> {
            if value > 0.0 && value.is_finite() {
                Ok(())
            } else {
                Err(ConfigError::NonPositive { field, value })
            }
        }
        fn require_non_negative(field: &'static str, value: f64) -> Result<(), ConfigError> {
            if value >= 0.0 && value.is_finite() {
                Ok(())
            } else {
                Err(ConfigError::Negative { field, value })
            }
        }

        require_positive("radius_m", self.radius_m)?;
        require_positive("speed_m_s", self.speed_m_s)?;
        require_positive("dt_s", self.dt_s)?;
        require_positive("detection_range_m", self.detection_range_m)?;
        require_positive("t_kill_s", self.t_kill_s)?;
        require_positive("row_gap_m", self.row_gap_m)?;
        require_non_negative("start_margin_m", self.start_margin_m)?;
        require_non_negative("end_margin_m", self.end_margin_m)?;

        if self.enemy_count < 1 {
            return Err(ConfigError::InvalidFormation {
                reason: "enemy_count must be >= 1".into(),
            });
        }
        if self.drone_rows < 1 {
            return Err(ConfigError::InvalidFormation {
                reason: "drone_rows must be >= 1".into(),
            });
        }
        if self.drones_per_row < 2 {
            return Err(ConfigError::InvalidFormation {
                reason: "drones_per_row must be >= 2".into(),
            });
        }
        if self.max_target_contacts != MAX_TARGET_CONTACTS
            || self.max_drone_contacts != MAX_DRONE_CONTACTS
        {
            return Err(ConfigError::ContactCapMismatch {
                expected: MAX_TARGET_CONTACTS,
                targets: self.max_target_contacts,
                drones: self.max_drone_contacts,
            });
        }
        if !self.expend_on_kill {
            return Err(ConfigError::Unsupported {
                flag: "expend_on_kill",
                value: "false".into(),
            });
        }

        if self.max_ticks == 0 {
            let step = self.speed_m_s * self.dt_s;
            let kinematic = (self.path_length_m() / step).ceil() as u64;
            self.max_ticks = kinematic.saturating_add(100);
        } else if self.max_ticks < 1 {
            return Err(ConfigError::InvalidFormation {
                reason: "max_ticks must be >= 1".into(),
            });
        }

        Ok(self)
    }

    pub fn kill_ticks(&self) -> u64 {
        (self.t_kill_s / self.dt_s).ceil() as u64
    }

    pub fn path_length_m(&self) -> f64 {
        (self.radius_m + self.end_margin_m) - (-self.radius_m - self.start_margin_m)
            + self.row_gap_m
    }

    pub fn drone_count(&self) -> u32 {
        self.drone_rows * self.drones_per_row
    }

    /// Kill-rectangle width is the diameter; rails sit on `[-R, R]`.
    pub fn kill_rect_width_m(&self) -> f64 {
        2.0 * self.radius_m
    }

    pub fn rail_spacing_m(&self) -> f64 {
        crate::geom::rail_spacing(self.radius_m, self.drones_per_row)
    }

    /// True when the default +Y sweep's detection band covers every disk point.
    pub fn kill_zone_contains_disk(&self) -> bool {
        crate::geom::disk_laterally_covered(
            self.radius_m,
            self.drones_per_row,
            self.detection_range_m,
        )
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("field {field} must be positive, got {value}")]
    NonPositive { field: &'static str, value: f64 },
    #[error("field {field} must be >= 0, got {value}")]
    Negative { field: &'static str, value: f64 },
    #[error("contact caps must be {expected} (got targets={targets}, drones={drones})")]
    ContactCapMismatch {
        expected: usize,
        targets: usize,
        drones: usize,
    },
    #[error("unsupported v1 flag {flag}={value}")]
    Unsupported { flag: &'static str, value: String },
    #[error("invalid formation: {reason}")]
    InvalidFormation { reason: String },
    #[error("unknown algorithm '{name}'")]
    UnknownAlgorithm { name: String },
}
