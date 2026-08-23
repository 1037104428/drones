//! 1-layer Transformer encoder policy over the 8+8 contact snapshot.
//!
//! Tokens: CLS + 8 enemies + 8 friends. Action space: 8 enemy slots + abstain.
//!
//! Extra `compute_ops` categories (in addition to examining each present contact):
//! - +1 per multiply-add in embed / attention / FFN / output head
//!
//! Training: synthetic CE on the research rule, then 200 live rounds of imitation.
//! Episode RL reward (if used) is **only** `-mean_survival / T_end` — compute
//! time is never a loss term. Eval is greedy (deterministic).

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rand::rngs::StdRng;
use rayon::prelude::*;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::algorithm::{SelectResult, TargetingAlgorithm, TargetingInput};
use crate::contact::{MAX_DRONE_CONTACTS, MAX_TARGET_CONTACTS};
use crate::ids::{DroneId, EnemyId};
use crate::metrics::RoundMetrics;

pub const N_ENEMY: usize = MAX_TARGET_CONTACTS;
pub const N_FRIEND: usize = MAX_DRONE_CONTACTS;
pub const N_TOKEN: usize = 1 + N_ENEMY + N_FRIEND; // CLS + enemies + friends
pub const N_ACT: usize = N_ENEMY + 1; // slots + abstain
const D_IN: usize = 4; // dist_norm, valid, engaged_other, is_friend
const D: usize = 16;
const H: usize = 2;
const DH: usize = D / H;
const FF: usize = 32;


#[derive(Clone, Serialize, Deserialize)]
struct Weights {
    w_in: Vec<f32>,  // D_IN * D
    b_in: Vec<f32>,  // D
    w_pos: Vec<f32>, // N_TOKEN * D
    wq: Vec<f32>,    // D * D
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    w1: Vec<f32>, // D * FF
    b1: Vec<f32>, // FF
    w2: Vec<f32>, // FF * D
    b2: Vec<f32>, // D
    w_out: Vec<f32>, // D * N_ACT
    b_out: Vec<f32>, // N_ACT
}

impl Weights {
    fn zeros() -> Self {
        Self {
            w_in: vec![0.0; D_IN * D],
            b_in: vec![0.0; D],
            w_pos: vec![0.0; N_TOKEN * D],
            wq: vec![0.0; D * D],
            wk: vec![0.0; D * D],
            wv: vec![0.0; D * D],
            wo: vec![0.0; D * D],
            w1: vec![0.0; D * FF],
            b1: vec![0.0; FF],
            w2: vec![0.0; FF * D],
            b2: vec![0.0; D],
            w_out: vec![0.0; D * N_ACT],
            b_out: vec![0.0; N_ACT],
        }
    }

    fn xavier(rng: &mut impl Rng) -> Self {
        fn xv(rng: &mut impl Rng, rows: usize, cols: usize) -> Vec<f32> {
            let s = (6.0 / (rows + cols) as f32).sqrt();
            (0..rows * cols)
                .map(|_| rng.gen::<f32>() * 2.0 * s - s)
                .collect()
        }
        Self {
            w_in: xv(rng, D_IN, D),
            b_in: vec![0.0; D],
            w_pos: xv(rng, N_TOKEN, D),
            wq: xv(rng, D, D),
            wk: xv(rng, D, D),
            wv: xv(rng, D, D),
            wo: xv(rng, D, D),
            w1: xv(rng, D, FF),
            b1: vec![0.0; FF],
            w2: xv(rng, FF, D),
            b2: vec![0.0; D],
            w_out: xv(rng, D, N_ACT),
            b_out: {
                let mut b = vec![0.0; N_ACT];
                // Don't default to always-abstain before any training.
                b[N_ENEMY] = -2.0;
                b
            },
        }
    }

    fn add_from(&mut self, other: &Weights) {
        fn add(a: &mut [f32], b: &[f32]) {
            for (x, y) in a.iter_mut().zip(b) {
                *x += *y;
            }
        }
        add(&mut self.w_in, &other.w_in);
        add(&mut self.b_in, &other.b_in);
        add(&mut self.w_pos, &other.w_pos);
        add(&mut self.wq, &other.wq);
        add(&mut self.wk, &other.wk);
        add(&mut self.wv, &other.wv);
        add(&mut self.wo, &other.wo);
        add(&mut self.w1, &other.w1);
        add(&mut self.b1, &other.b1);
        add(&mut self.w2, &other.w2);
        add(&mut self.b2, &other.b2);
        add(&mut self.w_out, &other.w_out);
        add(&mut self.b_out, &other.b_out);
    }
}

struct Adam {
    m: Weights,
    v: Weights,
    t: i32,
}

impl Adam {
    fn new() -> Self {
        Self {
            m: Weights::zeros(),
            v: Weights::zeros(),
            t: 0,
        }
    }
}

struct Step {
    feat: [f32; N_TOKEN * D_IN],
    valid_enemy: [bool; N_ENEMY],
    action: usize,
    drone: DroneId,
}

pub struct TransformerPolicy {
    name: &'static str,
    w: Mutex<Weights>,
    greedy: bool,
    temperature: f32,
    rng: Mutex<StdRng>,
    train_enabled: Mutex<bool>,
    steps: Mutex<Vec<Step>>,
    adam: Mutex<Adam>,
    baseline: Mutex<f32>,
    teacher_mix: Mutex<f32>,
}

impl TransformerPolicy {
    pub fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15);
        Self {
            name: "transformer",
            w: Mutex::new(Weights::xavier(&mut rng)),
            greedy: true,
            temperature: 1.0,
            rng: Mutex::new(rng),
            train_enabled: Mutex::new(false),
            steps: Mutex::new(Vec::new()),
            adam: Mutex::new(Adam::new()),
            baseline: Mutex::new(0.0),
            teacher_mix: Mutex::new(0.0),
        }
    }

    pub fn default_model_path() -> PathBuf {
        PathBuf::from("models/transformer.json")
    }

    pub fn load_or_new(path: impl AsRef<Path>, seed: u64) -> Self {
        match fs::read_to_string(path.as_ref()) {
            Ok(text) => match serde_json::from_str::<Weights>(&text) {
                Ok(w) => {
                    tracing::info!(path = %path.as_ref().display(), "loaded transformer weights");
                    let p = Self::new(seed);
                    *p.w.lock().unwrap() = w;
                    p
                }
                Err(e) => {
                    tracing::warn!("transformer json parse failed: {e}; using random init");
                    Self::new(seed)
                }
            },
            Err(_) => Self::new(seed),
        }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let text = serde_json::to_string(&*self.w.lock().unwrap()).map_err(|e| e.to_string())?;
        fs::write(path, text).map_err(|e| e.to_string())
    }

    pub fn set_name(&mut self, name: &'static str) {
        self.name = name;
    }

    pub fn set_greedy(&mut self, greedy: bool) {
        self.greedy = greedy;
    }

    pub fn set_temperature(&mut self, t: f32) {
        self.temperature = t.max(0.05);
    }

    /// Probability of cloning `CloserThanFriend` this episode (imitation warm-start).
    pub fn set_teacher_mix(&self, p: f32) {
        *self.teacher_mix.lock().unwrap() = p.clamp(0.0, 1.0);
    }

    /// Offline CE on synthetic 8+8 snapshots labelled by `CloserThanFriend`.
    /// Gives the encoder a fire-vs-hold prior before the 200 live rounds.
    pub fn synthetic_pretrain(&self, steps: u32, lr: f32, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed ^ 0xD1B5_4A32);
        let mut adam = self.adam.lock().unwrap();
        let mut w = self.w.lock().unwrap();
        for k in 0..steps {
            let (feat, valid, action) = if k % 2 == 0 {
                simple_two_distance_example(&mut rng)
            } else {
                random_teacher_example(&mut rng)
            };
            let (logits, cache, _) = forward(&w, &feat, &valid);
            let adv = if action < N_ENEMY { 4.0 } else { 1.0 };
            let mut d = d_nll_logsoftmax(&logits, action, adv);
            let mut g = Weights::zeros();
            backward(&w, &cache, &feat, &valid, &mut d, &mut g);
            clip_grads(&mut g, 1.0);
            adam_step(&mut w, &mut g, &mut adam, lr);
        }
    }

    pub fn begin_episode(&self) {
        *self.train_enabled.lock().unwrap() = true;
        self.steps.lock().unwrap().clear();
    }

    /// Clear the per-tick buffer so this `World::step` is a parallel update batch.
    pub fn begin_step(&self) {
        self.steps.lock().unwrap().clear();
    }

    pub fn end_training(&self) {
        *self.train_enabled.lock().unwrap() = false;
        *self.teacher_mix.lock().unwrap() = 0.0;
        self.steps.lock().unwrap().clear();
    }

    /// One REINFORCE step on **all** drones that acted in the last tick (parallel batch).
    /// `reward` is the shared team signal (mean-survival increment). Killers get `+1`.
    pub fn finish_step(&self, reward: f32, killers: &[DroneId], lr: f32) {
        let steps = {
            let mut g = self.steps.lock().unwrap();
            std::mem::take(&mut *g)
        };
        if steps.is_empty() {
            return;
        }
        let mut baseline = self.baseline.lock().unwrap();
        if *baseline == 0.0 {
            *baseline = reward;
        } else {
            *baseline = 0.95 * *baseline + 0.05 * reward;
        }
        let team_adv = reward - *baseline;
        drop(baseline);

        let n = steps.len() as f32;
        let w = self.w.lock().unwrap().clone();
        let killers = killers.to_vec();
        let parts: Vec<Weights> = steps
            .par_iter()
            .map(|s| {
                let eligible = s.valid_enemy.iter().any(|v| *v);
                let fired = s.action < N_ENEMY
                    && s.valid_enemy.get(s.action).copied().unwrap_or(false);
                let extra = if killers.iter().any(|k| *k == s.drone) {
                    1.5
                } else if fired {
                    // Kill credit arrives only after T_kill; without this,
                    // fire and abstain get the same team r_t and the policy
                    // collapses to always-abstain.
                    0.35
                } else if eligible {
                    -0.2
                } else {
                    0.0
                };
                let adv = team_adv + extra;
                let (logits, cache, _) = forward(&w, &s.feat, &s.valid_enemy);
                let mut d_logits = d_nll_logsoftmax(&logits, s.action, adv);
                add_entropy_grad(&mut d_logits, &logits, 0.03);
                let mut local = Weights::zeros();
                backward(&w, &cache, &s.feat, &s.valid_enemy, &mut d_logits, &mut local);
                local
            })
            .collect();
        let mut g = Weights::zeros();
        for p in &parts {
            g.add_from(p);
        }
        scale_grads(&mut g, 1.0 / n.max(1.0));
        clip_grads(&mut g, 1.0);
        let mut adam = self.adam.lock().unwrap();
        let mut w = self.w.lock().unwrap();
        adam_step(&mut w, &mut g, &mut adam, lr);
    }

    /// REINFORCE update from the just-finished round.
    pub fn finish_episode(&self, metrics: &RoundMetrics, lr: f32) {
        *self.train_enabled.lock().unwrap() = false;
        let steps = {
            let mut g = self.steps.lock().unwrap();
            std::mem::take(&mut *g)
        };
        if steps.is_empty() {
            return;
        }
        // Loss uses only mean target survival (design.txt). Compute time is not a train signal.
        let denom = metrics.sim_duration_s.max(1e-3) as f32;
        let reward = -metrics.mean_survival_s as f32 / denom;
        let mut baseline = self.baseline.lock().unwrap();
        if *baseline == 0.0 {
            *baseline = reward;
        } else {
            *baseline = 0.9 * *baseline + 0.1 * reward;
        }
        let adv = reward - *baseline;
        drop(baseline);

        let n = steps.len();
        let take = n.min(256);
        let mut idx: Vec<usize> = (0..n).collect();
        {
            let mut rng = self.rng.lock().unwrap();
            for i in 0..take {
                let j = rng.gen_range(i..n);
                idx.swap(i, j);
            }
        }
        idx.truncate(take);

        let mut g = Weights::zeros();
        {
            let w = self.w.lock().unwrap();
            for &i in &idx {
                let s = &steps[i];
                let (logits, cache, _) = forward(&w, &s.feat, &s.valid_enemy);
                let mut d_logits = d_nll_logsoftmax(&logits, s.action, adv);
                backward(&w, &cache, &s.feat, &s.valid_enemy, &mut d_logits, &mut g);
            }
        }
        let scale = 1.0 / take as f32;
        scale_grads(&mut g, scale);
        clip_grads(&mut g, 1.0);
        let mut adam = self.adam.lock().unwrap();
        let mut w = self.w.lock().unwrap();
        adam_step(&mut w, &mut g, &mut adam, lr);
    }
}

fn adam_step(w: &mut Weights, g: &mut Weights, opt: &mut Adam, lr: f32) {
    opt.t += 1;
    let t = opt.t as f32;
    let b1 = 0.9f32;
    let b2 = 0.999f32;
    let bc1 = 1.0 - b1.powf(t);
    let bc2 = 1.0 - b2.powf(t);
    adam_vec(&mut w.w_in, &g.w_in, &mut opt.m.w_in, &mut opt.v.w_in, lr, b1, b2, bc1, bc2);
    adam_vec(&mut w.b_in, &g.b_in, &mut opt.m.b_in, &mut opt.v.b_in, lr, b1, b2, bc1, bc2);
    adam_vec(&mut w.w_pos, &g.w_pos, &mut opt.m.w_pos, &mut opt.v.w_pos, lr, b1, b2, bc1, bc2);
    adam_vec(&mut w.wq, &g.wq, &mut opt.m.wq, &mut opt.v.wq, lr, b1, b2, bc1, bc2);
    adam_vec(&mut w.wk, &g.wk, &mut opt.m.wk, &mut opt.v.wk, lr, b1, b2, bc1, bc2);
    adam_vec(&mut w.wv, &g.wv, &mut opt.m.wv, &mut opt.v.wv, lr, b1, b2, bc1, bc2);
    adam_vec(&mut w.wo, &g.wo, &mut opt.m.wo, &mut opt.v.wo, lr, b1, b2, bc1, bc2);
    adam_vec(&mut w.w1, &g.w1, &mut opt.m.w1, &mut opt.v.w1, lr, b1, b2, bc1, bc2);
    adam_vec(&mut w.b1, &g.b1, &mut opt.m.b1, &mut opt.v.b1, lr, b1, b2, bc1, bc2);
    adam_vec(&mut w.w2, &g.w2, &mut opt.m.w2, &mut opt.v.w2, lr, b1, b2, bc1, bc2);
    adam_vec(&mut w.b2, &g.b2, &mut opt.m.b2, &mut opt.v.b2, lr, b1, b2, bc1, bc2);
    adam_vec(&mut w.w_out, &g.w_out, &mut opt.m.w_out, &mut opt.v.w_out, lr, b1, b2, bc1, bc2);
    adam_vec(&mut w.b_out, &g.b_out, &mut opt.m.b_out, &mut opt.v.b_out, lr, b1, b2, bc1, bc2);
}

fn adam_vec(
    w: &mut [f32],
    g: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    lr: f32,
    b1: f32,
    b2: f32,
    bc1: f32,
    bc2: f32,
) {
    for i in 0..w.len() {
        m[i] = b1 * m[i] + (1.0 - b1) * g[i];
        v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
        let mh = m[i] / bc1;
        let vh = v[i] / bc2;
        w[i] -= lr * mh / (vh.sqrt() + 1e-8);
    }
}

fn scale_grads(g: &mut Weights, s: f32) {
    for v in [
        g.w_in.as_mut_slice(),
        g.b_in.as_mut_slice(),
        g.w_pos.as_mut_slice(),
        g.wq.as_mut_slice(),
        g.wk.as_mut_slice(),
        g.wv.as_mut_slice(),
        g.wo.as_mut_slice(),
        g.w1.as_mut_slice(),
        g.b1.as_mut_slice(),
        g.w2.as_mut_slice(),
        g.b2.as_mut_slice(),
        g.w_out.as_mut_slice(),
        g.b_out.as_mut_slice(),
    ] {
        for x in v.iter_mut() {
            *x *= s;
        }
    }
}

fn clip_grads(g: &mut Weights, max_norm: f32) {
    let mut ss = 0.0f32;
    for v in [
        g.w_in.as_slice(),
        g.b_in.as_slice(),
        g.w_pos.as_slice(),
        g.wq.as_slice(),
        g.wk.as_slice(),
        g.wv.as_slice(),
        g.wo.as_slice(),
        g.w1.as_slice(),
        g.b1.as_slice(),
        g.w2.as_slice(),
        g.b2.as_slice(),
        g.w_out.as_slice(),
        g.b_out.as_slice(),
    ] {
        for x in v {
            ss += *x * *x;
        }
    }
    let n = ss.sqrt();
    if n > max_norm && n > 0.0 {
        scale_grads(g, max_norm / n);
    }
}

struct Cache {
    x: Vec<f32>,      // [T, D] after embed+pos
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    _scores: Vec<f32>, // [H, T, T]
    attn: Vec<f32>,    // [H, T, T]
    ctx: Vec<f32>,     // [T, D] after concat heads
    _attn_out: Vec<f32>,
    y1: Vec<f32>, // residual after attn
    h: Vec<f32>,  // [T, FF] pre-relu
    h_relu: Vec<f32>,
    y2: Vec<f32>, // residual after ffn
    _logits: Vec<f32>,
}

fn gemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, ops: &mut u64) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let ap = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += ap * b[p * n + j];
                *ops += 1;
            }
        }
    }
    c
}

fn add_row_bias(m: &mut [f32], rows: usize, cols: usize, b: &[f32]) {
    for i in 0..rows {
        for j in 0..cols {
            m[i * cols + j] += b[j];
        }
    }
}

fn add_inplace(a: &mut [f32], b: &[f32]) {
    for i in 0..a.len() {
        a[i] += b[i];
    }
}

fn forward(w: &Weights, feat: &[f32], valid_enemy: &[bool; N_ENEMY]) -> (Vec<f32>, Cache, u64) {
    let mut ops = 0u64;
    // embed: feat [T, D_IN] @ w_in [D_IN, D] + b + pos
    let mut x = gemm(feat, &w.w_in, N_TOKEN, D_IN, D, &mut ops);
    add_row_bias(&mut x, N_TOKEN, D, &w.b_in);
    add_inplace(&mut x, &w.w_pos);

    let q = gemm(&x, &w.wq, N_TOKEN, D, D, &mut ops);
    let k = gemm(&x, &w.wk, N_TOKEN, D, D, &mut ops);
    let v = gemm(&x, &w.wv, N_TOKEN, D, D, &mut ops);

    let mut scores = vec![0.0f32; H * N_TOKEN * N_TOKEN];
    let scale = (DH as f32).sqrt();
    for h in 0..H {
        for i in 0..N_TOKEN {
            for j in 0..N_TOKEN {
                let mut s = 0.0f32;
                for d in 0..DH {
                    s += q[i * D + h * DH + d] * k[j * D + h * DH + d];
                    ops += 1;
                }
                scores[(h * N_TOKEN + i) * N_TOKEN + j] = s / scale;
            }
        }
    }
    let mut attn = vec![0.0f32; scores.len()];
    for h in 0..H {
        for i in 0..N_TOKEN {
            let row = h * N_TOKEN * N_TOKEN + i * N_TOKEN;
            let mut m = scores[row];
            for j in 1..N_TOKEN {
                if scores[row + j] > m {
                    m = scores[row + j];
                }
            }
            let mut z = 0.0f32;
            for j in 0..N_TOKEN {
                let e = (scores[row + j] - m).exp();
                attn[row + j] = e;
                z += e;
            }
            let inv = 1.0 / z.max(1e-12);
            for j in 0..N_TOKEN {
                attn[row + j] *= inv;
            }
        }
    }

    let mut ctx = vec![0.0f32; N_TOKEN * D];
    for h in 0..H {
        for i in 0..N_TOKEN {
            for d in 0..DH {
                let mut s = 0.0f32;
                for j in 0..N_TOKEN {
                    s += attn[(h * N_TOKEN + i) * N_TOKEN + j] * v[j * D + h * DH + d];
                    ops += 1;
                }
                ctx[i * D + h * DH + d] = s;
            }
        }
    }
    let attn_out = gemm(&ctx, &w.wo, N_TOKEN, D, D, &mut ops);
    let mut y1 = x.clone();
    add_inplace(&mut y1, &attn_out);

    let mut hlin = gemm(&y1, &w.w1, N_TOKEN, D, FF, &mut ops);
    add_row_bias(&mut hlin, N_TOKEN, FF, &w.b1);
    let mut h_relu = hlin.clone();
    for x in h_relu.iter_mut() {
        if *x < 0.0 {
            *x = 0.0;
        }
    }
    let ffn = gemm(&h_relu, &w.w2, N_TOKEN, FF, D, &mut ops);
    let mut y2 = y1.clone();
    add_inplace(&mut y2, &ffn);
    add_row_bias(&mut y2, N_TOKEN, D, &w.b2);

    // CLS token -> logits
    let cls = &y2[0..D];
    let mut logits = gemm(cls, &w.w_out, 1, D, N_ACT, &mut ops);
    for a in 0..N_ACT {
        logits[a] += w.b_out[a];
    }
    for i in 0..N_ENEMY {
        if !valid_enemy[i] {
            logits[i] = -1e9;
        }
    }

    (
        logits.clone(),
        Cache {
            x,
            q,
            k,
            v,
            _scores: scores,
            attn,
            ctx,
            _attn_out: attn_out,
            y1,
            h: hlin,
            h_relu,
            y2,
            _logits: logits,
        },
        ops,
    )
}

fn d_nll_logsoftmax(logits: &[f32], action: usize, adv: f32) -> Vec<f32> {
    // loss = -adv * log softmax[action]
    // dL/dlogit_i = -adv * (1[i=a] - p_i) wait:
    // L = -adv * (logit_a - logsumexp)
    // dL/dlogit_i = -adv * (delta_ia - p_i)
    let mut m = logits[0];
    for &x in logits.iter().skip(1) {
        if x > m {
            m = x;
        }
    }
    let mut z = 0.0f32;
    let mut p = vec![0.0f32; logits.len()];
    for i in 0..logits.len() {
        p[i] = (logits[i] - m).exp();
        z += p[i];
    }
    for pi in p.iter_mut() {
        *pi /= z.max(1e-12);
    }
    let mut d = vec![0.0f32; logits.len()];
    for i in 0..logits.len() {
        let delta = if i == action { 1.0 } else { 0.0 };
        d[i] = -adv * (delta - p[i]);
    }
    d
}

/// Maximize entropy: L += -β H, so g += β p (log p + H).
fn add_entropy_grad(d: &mut [f32], logits: &[f32], beta: f32) {
    if beta <= 0.0 {
        return;
    }
    let mut m = logits[0];
    for &x in logits.iter().skip(1) {
        if x > m {
            m = x;
        }
    }
    let mut p = vec![0.0f32; logits.len()];
    let mut z = 0.0f32;
    for i in 0..logits.len() {
        p[i] = (logits[i] - m).exp();
        z += p[i];
    }
    let inv = 1.0 / z.max(1e-12);
    let mut h = 0.0f32;
    for pi in p.iter_mut() {
        *pi *= inv;
        if *pi > 1e-12 {
            h -= *pi * pi.ln();
        }
    }
    for i in 0..d.len() {
        if p[i] > 1e-12 {
            d[i] += beta * p[i] * (p[i].ln() + h);
        }
    }
}

fn gemm_da(a: &[f32], b: &[f32], dc: &[f32], m: usize, k: usize, n: usize, da: &mut [f32], db: &mut [f32]) {
    // C = A[m,k] @ B[k,n]
    // dA = dC @ B^T ; dB = A^T @ dC
    for i in 0..m {
        for p in 0..k {
            let mut s = 0.0;
            for j in 0..n {
                s += dc[i * n + j] * b[p * n + j];
            }
            da[i * k + p] += s;
        }
    }
    for p in 0..k {
        for j in 0..n {
            let mut s = 0.0;
            for i in 0..m {
                s += a[i * k + p] * dc[i * n + j];
            }
            db[p * n + j] += s;
        }
    }
}

fn backward(
    w: &Weights,
    cache: &Cache,
    feat: &[f32],
    valid_enemy: &[bool; N_ENEMY],
    d_logits: &mut [f32],
    g: &mut Weights,
) {
    for i in 0..N_ENEMY {
        if !valid_enemy[i] {
            d_logits[i] = 0.0;
        }
    }
    // logits = cls @ w_out + b_out
    let cls = &cache.y2[0..D];
    let mut d_y2 = vec![0.0f32; N_TOKEN * D];
    // d w_out, d b_out, d cls
    for a in 0..N_ACT {
        g.b_out[a] += d_logits[a];
        for d in 0..D {
            g.w_out[d * N_ACT + a] += cls[d] * d_logits[a];
            d_y2[d] += w.w_out[d * N_ACT + a] * d_logits[a];
        }
    }
    // y2 = y1 + h_relu @ w2 + b2  (b2 added as row bias on all tokens; CLS uses y2[0])
    for d in 0..D {
        g.b2[d] += d_y2[d];
    }
    let mut d_hrelu = vec![0.0f32; N_TOKEN * FF];
    let mut d_y1 = vec![0.0f32; N_TOKEN * D];
    // Only CLS row of ffn contributes through y2[0]; other tokens of y2 have zero d unless we
    // also flow residual from y1 which is used only via CLS after ffn. Attention still needs
    // all tokens because CLS attends to them. So we must backprop through full y1 via attn.
    // d_y2 is only nonzero on token 0. Residual: d_y1 += d_y2; d_ffn = d_y2 on token 0.
    for i in 0..D {
        d_y1[i] += d_y2[i];
    }
    // d (h_relu @ w2) for token 0: h_relu is [T, FF], but only row 0 of output matters
    // ffn = h_relu @ w2  => d_hrelu[0] and d_w2 from d_y2[0]
    gemm_da(
        &cache.h_relu,
        &w.w2,
        &{
            let mut d_ffn = vec![0.0f32; N_TOKEN * D];
            d_ffn[0..D].copy_from_slice(&d_y2[0..D]);
            d_ffn
        },
        N_TOKEN,
        FF,
        D,
        &mut d_hrelu,
        &mut g.w2,
    );
    for t in 0..N_TOKEN {
        for f in 0..FF {
            if cache.h[t * FF + f] <= 0.0 {
                d_hrelu[t * FF + f] = 0.0;
            }
        }
    }
    // h = y1 @ w1 + b1
    let mut d_y1_from_ff = vec![0.0f32; N_TOKEN * D];
    gemm_da(
        &cache.y1,
        &w.w1,
        &d_hrelu,
        N_TOKEN,
        D,
        FF,
        &mut d_y1_from_ff,
        &mut g.w1,
    );
    for t in 0..N_TOKEN {
        for f in 0..FF {
            g.b1[f] += d_hrelu[t * FF + f];
        }
    }
    add_inplace(&mut d_y1, &d_y1_from_ff);

    // y1 = x + attn_out; attn_out = ctx @ wo
    let mut d_x = d_y1.clone();
    let mut d_ctx = vec![0.0f32; N_TOKEN * D];
    gemm_da(
        &cache.ctx,
        &w.wo,
        &d_y1,
        N_TOKEN,
        D,
        D,
        &mut d_ctx,
        &mut g.wo,
    );

    // ctx from attn @ v per head
    let mut d_attn = vec![0.0f32; H * N_TOKEN * N_TOKEN];
    let mut d_v = vec![0.0f32; N_TOKEN * D];
    for h in 0..H {
        for i in 0..N_TOKEN {
            for d in 0..DH {
                let gctx = d_ctx[i * D + h * DH + d];
                for j in 0..N_TOKEN {
                    d_attn[(h * N_TOKEN + i) * N_TOKEN + j] += gctx * cache.v[j * D + h * DH + d];
                    d_v[j * D + h * DH + d] += gctx * cache.attn[(h * N_TOKEN + i) * N_TOKEN + j];
                }
            }
        }
    }
    // softmax backward per row
    let mut d_scores = vec![0.0f32; d_attn.len()];
    for h in 0..H {
        for i in 0..N_TOKEN {
            let row = (h * N_TOKEN + i) * N_TOKEN;
            let mut sp = 0.0f32;
            for j in 0..N_TOKEN {
                sp += d_attn[row + j] * cache.attn[row + j];
            }
            for j in 0..N_TOKEN {
                d_scores[row + j] = cache.attn[row + j] * (d_attn[row + j] - sp);
            }
        }
    }
    let scale = (DH as f32).sqrt();
    let mut d_q = vec![0.0f32; N_TOKEN * D];
    let mut d_k = vec![0.0f32; N_TOKEN * D];
    for h in 0..H {
        for i in 0..N_TOKEN {
            for j in 0..N_TOKEN {
                let ds = d_scores[(h * N_TOKEN + i) * N_TOKEN + j] / scale;
                for d in 0..DH {
                    d_q[i * D + h * DH + d] += ds * cache.k[j * D + h * DH + d];
                    d_k[j * D + h * DH + d] += ds * cache.q[i * D + h * DH + d];
                }
            }
        }
    }

    gemm_da(&cache.x, &w.wq, &d_q, N_TOKEN, D, D, &mut d_x, &mut g.wq);
    gemm_da(&cache.x, &w.wk, &d_k, N_TOKEN, D, D, &mut d_x, &mut g.wk);
    gemm_da(&cache.x, &w.wv, &d_v, N_TOKEN, D, D, &mut d_x, &mut g.wv);

    // x = feat @ w_in + b_in + pos
    let dummy_feat_grad = &mut vec![0.0f32; N_TOKEN * D_IN];
    gemm_da(feat, &w.w_in, &d_x, N_TOKEN, D_IN, D, dummy_feat_grad, &mut g.w_in);
    for t in 0..N_TOKEN {
        for d in 0..D {
            g.b_in[d] += d_x[t * D + d];
            g.w_pos[t * D + d] += d_x[t * D + d];
        }
    }
}

fn simple_two_distance_example(rng: &mut impl Rng) -> ([f32; N_TOKEN * D_IN], [bool; N_ENEMY], usize) {
    let mut feat = [0.0f32; N_TOKEN * D_IN];
    let mut valid = [false; N_ENEMY];
    feat[1] = 1.0;
    let d_t = 0.05 + rng.gen::<f32>() * 0.9; // in range
    let d_f = 0.05 + rng.gen::<f32>() * 1.2;
    feat[D_IN] = d_t;
    feat[D_IN + 1] = 1.0;
    valid[0] = true;
    feat[(1 + N_ENEMY) * D_IN] = d_f;
    feat[(1 + N_ENEMY) * D_IN + 1] = 1.0;
    feat[(1 + N_ENEMY) * D_IN + 3] = 1.0;
    let action = if d_t < d_f { 0 } else { N_ENEMY };
    (feat, valid, action)
}

fn random_teacher_example(rng: &mut impl Rng) -> ([f32; N_TOKEN * D_IN], [bool; N_ENEMY], usize) {
    let mut feat = [0.0f32; N_TOKEN * D_IN];
    let mut valid = [false; N_ENEMY];
    feat[1] = 1.0; // CLS valid
    let n_t = rng.gen_range(1..=N_ENEMY);
    let n_f = rng.gen_range(0..=N_FRIEND);
    let mut min_t = f32::INFINITY;
    let mut best = 0usize;
    for i in 0..n_t {
        let d = rng.gen::<f32>() * 1.4; // dist / D, in-range if <= 1
        let engaged_other = if rng.gen::<f32>() < 0.1 { 1.0 } else { 0.0 };
        let t = 1 + i;
        let base = t * D_IN;
        feat[base] = d;
        feat[base + 1] = 1.0;
        feat[base + 2] = engaged_other;
        valid[i] = true;
        let eligible = d <= 1.0 && engaged_other < 0.5;
        if eligible && d < min_t {
            min_t = d;
            best = i;
        }
    }
    let mut min_f = f32::INFINITY;
    for i in 0..n_f {
        let d = rng.gen::<f32>() * 1.4;
        let t = 1 + N_ENEMY + i;
        let base = t * D_IN;
        feat[base] = d;
        feat[base + 1] = 1.0;
        feat[base + 3] = 1.0;
        if d < min_f {
            min_f = d;
        }
    }
    let action = if min_t < min_f { best } else { N_ENEMY };
    (feat, valid, action)
}

fn encode(input: &TargetingInput<'_>) -> ([f32; N_TOKEN * D_IN], [bool; N_ENEMY], [Option<EnemyId>; N_ENEMY], u64) {
    let mut feat = [0.0f32; N_TOKEN * D_IN];
    let mut valid = [false; N_ENEMY];
    let mut ids = [None; N_ENEMY];
    let mut ops = 0u64;
    let range = input.detection_range.max(1e-6);

    // CLS
    feat[1] = 1.0; // valid

    for (i, c) in input.targets.iter().take(N_ENEMY).enumerate() {
        ops += 1;
        let t = 1 + i;
        let base = t * D_IN;
        feat[base] = (c.distance / range) as f32;
        feat[base + 1] = 1.0;
        let engaged_other = match c.engaged_by {
            Some(owner) if owner != input.self_id => 1.0,
            _ => 0.0,
        };
        feat[base + 2] = engaged_other;
        feat[base + 3] = 0.0;
        valid[i] = true;
        ids[i] = Some(c.id);
    }
    for (i, c) in input.drones.iter().take(N_FRIEND).enumerate() {
        ops += 1;
        let t = 1 + N_ENEMY + i;
        let base = t * D_IN;
        feat[base] = (c.distance / range) as f32;
        feat[base + 1] = 1.0;
        feat[base + 2] = 0.0;
        feat[base + 3] = 1.0;
    }
    (feat, valid, ids, ops)
}

fn logsoftmax_sample(logits: &[f32], rng: &mut impl Rng, temperature: f32) -> usize {
    let t = temperature.max(1e-3);
    let mut scaled: Vec<f32> = logits.iter().map(|x| x / t).collect();
    let mut m = scaled[0];
    for &x in &scaled {
        if x > m {
            m = x;
        }
    }
    let mut z = 0.0f32;
    for x in scaled.iter_mut() {
        *x = (*x - m).exp();
        z += *x;
    }
    let u = rng.gen::<f32>() * z.max(1e-12);
    let mut acc = 0.0f32;
    for (i, p) in scaled.iter().enumerate() {
        acc += *p;
        if u <= acc {
            return i;
        }
    }
    logits.len() - 1
}

fn argmax(logits: &[f32]) -> usize {
    let mut b = 0;
    for i in 1..logits.len() {
        if logits[i] > logits[b] {
            b = i;
        }
    }
    b
}

impl TargetingAlgorithm for TransformerPolicy {
    fn name(&self) -> &'static str {
        self.name
    }

    fn select(&self, input: &TargetingInput<'_>) -> SelectResult {
        let (feat, valid, ids, enc_ops) = encode(input);
        let (logits, _cache, fwd_ops) = {
            let w = self.w.lock().unwrap();
            forward(&w, &feat, &valid)
        };
        let mut action = if self.greedy {
            argmax(&logits)
        } else {
            let mut rng = self.rng.lock().unwrap();
            logsoftmax_sample(&logits, &mut *rng, self.temperature)
        };
        if action < N_ENEMY && !valid[action] {
            action = N_ENEMY; // abstain
        }
        let any_eligible = input.targets.iter().any(|c| {
            c.distance <= input.detection_range
                && match c.engaged_by {
                    None => true,
                    Some(owner) => owner == input.self_id,
                }
        });
        if *self.train_enabled.lock().unwrap()
            && input.current_engagement.is_none()
            && any_eligible
        {
            self.steps.lock().unwrap().push(Step {
                feat,
                valid_enemy: valid,
                action,
                drone: input.self_id,
            });
        }
        let target = if action < N_ENEMY { ids[action] } else { None };
        SelectResult {
            target,
            compute_ops: enc_ops + fwd_ops + 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_masks_invalid_slots() {
        let mut rng = StdRng::seed_from_u64(1);
        let w = Weights::xavier(&mut rng);
        let feat = [0.0f32; N_TOKEN * D_IN];
        let valid = [false; N_ENEMY];
        let (logits, _, ops) = forward(&w, &feat, &valid);
        for i in 0..N_ENEMY {
            assert!(logits[i] < -1e8, "slot {i} should be masked");
        }
        assert!(logits[N_ENEMY] > -1e8);
        assert!(ops > 0);
    }

    #[test]
    fn supervised_ce_fits_fire_slot_zero() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut w = Weights::xavier(&mut rng);
        let mut feat = [0.0f32; N_TOKEN * D_IN];
        feat[1] = 1.0;
        feat[D_IN] = 0.4;
        feat[D_IN + 1] = 1.0;
        let mut valid = [false; N_ENEMY];
        valid[0] = true;
        let mut adam = Adam::new();
        for _ in 0..250 {
            let (logits, cache, _) = forward(&w, &feat, &valid);
            let mut d = d_nll_logsoftmax(&logits, 0, 1.0);
            let mut g = Weights::zeros();
            backward(&w, &cache, &feat, &valid, &mut d, &mut g);
            adam_step(&mut w, &mut g, &mut adam, 0.05);
        }
        let (logits, _, _) = forward(&w, &feat, &valid);
        assert_eq!(
            argmax(&logits),
            0,
            "expected slot 0, logits={logits:?}"
        );
    }

    #[test]
    fn synthetic_pretrain_learns_to_fire_when_target_closer() {
        let mut p = TransformerPolicy::new(7);
        p.synthetic_pretrain(4000, 0.03, 7);
        p.set_greedy(true);
        let t = [crate::contact::TargetContact {
            id: crate::ids::EnemyId(1),
            distance: 10.0,
            pos: crate::geom::Position { x: 10.0, y: 0.0 },
            engaged_by: None,
        }];
        let f = [crate::contact::DroneContact {
            id: crate::ids::DroneId(2),
            distance: 40.0,
            pos: crate::geom::Position { x: 0.0, y: 40.0 },
        }];
        let inp = TargetingInput {
            self_id: crate::ids::DroneId(0),
            self_pos: crate::geom::Position { x: 0.0, y: 0.0 },
            tick: 0,
            detection_range: 80.0,
            targets: &t,
            drones: &f,
            current_engagement: None,
        };
        let r = p.select(&inp);
        assert_eq!(r.target, Some(crate::ids::EnemyId(1)), "pretrained net should fire");
    }
}
