use std::fs;
use std::path::Path;

use crate::metrics::RoundMetrics;
use crate::persist::RunError;

#[derive(Clone, Copy)]
struct BoxStats {
    min: f64,
    q1: f64,
    median: f64,
    q3: f64,
    max: f64,
    mean: f64,
}

fn stats(values: &[f64]) -> BoxStats {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    let mean = v.iter().sum::<f64>() / n.max(1) as f64;
    let at = |p: f64| {
        if n == 0 {
            return 0.0;
        }
        let idx = ((p * (n as f64 - 1.0)).round() as usize).min(n - 1);
        v[idx]
    };
    BoxStats {
        min: *v.first().unwrap_or(&0.0),
        q1: at(0.25),
        median: at(0.5),
        q3: at(0.75),
        max: *v.last().unwrap_or(&0.0),
        mean,
    }
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

pub fn write_compare_plots(
    transformer: &[RoundMetrics],
    research: &[RoundMetrics],
    out_dir: &Path,
) -> Result<(), RunError> {
    let plots = out_dir.join("plots");
    fs::create_dir_all(&plots).map_err(|e| RunError::Invariant(e.to_string()))?;

    let t_ops: Vec<f64> = transformer
        .iter()
        .map(|m| m.mean_compute_ops * m.decision_count as f64)
        .collect();
    let r_ops: Vec<f64> = research
        .iter()
        .map(|m| m.mean_compute_ops * m.decision_count as f64)
        .collect();
    let t_mean_ops: Vec<f64> = transformer.iter().map(|m| m.mean_compute_ops).collect();
    let r_mean_ops: Vec<f64> = research.iter().map(|m| m.mean_compute_ops).collect();
    let t_surv: Vec<f64> = transformer.iter().map(|m| m.mean_survival_s).collect();
    let r_surv: Vec<f64> = research.iter().map(|m| m.mean_survival_s).collect();

    let ops_svg = boxplot_svg(
        "算法耗时对比（每轮总 compute_ops）",
        "总 compute_ops / 轮（对数纵轴）",
        &[("Transformer", &t_ops), ("closer_than_friend", &r_ops)],
        true,
        Some((
            "每次 select 平均 compute_ops",
            &[
                ("Transformer", &t_mean_ops),
                ("closer_than_friend", &r_mean_ops),
            ],
        )),
    );
    fs::write(plots.join("compute_ops_compare.svg"), ops_svg)
        .map_err(|e| RunError::Invariant(e.to_string()))?;

    let surv_svg = histogram_svg(
        "目标平均存活时间分布（30 轮评估）",
        "mean_survival_s（秒，含 T_end 删失）",
        &[
            ("Transformer", &t_surv),
            ("closer_than_friend", &r_surv),
        ],
    );
    fs::write(plots.join("survival_dist_compare.svg"), surv_svg)
        .map_err(|e| RunError::Invariant(e.to_string()))?;

    Ok(())
}

fn boxplot_svg(
    title: &str,
    ylabel: &str,
    series: &[(&str, &[f64])],
    log_y: bool,
    inset: Option<(&str, &[(&str, &[f64])])>,
) -> String {
    let width = 900.0;
    let height = 560.0;
    let left = 80.0;
    let right = 40.0;
    let top = 50.0;
    let bottom = 70.0;
    let plot_w = width - left - right;
    let plot_h = height - top - bottom;

    let mut all = Vec::new();
    for (_, v) in series {
        all.extend_from_slice(v);
    }
    let (y_min, y_max) = if log_y {
        let mn = all.iter().cloned().fold(f64::INFINITY, f64::min).max(1.0);
        let mx = all.iter().cloned().fold(1.0, f64::max);
        (mn / 1.2, mx * 1.2)
    } else {
        let mn = all.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = all.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let pad = (mx - mn).abs() * 0.1 + 1e-6;
        (mn - pad, mx + pad)
    };

    let ymap = |y: f64| -> f64 {
        if log_y {
            let a = y_min.ln();
            let b = y_max.ln();
            top + plot_h * (1.0 - (y.max(y_min).ln() - a) / (b - a))
        } else {
            top + plot_h * (1.0 - (y - y_min) / (y_max - y_min))
        }
    };

    let mut body = String::new();
    body.push_str(&format!(
        "<rect x='{left}' y='{top}' width='{plot_w}' height='{plot_h}' fill='white' stroke='black'/>"
    ));

    let n = series.len().max(1);
    let slot = plot_w / n as f64;
    let colors = ["#2563eb", "#dc2626", "#16a34a"];
    for (i, (name, vals)) in series.iter().enumerate() {
        let st = stats(vals);
        let cx = left + slot * (i as f64 + 0.5);
        let box_w = slot * 0.35;
        let color = colors[i % colors.len()];
        let y_whisker_lo = ymap(st.min);
        let y_whisker_hi = ymap(st.max);
        let y_q1 = ymap(st.q1);
        let y_q3 = ymap(st.q3);
        let y_med = ymap(st.median);
        let y_mean = ymap(st.mean);
        let box_top = y_q3.min(y_q1);
        let box_h = (y_q1 - y_q3).abs().max(1.0);
        let x0 = cx - box_w / 2.0;
        let x1 = cx + box_w / 2.0;
        body.push_str(&format!(
            "<line x1='{cx}' y1='{y_whisker_lo}' x2='{cx}' y2='{y_whisker_hi}' stroke='{color}' stroke-width='1.5'/>"
        ));
        body.push_str(&format!(
            "<rect x='{x0}' y='{box_top}' width='{box_w}' height='{box_h}' fill='{color}' fill-opacity='0.25' stroke='{color}' stroke-width='2'/>"
        ));
        body.push_str(&format!(
            "<line x1='{x0}' y1='{y_med}' x2='{x1}' y2='{y_med}' stroke='black' stroke-width='2'/>"
        ));
        body.push_str(&format!(
            "<circle cx='{cx}' cy='{y_mean}' r='4' fill='{color}'/>"
        ));
        for (k, val) in vals.iter().enumerate() {
            let dx = ((k as f64) * 17.0) % (box_w * 0.6) - box_w * 0.3;
            let px = cx + dx;
            let py = ymap(*val);
            body.push_str(&format!(
                "<circle cx='{px}' cy='{py}' r='2.2' fill='{color}' opacity='0.55'/>"
            ));
        }
        let ly = top + plot_h + 22.0;
        let ly2 = top + plot_h + 40.0;
        body.push_str(&format!(
            "<text x='{cx}' y='{ly}' text-anchor='middle' font-size='13' font-family='sans serif'>{}</text>",
            xml_escape(name)
        ));
        body.push_str(&format!(
            "<text x='{cx}' y='{ly2}' text-anchor='middle' font-size='11' fill='#444' font-family='sans serif'>mean={:.3}</text>",
            st.mean
        ));
    }

    let ticks = 5;
    for t in 0..=ticks {
        let frac = t as f64 / ticks as f64;
        let yv = if log_y {
            (y_min.ln() + frac * (y_max.ln() - y_min.ln())).exp()
        } else {
            y_min + frac * (y_max - y_min)
        };
        let y = ymap(yv);
        let x2 = left + plot_w;
        let tx = left - 8.0;
        let ty = y + 4.0;
        body.push_str(&format!(
            "<line x1='{left}' y1='{y}' x2='{x2}' y2='{y}' stroke='#eee'/>"
        ));
        body.push_str(&format!(
            "<text x='{tx}' y='{ty}' text-anchor='end' font-size='11' font-family='sans serif'>{yv:.4}</text>"
        ));
    }

    if let Some((inset_title, inset_series)) = inset {
        let mut line = format!(
            "<text x='{left}' y='24' font-size='12' font-family='sans serif' fill='#555'>{}: ",
            xml_escape(inset_title)
        );
        for (name, vals) in inset_series.iter() {
            let m = vals.iter().sum::<f64>() / vals.len().max(1) as f64;
            line.push_str(&format!("{}={:.2}  ", xml_escape(name), m));
        }
        line.push_str("</text>");
        body.push_str(&line);
    }

    let cx_title = width / 2.0;
    let ylab_y = top + plot_h / 2.0;
    format!(
        "<?xml version='1.0' encoding='UTF-8'?>\n\
<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>\n\
  <rect width='100%' height='100%' fill='#fafafa'/>\n\
  <text x='{cx_title}' y='28' text-anchor='middle' font-size='18' font-family='sans serif' font-weight='700'>{}</text>\n\
  <text x='18' y='{ylab_y}' transform='rotate(-90 18 {ylab_y})' font-size='12' font-family='sans serif'>{}</text>\n\
  {body}\n\
</svg>\n",
        xml_escape(title),
        xml_escape(ylabel)
    )
}

fn histogram_svg(title: &str, xlabel: &str, series: &[(&str, &[f64])]) -> String {
    let width = 900.0;
    let height = 520.0;
    let left = 70.0;
    let right = 30.0;
    let top = 50.0;
    let bottom = 70.0;
    let plot_w = width - left - right;
    let plot_h = height - top - bottom;

    let mut all = Vec::new();
    for (_, v) in series {
        all.extend_from_slice(v);
    }
    let mn = all.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = all.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let span = (mx - mn).abs().max(1e-3);
    let bins = 12usize;
    let mut counts: Vec<Vec<f64>> = series.iter().map(|_| vec![0.0; bins]).collect();
    for (si, (_, v)) in series.iter().enumerate() {
        for x in *v {
            let mut b = ((*x - mn) / span * bins as f64).floor() as usize;
            if b >= bins {
                b = bins - 1;
            }
            counts[si][b] += 1.0;
        }
    }
    let maxc = counts
        .iter()
        .flat_map(|c| c.iter())
        .cloned()
        .fold(1.0, f64::max);

    let colors = ["#2563eb", "#dc2626"];
    let mut body = String::new();
    body.push_str(&format!(
        "<rect x='{left}' y='{top}' width='{plot_w}' height='{plot_h}' fill='white' stroke='black'/>"
    ));
    let group_w = plot_w / bins as f64;
    let bar_w = group_w / (series.len() as f64 + 0.4);
    for b in 0..bins {
        for (si, _) in series.iter().enumerate() {
            let c = counts[si][b];
            let h = if maxc > 0.0 { plot_h * (c / maxc) } else { 0.0 };
            let x = left + b as f64 * group_w + 4.0 + si as f64 * bar_w;
            let y = top + plot_h - h;
            let bw = bar_w - 2.0;
            let color = colors[si % colors.len()];
            body.push_str(&format!(
                "<rect x='{x}' y='{y}' width='{bw}' height='{h}' fill='{color}' opacity='0.85'/>"
            ));
        }
        let edge = mn + span * b as f64 / bins as f64;
        if b % 2 == 0 {
            let tx = left + (b as f64 + 0.5) * group_w;
            let ty = top + plot_h + 16.0;
            body.push_str(&format!(
                "<text x='{tx}' y='{ty}' text-anchor='middle' font-size='10' font-family='sans serif'>{edge:.2}</text>"
            ));
        }
    }
    for (i, (name, vals)) in series.iter().enumerate() {
        let st = stats(vals);
        let x = left + i as f64 * 280.0;
        let y = top + plot_h + 36.0;
        let ty = top + plot_h + 48.0;
        let color = colors[i % colors.len()];
        body.push_str(&format!(
            "<rect x='{x}' y='{y}' width='14' height='14' fill='{color}'/>"
        ));
        body.push_str(&format!(
            "<text x='{}' y='{ty}' font-size='13' font-family='sans serif'>{}  n={} mean={:.3} med={:.3}</text>",
            x + 20.0,
            xml_escape(name),
            vals.len(),
            st.mean,
            st.median
        ));
    }

    let cx_title = width / 2.0;
    let xlab_x = left + plot_w / 2.0;
    let xlab_y = height - 8.0;
    format!(
        "<?xml version='1.0' encoding='UTF-8'?>\n\
<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>\n\
  <rect width='100%' height='100%' fill='#fafafa'/>\n\
  <text x='{cx_title}' y='28' text-anchor='middle' font-size='18' font-family='sans serif' font-weight='700'>{}</text>\n\
  <text x='{xlab_x}' y='{xlab_y}' text-anchor='middle' font-size='12' font-family='sans serif'>{}</text>\n\
  {body}\n\
</svg>\n",
        xml_escape(title),
        xml_escape(xlabel)
    )
}

pub fn write_metrics_csv(path: &Path, algo: &str, rows: &[RoundMetrics]) -> Result<(), RunError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| RunError::Invariant(e.to_string()))?;
    }
    let mut s = String::from(
        "algo,round_index,seed,ticks,killed,remaining,mean_survival_s,mean_compute_ops,decision_count,total_compute_ops\n",
    );
    for m in rows {
        s.push_str(&format!(
            "{algo},{},{},{},{},{},{:.8},{:.8},{},{:.8}\n",
            m.round_index,
            m.seed,
            m.ticks,
            m.enemies_neutralized,
            m.remaining_enemies,
            m.mean_survival_s,
            m.mean_compute_ops,
            m.decision_count,
            m.mean_compute_ops * m.decision_count as f64
        ));
    }
    fs::write(path, s).map_err(|e| RunError::Invariant(e.to_string()))
}
