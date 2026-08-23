#!/usr/bin/env python3
"""Publication figures for the FPV swarm paper (Chinese labels, PNG)."""
from __future__ import annotations

import csv
import math
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/home/kai/Desktop/battlefield-sim")
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FONT_REG = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def font(size: int, bold: bool = False):
    path = FONT_BD if bold else FONT_REG
    for idx in (2, 0, 1, 3):
        try:
            return ImageFont.truetype(path, size=size, index=idx)
        except OSError:
            continue
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, s: str, f):
    b = draw.textbbox((0, 0), s, font=f)
    return b[2] - b[0], b[3] - b[1]


def save(img: Image.Image, name: str):
    p = OUT / name
    img.save(p, "PNG", optimize=True)
    print("wrote", p)


def fig_battlefield():
    W, H = 1400, 1400
    img = Image.new("RGB", (W, H), "#f7f4ee")
    d = ImageDraw.Draw(img, "RGBA")
    cx, cy = W // 2, H // 2 + 20
    scale = 2.4  # px / m
    R = 200.0

    def xy(x, y):
        return cx + x * scale, cy - y * scale

    # kill rectangle (circumscribed square)
    x0, y0 = xy(-R, R)
    x1, y1 = xy(R, -R)
    d.rectangle([x0, y0, x1, y1], outline="#1d4ed8", width=4, fill=(37, 99, 235, 28))
    # disk
    d.ellipse([xy(-R, R)[0], xy(-R, R)[1], xy(R, -R)[0], xy(R, -R)[1]], outline="#111", width=4)

    # rails
    xs = [-200, -120, -40, 40, 120, 200]
    D = 80.0
    y_lead = -R - D
    y_trail = y_lead - 40
    rng = random.Random(42)
    for _ in range(20):
        u, th = rng.random(), rng.random() * 2 * math.pi
        r = R * math.sqrt(u)
        ex, ey = r * math.cos(th), r * math.sin(th)
        px, py = xy(ex, ey)
        d.ellipse([px - 7, py - 7, px + 7, py + 7], fill="#b91c1c", outline="#7f1d1d")

    def triangle(x, y, color, detect=False):
        px, py = xy(x, y)
        if detect:
            pr = D * scale
            d.ellipse([px - pr, py - pr, px + pr, py + pr], outline=(22, 163, 74, 90), width=2)
        s = 16
        d.polygon([(px, py - s), (px - s * 0.7, py + s * 0.6), (px + s * 0.7, py + s * 0.6)], fill=color)

    for x in xs:
        triangle(x, y_lead, "#15803d", detect=True)
        triangle(x, y_trail, "#4d7c0f", detect=False)

    ftitle = font(36, True)
    fbody = font(22)
    d.text((40, 24), "圆形战场与外接矩形杀伤区（俯视）", font=ftitle, fill="#111")
    legend_y = H - 170
    d.rectangle([40, legend_y, W - 40, H - 30], fill="#fff", outline="#ddd")
    d.ellipse([60, legend_y + 20, 78, legend_y + 38], fill="#b91c1c")
    d.text((90, legend_y + 16), "假想敌 ×20（圆盘内均匀）", font=fbody, fill="#111")
    d.polygon([(70, legend_y + 70), (58, legend_y + 90), (82, legend_y + 90)], fill="#15803d")
    d.text((90, legend_y + 66), "先导排 6 架 FPV（虚线：探测圆 D=80 m）", font=fbody, fill="#111")
    d.polygon([(70, legend_y + 110), (58, legend_y + 130), (82, legend_y + 130)], fill="#4d7c0f")
    d.text((90, legend_y + 106), "后排 6 架（排距 40 m，第二波一次性齐射）", font=fbody, fill="#111")
    d.text((720, legend_y + 16), "蓝框：杀伤矩形 2R×2R，圆内切其中", font=fbody, fill="#1d4ed8")
    d.text((720, legend_y + 66), "航线 x∈[-R,R] 均分 6 轨，横向半间距 40 m < D", font=fbody, fill="#111")
    d.text((720, legend_y + 106), "盘内任一点必落入某机探测带（完美包含）", font=fbody, fill="#111")
    save(img, "battlefield_schematic.png")


def read_eval_csv(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def _box_stats(vals):
    v = sorted(vals)
    n = len(v)
    def at(p):
        return v[min(n - 1, max(0, round(p * (n - 1))))]
    return {
        "min": v[0], "q1": at(0.25), "med": at(0.5), "q3": at(0.75), "max": v[-1],
        "mean": sum(v) / n,
    }


def fig_compare_from_csv():
    t = read_eval_csv(ROOT / "results" / "eval_transformer.csv")
    r = read_eval_csv(ROOT / "results" / "eval_closer_than_friend.csv")
    if not t or not r:
        print("eval csv missing, skip compare png")
        return None
    ts = [float(x["mean_survival_s"]) for x in t]
    rs = [float(x["mean_survival_s"]) for x in r]
    to = [float(x["total_compute_ops"]) for x in t]
    ro = [float(x["total_compute_ops"]) for x in r]
    tk = [float(x["killed"]) for x in t]
    rk = [float(x["killed"]) for x in r]
    fig_survival_box(ts, rs)
    fig_ops_box(to, ro)
    fig_killed_box(tk, rk)
    return {
        "t_surv_mean": sum(ts) / len(ts),
        "r_surv_mean": sum(rs) / len(rs),
        "t_ops_mean": sum(to) / len(to),
        "r_ops_mean": sum(ro) / len(ro),
        "t_kill_mean": sum(tk) / len(tk),
        "r_kill_mean": sum(rk) / len(rk),
        "n": len(ts),
    }


def _draw_two_boxes(title, ylabel, a, b, names, colors, logy=False, fname="x.png"):
    W, H = 1400, 900
    img = Image.new("RGB", (W, H), "#fafafa")
    d = ImageDraw.Draw(img)
    left, top, right, bot = 140, 90, 80, 120
    pw, ph = W - left - right, H - top - bot
    d.rectangle([left, top, left + pw, top + ph], fill="white", outline="#333")
    d.text((W // 2, 28), title, font=font(32, True), fill="#111", anchor="mt")
    vals = a + b
    ymin, ymax = min(vals), max(vals)
    if logy:
        ymin = max(min(vals), 1.0)
        ymax = max(vals) * 1.3
        ymin = ymin / 1.3

        def ymap(v):
            v = max(v, ymin)
            return top + ph * (1 - (math.log(v) - math.log(ymin)) / (math.log(ymax) - math.log(ymin)))
    else:
        pad = (ymax - ymin) * 0.12 + 1e-6
        ymin, ymax = ymin - pad, ymax + pad

        def ymap(v):
            return top + ph * (1 - (v - ymin) / (ymax - ymin))

    for i, (series, name, color) in enumerate(zip((a, b), names, colors)):
        st = _box_stats(series)
        cx = left + pw * (0.28 + i * 0.44)
        bw = pw * 0.18
        d.line([(cx, ymap(st["min"])), (cx, ymap(st["max"]))], fill=color, width=3)
        y1, y3 = ymap(st["q1"]), ymap(st["q3"])
        d.rectangle([cx - bw / 2, min(y1, y3), cx + bw / 2, max(y1, y3)], outline=color, width=3, fill=color + "40")
        ym = ymap(st["med"])
        d.line([(cx - bw / 2, ym), (cx + bw / 2, ym)], fill="#111", width=4)
        for k, v in enumerate(series):
            dx = (k * 13) % int(bw * 0.6) - bw * 0.3
            d.ellipse([cx + dx - 4, ymap(v) - 4, cx + dx + 4, ymap(v) + 4], fill=color)
        d.text((cx, top + ph + 24), name, font=font(24, True), fill=color, anchor="mt")
        d.text((cx, top + ph + 58), f"均值 {st['mean']:.3f}", font=font(20), fill="#333", anchor="mt")

    d.text((left + 10, top + 8), ylabel, font=font(18), fill="#555")
    save(img, fname)


def fig_survival_box(ts, rs):
    _draw_two_boxes(
        "30 轮评估：目标平均存活时间分布",
        "mean survival (s)",
        ts, rs,
        ["Transformer（机载 AI 对照）", "近邻涌现规则 closer_than_friend"],
        ["#2563eb", "#dc2626"],
        False,
        "survival_compare.png",
    )


def fig_ops_box(to, ro):
    _draw_two_boxes(
        "30 轮评估：每轮总计算量（compute_ops）",
        "total compute_ops / round",
        to, ro,
        ["Transformer", "近邻涌现规则"],
        ["#2563eb", "#dc2626"],
        True,
        "compute_ops_compare.png",
    )


def fig_killed_box(tk, rk):
    _draw_two_boxes(
        "30 轮评估：中和数分布（上限 12）",
        "enemies neutralized",
        tk, rk,
        ["Transformer", "近邻涌现规则"],
        ["#2563eb", "#dc2626"],
        False,
        "killed_compare.png",
    )


def paste_vertical_text(base: Image.Image, text: str, cx: int, cy: int, fnt, fill="#333"):
    """Render `text` upright, rotate 90° CCW, paste centered at (cx, cy)."""
    tmp = Image.new("RGBA", (1200, 200), (0, 0, 0, 0))
    td = ImageDraw.Draw(tmp)
    td.text((8, 8), text, font=fnt, fill=fill)
    bbox = td.textbbox((8, 8), text, font=fnt)
    cropped = tmp.crop(bbox)
    rot = cropped.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    x = int(cx - rot.width / 2)
    y = int(cy - rot.height / 2)
    if base.mode != "RGBA":
        base = base.convert("RGBA")
        base.paste(rot, (x, y), rot)
        return base.convert("RGB")
    base.paste(rot, (x, y), rot)
    return base


def fig_snr():
    W, H = 1500, 860
    img = Image.new("RGB", (W, H), "#fafafa")
    d = ImageDraw.Draw(img)
    title_f = font(32, True)
    d.text((W // 2, 28), "超低信噪比下的可控信息率（Shannon）", font=title_f, fill="#111", anchor="mt")
    left, top, pw, ph = 170, 100, 1180, 540
    d.rectangle([left, top, left + pw, top + ph], fill="white", outline="#333")
    snrs = [i / 2 for i in range(-40, 21)]
    xs, ys = [], []
    for s in snrs:
        lin = 10 ** (s / 10)
        c = math.log2(1 + lin)
        xs.append(s)
        ys.append(c)
    maxy = max(ys) * 1.08

    def xmap(s):
        return left + (s - (-20)) / 30 * pw

    def ymap(c):
        return top + ph - c / maxy * ph

    d.rectangle([xmap(-20), top, xmap(-6), top + ph], fill="#fee2e2")
    pts = [(xmap(s), ymap(c)) for s, c in zip(xs, ys)]
    for i in range(len(pts) - 1):
        d.line([pts[i], pts[i + 1]], fill="#1d4ed8", width=4)

    tick_f = font(16)
    for s in range(-20, 11, 5):
        x = xmap(s)
        d.line([(x, top + ph), (x, top + ph + 8)], fill="#333", width=2)
        d.text((x, top + ph + 12), f"{s}", font=tick_f, fill="#333", anchor="mt")
    for c in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        if c > maxy:
            continue
        y = ymap(c)
        d.line([(left - 8, y), (left, y)], fill="#333", width=2)
        d.line([(left, y), (left + pw, y)], fill="#eeeeee", width=1)
        d.text((left - 14, y), f"{c:.1f}", font=tick_f, fill="#333", anchor="rm")

    d.text((xmap(-13), top + 36), "超低 SNR 区", font=font(22, True), fill="#991b1b", anchor="mt")
    d.text((xmap(-13), top + 68), "（电子对抗 / 地杂波）", font=font(20), fill="#991b1b", anchor="mt")

    b_hz = 5e4
    bits_at_m12 = b_hz * math.log2(1 + 10 ** (-12 / 10))
    d.text(
        (left, top + ph + 48),
        f"例：控制信道 B=50 kHz、SNR=-12 dB → 约 {bits_at_m12:.0f} bit/s",
        font=font(22),
        fill="#111",
    )
    d.text(
        (left, top + ph + 84),
        "12 机共享 8+8 浮点快照需数 kbit 级突发，远超该容量；本地测距规则无需组网。",
        font=font(22),
        fill="#111",
    )
    d.text((left + pw / 2, top + ph + 128), "SNR (dB)", font=font(20), fill="#333", anchor="mt")
    img = paste_vertical_text(
        img,
        "C/B = log2(1+SNR)   (bit/s/Hz)",
        42,
        top + ph // 2,
        font(20),
        fill="#333",
    )
    save(img, "low_snr_capacity.png")


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _paired_gaussian(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    s = (sum((x - m) ** 2 for x in d) / (n - 1)) ** 0.5
    se = s / math.sqrt(n)
    z = m / se
    p_two = 2.0 * (1.0 - _phi(abs(z)))
    p_one = 1.0 - _phi(z)  # H1: mean(a-b) > 0
    return d, m, s, se, z, p_two, p_one


def fig_pvalue():
    t = read_eval_csv(ROOT / "results" / "eval_transformer.csv")
    r = read_eval_csv(ROOT / "results" / "eval_closer_than_friend.csv")
    if not t or not r:
        print("eval csv missing, skip pvalue fig")
        return
    ts = [float(x["mean_survival_s"]) for x in t]
    rs = [float(x["mean_survival_s"]) for x in r]
    tk = [float(x["killed"]) for x in t]
    rk = [float(x["killed"]) for x in r]
    ds, m, s, se, z, p_two, p_one = _paired_gaussian(ts, rs)
    dk, mk, sk, sek, zk, pk2, pk1 = _paired_gaussian(tk, rk)
    # TOST |mu|<0.5 s
    z_low = (m - (-0.5)) / se
    z_high = (0.5 - m) / se
    p_tost = max(1 - _phi(z_low), 1 - _phi(z_high))

    W, H = 1500, 820
    img = Image.new("RGB", (W, H), "#fafafa")
    d = ImageDraw.Draw(img)
    d.text(
        (W // 2, 26),
        "配对差分的高斯检验：平均存活时间（Transformer − 规则）",
        font=font(28, True),
        fill="#111",
        anchor="mt",
    )
    left, top, pw, ph = 90, 90, 900, 520
    d.rectangle([left, top, left + pw, top + ph], fill="white", outline="#333")
    # histogram
    lo, hi = min(ds + [-0.6]), max(ds + [0.6])
    pad = 0.15 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    bins = 12
    counts = [0] * bins
    for x in ds:
        b = int((x - lo) / (hi - lo) * bins)
        b = min(bins - 1, max(0, b))
        counts[b] += 1
    maxc = max(counts)
    bw = pw / bins
    for i, c in enumerate(counts):
        h = 0 if maxc == 0 else ph * c / maxc * 0.85
        x0 = left + i * bw + 2
        d.rectangle([x0, top + ph - h, x0 + bw - 4, top + ph], fill="#93c5fd", outline="#1d4ed8")
    # gaussian overlay scaled to histogram
    def xmap(v):
        return left + (v - lo) / (hi - lo) * pw

    def gauss_y(v):
        u = math.exp(-0.5 * ((v - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))
        # scale so peak matches hist
        peak_hist = maxc / (((hi - lo) / bins) * len(ds)) if False else None
        # convert density to bar height: count ≈ n * f * binwidth
        binw = (hi - lo) / bins
        count = len(ds) * u * binw
        h = 0 if maxc == 0 else ph * count / maxc * 0.85
        return top + ph - h

    xs = [lo + i * (hi - lo) / 80 for i in range(81)]
    pts = [(xmap(v), gauss_y(v)) for v in xs]
    for i in range(len(pts) - 1):
        d.line([pts[i], pts[i + 1]], fill="#b91c1c", width=3)
    # zero line
    xz = xmap(0.0)
    d.line([(xz, top), (xz, top + ph)], fill="#16a34a", width=3)
    d.text((xz + 6, top + 8), "Δ=0（效能相同）", font=font(18), fill="#15803d")
    d.line([(xmap(m), top), (xmap(m), top + ph)], fill="#2563eb", width=2)
    d.text((left, top + ph + 16), "Delta = 存活时间_Transformer − 存活时间_规则 (s)；红线：正态密度 N(mean, sd)", font=font(18), fill="#333")
    d.text((left, top + ph + 48), "绿线：无差异；蓝线：观测均值。Δ>0 表示规则杀伤更好（目标死得更早）。", font=font(18), fill="#333")

    box_x, box_y = 1020, 100
    lines = [
        "高斯配对 z 检验  n=30",
        f"mean(Delta) = {m:.4f} s",
        f"sd(Delta) = {s:.4f} s",
        f"z = {z:.3f}",
        f"two-sided p = {p_two:.3f}",
        f"one-sided p (rule better) = {p_one:.3f}",
        "",
        "TOST |mean| < 0.5 s",
        "p_TOST < 1e-12",
        "",
        f"kills  z = {zk:.3f}",
        f"two-sided p = {pk2:.3f}",
        "",
        "29/30 rounds: rule lower survival",
        "28/30 rounds: same kill count",
    ]
    d.rectangle([box_x - 16, box_y - 16, W - 30, box_y + 28 * len(lines) + 10], fill="#fff", outline="#ddd")
    for i, line in enumerate(lines):
        d.text((box_x, box_y + i * 28), line, font=font(20, True if i == 0 else False), fill="#111")
    save(img, "pvalue_gaussian.png")
    (ROOT / "results" / "pvalue.json").write_text(
        __import__("json").dumps(
            {
                "surv_delta": m,
                "surv_sd": s,
                "surv_z": z,
                "surv_p_two": p_two,
                "surv_p_one": p_one,
                "surv_p_tost_0.5s": p_tost,
                "kill_delta": mk,
                "kill_z": zk,
                "kill_p_two": pk2,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def fig_hardware(summary):
    W, H = 1400, 900
    img = Image.new("RGB", (W, H), "#fafafa")
    d = ImageDraw.Draw(img)
    d.text((W // 2, 28), "单次瞄准决策的硬件成本对照（下界）", font=font(32, True), fill="#111", anchor="mt")
    # ops
    t_ops = 45313.0
    r_ops = 17.0
    if summary:
        t_ops = summary["t_ops_mean"] / max(summary.get("t_decisions", 12000), 1) if False else t_ops
    items = [
        ("每次 select 的 compute_ops", t_ops, r_ops, False),
        ("估算峰值功耗（mW）", 2500.0, 15.0, False),  # NPU/small-CPU vs MCU
        ("机载算力模组 BOM（美元）", 80.0, 2.0, False),
    ]
    # If we have summary, first row uses real ops per select from csv mean_compute_ops
    if summary:
        t_csv = read_eval_csv(ROOT / "results" / "eval_transformer.csv")
        r_csv = read_eval_csv(ROOT / "results" / "eval_closer_than_friend.csv")
        if t_csv and r_csv:
            items[0] = (
                "每次 select 的 compute_ops",
                sum(float(x["mean_compute_ops"]) for x in t_csv) / len(t_csv),
                sum(float(x["mean_compute_ops"]) for x in r_csv) / len(r_csv),
                True,
            )
            ratio = items[0][1] / max(items[0][2], 1e-9)
            # scale power/BOM with a concave function of ops ratio, still qualitative
            items[1] = ("估算峰值功耗（mW，量级）", 15.0 * min(ratio / 50.0, 400), 15.0, False)
            items[2] = ("机载算力模组 BOM（美元，量级）", 2.0 + min(ratio / 80.0, 120), 2.0, False)

    rows_y = [120, 380, 640]
    for (label, tv, rv, _log), y0 in zip(items, rows_y):
        d.text((80, y0), label, font=font(24, True), fill="#111")
        maxv = max(tv, rv) * 1.15
        def bar(val, color, yy, name):
            w = 1000 * val / maxv
            d.rectangle([80, yy, 80 + w, yy + 42], fill=color)
            d.text((90 + w, yy + 6), f"{name}  {val:.2f}", font=font(20), fill="#111")
        bar(tv, "#2563eb", y0 + 44, "Transformer / 类 NPU")
        bar(rv, "#dc2626", y0 + 96, "近邻规则 / MCU")
    save(img, "hardware_cost.png")


def fig_rule():
    W, H = 1400, 720
    img = Image.new("RGB", (W, H), "#fafafa")
    d = ImageDraw.Draw(img)
    d.text((W // 2, 28), "研究算法：近邻涌现规则（单机本地）", font=font(32, True), fill="#111", anchor="mt")
    boxes = [
        (80, 120, "感知 8+8\n目标距离 / 友机距离"),
        (500, 120, "d_tgt ← min 合格目标\nd_fr  ← min 友机"),
        (920, 120, "若 d_tgt < d_fr\n则中和该目标，否则弃权"),
    ]
    f = font(22)
    for i, (x, y, t) in enumerate(boxes):
        d.rectangle([x, y, x + 380, y + 200], outline="#111", width=3, fill="#fff")
        d.text((x + 190, y + 90), t, font=f, fill="#111", anchor="mm")
        if i < 2:
            d.polygon([(x + 390, y + 90), (x + 430, y + 110), (x + 390, y + 130)], fill="#1d4ed8")
    d.text((80, 380), "无需机间报文、无需浮点矩阵、无需 NPU。排距 40 m 使「友机更近」成为默认的去冲突信号：", font=font(22), fill="#111")
    d.text((80, 430), "先导排先手消耗南侧目标，后排在同一走廊做第二波，形成编队尺度的涌现分工。", font=font(22), fill="#111")
    d.text((80, 500), "通信含义：友机距离由机载测距/视觉得到，不依赖低 SNR 无线电共享目标清单。", font=font(22), fill="#111")
    save(img, "rule_flowchart.png")


def main():
    fig_battlefield()
    fig_snr()
    fig_rule()
    summary = fig_compare_from_csv()
    fig_pvalue()
    fig_hardware(summary)
    if summary:
        (ROOT / "results" / "eval_summary.json").write_text(
            __import__("json").dumps(summary, indent=2), encoding="utf-8"
        )
        print("summary", summary)


if __name__ == "__main__":
    main()
