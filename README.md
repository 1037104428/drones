# battlefield-sim

圆形战场上的 FPV 扫掠仿真（Rust）：20 个静止假想敌，12 架一次性自杀式无人机（2×6）沿圆盘直径扫过。瞄准算法可插拔；评分用确定性 `compute_ops` 与目标平均存活时间（含 \(T_{\mathrm{end}}\) 删失）。

**仓库：** <https://github.com/1037104428/drones>

中文论文：[`paper/fpv_emergent.pdf`](paper/fpv_emergent.pdf)（源稿 [`paper/fpv_emergent.tex`](paper/fpv_emergent.tex)）。

## 构建与测试

```bash
git clone https://github.com/1037104428/drones.git
cd drones
cargo test
```

## 复现本文实验

```bash
# 必须 --release：debug 下 Transformer 前向过慢
cargo run --release -- experiment --train-rounds 200 --eval-rounds 30 --seed 20260823
```

产出：

| 路径 | 内容 |
|------|------|
| `models/transformer.json` | 训练权重 |
| `results/eval_transformer.csv` | Transformer 30 轮评估 |
| `results/eval_closer_than_friend.csv` | 近邻规则 30 轮评估 |
| `results/pvalue.json` | 高斯配对 \(z\) / TOST |
| `data/experiment.sqlite` | 评估轮次的决策与存活记录 |
| `paper/figures/` | 论文插图 |

仓库里已包含一次完整评估的 CSV 与 SQLite，可直接对照论文表格，不必先训练。

单算法：

```bash
cargo run --release -- run --rounds 1 --seed 42 --algo nearest_in_range
cargo run --release -- run --rounds 30 --seed 10000 --algo closer_than_friend
```

退出码：0 成功，1 不变式，2 配置，3 SQLite。日志：`RUST_LOG=battlefield_sim=info`。
