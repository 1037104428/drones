# battlefield-sim

圆形战场 FPV 扫掠仿真（Rust）。20 个静止假想敌均匀落在圆盘内；12 架一次性无人机按 2×6 编队沿直径扫过。瞄准算法可插拔。评分：确定性 `compute_ops` 与目标平均存活时间（未击中者在 \(T_{\mathrm{end}}\) 删失）。

**仓库：** <https://github.com/1037104428/drones>

- 中文论文：[`paper/fpv_emergent.pdf`](paper/fpv_emergent.pdf)
- 英文论文：[`paper/fpv_emergent_en.pdf`](paper/fpv_emergent_en.pdf)
- English README: [`README.md`](README.md)

## 构建与测试

```bash
git clone https://github.com/1037104428/drones.git
cd drones
cargo test
```

## 复现论文实验

Transformer 用 **逐步 REINFORCE** 训练（不用模仿学习）：每一仿真拍把当拍全部无人机的决策做成一个并行批次更新。团队奖励是平均存活时间的增量 \(-n_{\mathrm{alive}}\Delta t/N + n_{\mathrm{kills}}(T_{\mathrm{end}}-t)/N\)。计算耗时不进入 loss。

```bash
cargo run --release -- experiment --train-rounds 200 --eval-rounds 30 --seed 20260823
# 高斯进入（σx=70 m，σy=30 m）：
cargo run --release -- experiment --ingress gaussian --train-rounds 200 --eval-rounds 30 --seed 20260823
# 同拍通信 120 m（D_tgt=50 m 时结果与无通信相同）：
cargo run --release -- experiment --comms-range 120 --train-rounds 200 --eval-rounds 30 --seed 20260823
```

同一种子评估：贪心 `nearest_in_range`、无通信贪心 `greedy_no_comms`、近邻规则 `closer_than_friend`、强化学习 Transformer。

仓库已含一次完整评估的 CSV 与 SQLite，可直接对照论文表格。
