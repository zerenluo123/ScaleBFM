# OMOMO → g1d

原始数据 `omomo/data.tar.gz`（22 GB）只读。只用里面两个整段序列文件 `data/train_diffusion_manip_seq_joints24.p`、
`data/test_diffusion_manip_seq_joints24.p`（17 个受试者搬、推、拖 15 种大物件，SMPL-X 身体姿态，30 fps；其余是物体几何和训练用的
切窗），解到 `omomo_unpacked/data/`。产物都在 `omomo_retarget`：`bvh/` 是每条一个 pkl，`survey/`、`vN/` 同其他数据集。换机器用
环境变量 `OMOMO`、`OMOMO_UNPACKED`、`OMOMO_OUT`（`dataset.py`）。

流程、体型、对应表、参考跳变都和 GRAB 一样（`scripts/grab/SOP.md`，SMPL-X 的部分写在那里）。不同的地方：

- 一条是 `rel = {序列名}`，如 `sub10_clothesstand_000`（受试者、物体、编号）。一个受试者算一次采集。训练集、测试集都用
  （测试集是 sub16、sub17）。
- 原始就是 30 fps，参考跳变的一步就是一个原始帧。
- 没有文字，文件名只有物体，文字规则不适用；从地上搬东西靠胯部高度判「贴地」「升降到底」。

## 0. 解出序列、转格式

```bash
mkdir -p $OMOMO_UNPACKED && tar -xzf $OMOMO/data.tar.gz -C $OMOMO_UNPACKED \
  data/train_diffusion_manip_seq_joints24.p data/test_diffusion_manip_seq_joints24.p     # 要扫完整个包，约 3 分钟
python scripts/omomo/convert.py
```

`convert.py` 按 `scaleretarget/utils/convert_omomo_to_smplx.py` 的格式把 5882 条写成 `omomo_retarget/bvh/{序列名}.pkl`（约 2 GB），
已有的跳过。

## 1–2. 扫描、分区

```bash
python scripts/omomo/survey.py --out $OMOMO_OUT/survey/survey_raw.parquet --workers 64
python scripts/omomo/classify.py --survey $OMOMO_OUT/survey/survey_raw.parquet --out_dir $OMOMO_OUT/survey
```

本机 20 进程 13 分钟。2026-10 的结果（共 5882 条 9.80 h）：

| 区 | 条数 | 时长 |
|---|---|---|
| 绿 | 4501 | 7.31 h |
| 黄（升降到底 765、后仰超限 141、底盘超速 10） | 916 | 1.62 h |
| 红（贴地 462、坐跪 3） | 465 | 0.87 h |

参考手臂跳变 0 条。

## 3–6. 批量转换、检查、剔除 OOD

```bash
VERSION=v1 NOTE="first run" NUM_WORKERS=64 scripts/omomo/retarget.sh    # 在 tmux 里跑
D=omomo OUT=$OMOMO_OUT V=v1
python scripts/$D/audit.py --run_dir $OUT/$V --out_dir $OUT/$V/audit --workers 64
python scripts/$D/jump_audit.py flips --workers 64
python scripts/$D/jump_audit.py segments --version $V --workers 64
python scripts/$D/exclude_ood.py --version $V               # 只打印名单
python scripts/$D/exclude_ood.py --version $V --apply
python scripts/$D/browse.py --build_cache $V
```

面板「受试者」一列是 sub1–sub17，「类别」是物体。`jump_audit.py flips` 按受试者筛用 `--subjects sub1 sub4`。
