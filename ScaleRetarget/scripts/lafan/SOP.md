# LAFAN1 → g1d

原始数据 `lafan1`（从 `lafan1.zip` 解出的 77 个 BVH，一个 take 一个文件，如 `walk1_subject1.bvh`，30 fps）只读。产物都在
`lafan1_retarget`：`bvh/` 是切好的片段，`survey/` 是分区和参考扫描，`vN/` 是一次批量转换。本机路径在
`/home/zerenluo/Space/bfm_datasets/` 下，换机器用环境变量 `LAFAN`、`LAFAN_OUT` 指过去（`dataset.py`）。命令都在
`ScaleRetarget` 目录下、`conda activate scaleretarget` 之后执行。

流程和 BONES-SEED、HiPHI 一样（`scripts/hiphi/SOP.md`），代码共用 `scaleretarget/pipeline/`。不同的地方：

- 一个 take 1.7–5 分钟，走着走着躺到地上、摔倒再站起来都在同一条里，整条分区的话 77 条只剩 27 条（1.75 h / 4.6 h）。所以先把
  每个 take 切成约 30 秒的片段（第 0 步），片段是流水线里的一条：`rel = {take}_{subject}_pNN`，如 `walk1_subject1_p03`。
- 没有镜像。一个演员（subject1–5）算一次采集（`session`）。
- 对应表 `lafan_to_unitree_g1d.yaml` 从原版 `lafan_to_unitree_g1.yaml` 推出来，做法同 SOMA：上身的连杆、偏移照抄，腿去掉，
  Hips 驱动 pelvis_proxy、按人的尺度（1.0），胸口和手臂相对 Hips 用原版 G1 的比例（0.9、0.75 乘 1.75/1.8）。权重是原版的
  0.2 倍：g1d 的防跳项（frame_smooth、joint_comfort）和阻尼是按手的位置权重 2.0 定的，原版的 10 大了 5 倍，防跳项就相对
  弱了 5 倍。8 条各 1 分钟的试转：原版权重 102 处手臂突跳（> 20 rad/s），0.2 倍 6 处，手腕误差不变（中位 6.6 cm、p95 10.2 cm）。
  标准 G1 同样 8 条 81 处。
- 「对比标准 G1」和突跳归因的对照，是当场用原版 ScaleRetarget 的 `lafan_to_unitree_g1.yaml` 转标准 G1。
  另外 Hugging Face 上 `lvhaidong/LAFAN1_Retargeting_Dataset` 有宇树自己转的 G1（40 个 take 的 CSV，在服务器
  `bfm_datasets/lafan1_g1_official/g1`），不是 ScaleRetarget 的结果，流水线没用它。
- 原版表把肩关节点给了 shoulder_yaw_link（在上臂中段），所以质量检查里肩的位置误差中位 16–18 cm，标准 G1 也有 14–16 cm，
  是表本身的，不是 g1d 的问题；胸口位置权重是 0，误差约 21 cm 也一样。g1d 骨盆不能侧倾、前后倾，pelvis 朝向差 11–14°，
  手腕比标准 G1 差 1–2 cm（3 条各 30 秒的试转）。

```
split.py  →  survey.py  →  classify.py  →  retarget.sh  →  audit.py
                                                 ↘  jump_audit.py flips
                                                    jump_audit.py segments  →  exclude_ood.py
```

## 0. 切片段

每个 take 切成 round(时长 / 30 秒) 段（至少一段）等长、不重叠的片段，写到 `lafan1_retarget/bvh/{take}_{subject}_pNN.bvh`：
文件头不变（只改帧数），动作行原样，位置还在 take 的世界坐标里。77 个 take 切成 547 段，每段 25–35 秒，拼回去和原文件
逐字节一样（496672 帧）。重跑会重写同样的文件。

```bash
python scripts/lafan/split.py
```

## 1. 运动学扫描

每段用 LafanLoader 按批量转换的方式读成 30 fps，算工作空间指标，记下骨架（BVH 骨长的哈希；每个 take 的骨长都略有不同，
77 个 take 77 种）和参考手臂跳变
（`scaleretarget/pipeline/ref_pops.py`，规则同 HiPHI；LAFAN 原始就是 30 fps，一步就是一个原始帧）。547 段 16 进程约半分钟。

```bash
python scripts/lafan/survey.py --out /home/zerenluo/Space/bfm_datasets/lafan1_retarget/survey/survey_raw.parquet --workers 16
```

## 2. 分区

规则和 BONES-SEED、HiPHI 一样，每段单独分区。LAFAN 没有文字描述，文字规则读 take 的动作名拆成的词（`fallAndGetUp` →
fall and get up）：`obstacles` 的 6 种 take 全部按「地形」进红区（翻越、钻过、踩上障碍），摔倒、地面动作、坐下靠胯部高度判「贴地」。

```bash
python scripts/lafan/classify.py \
  --survey /home/zerenluo/Space/bfm_datasets/lafan1_retarget/survey/survey_raw.parquet \
  --out_dir /home/zerenluo/Space/bfm_datasets/lafan1_retarget/survey
```

写出 `survey/lafan_zones.parquet`（和 `.csv`）、`green.txt`、`yellow.txt`、`red.txt`。2026-10 的结果：

| 区 | 段数 | 时长 |
|---|---|---|
| 绿 | 60 | 0.50 h |
| 黄（底盘超速 121、升降到底 89、后仰超限 11、前俯超限 1） | 222 | 1.84 h |
| 红（地形 122、贴地 121、跳跃出工作空间 22） | 265 | 2.25 h |

## 3. 批量转换

和 BONES-SEED、HiPHI 一样：`VERSION` 不存在时必须是下一个编号，续跑跳过转过的片段和 OOD 里的，配置文件改过就要开新版本。

```bash
VERSION=v1 NOTE="first run" NUM_WORKERS=64 scripts/lafan/retarget.sh
```

面板「演员」一列是 subject，「类别」是动作名（walk、dance……），描述里写着来自哪个 take 的第几段。

## 4–6. 质量检查、突跳归因、剔除 OOD

```bash
D=lafan OUT=$LAFAN_OUT V=v1
python scripts/$D/audit.py --run_dir $OUT/$V --out_dir $OUT/$V/audit --workers 64
python scripts/$D/jump_audit.py flips --workers 64
python scripts/$D/jump_audit.py segments --version $V --workers 64
python scripts/$D/exclude_ood.py --version $V               # 只打印名单
python scripts/$D/exclude_ood.py --version $V --apply
python scripts/$D/browse.py --build_cache $V
```

`jump_audit.py flips` 按演员筛用 `--subjects subject1 subject4`。

## 7. 在服务器上跑

服务器、tmux、本机挂载看面板见 HiPHI SOP 第 7 节。数据在 `~/humanoid_wbc/bfm_datasets/lafan1`（77 个 BVH 和 `lafan1.zip`），
`source ~/humanoid_wbc/bfm_env.sh` 设好 `LAFAN`、`LAFAN_OUT` 之后：

```bash
python scripts/lafan/split.py
python scripts/lafan/survey.py --out $LAFAN_OUT/survey/survey_raw.parquet --workers 64
python scripts/lafan/classify.py --survey $LAFAN_OUT/survey/survey_raw.parquet --out_dir $LAFAN_OUT/survey
VERSION=v1 NOTE="first run" NUM_WORKERS=64 scripts/lafan/retarget.sh    # 在 tmux 里跑
```

本机看面板：

```bash
R=~/mnt/hkeai/humanoid_wbc/bfm_datasets
LAFAN=$R/lafan1 LAFAN_OUT=$R/lafan1_retarget python scripts/lafan/browse.py --version v1 --cache_only
```
