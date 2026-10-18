# BONES-SEED → g1d

原始数据 `/home/zerenluo/Space/bfm_datasets/bones-seed` 只读。产物都在
`/home/zerenluo/Space/bfm_datasets/bone-seed_retarget`：`survey/` 是分区和参考扫描，`vN/` 是一次批量转换。
路径集中在 `dataset.py`。分区规则、面板、检查、归因、剔除的代码在 `scaleretarget/pipeline/`，和 HiPHI（`scripts/hiphi/SOP.md`）共用；
这里的脚本只是入口。命令都在 `ScaleRetarget` 目录下、`conda activate scaleretarget` 之后执行。

一个版本一份配置。`vN/config` 和 `vN/assets` 是开跑时的拷贝；这几份文件改过就不能续跑，要开下一个版本号。

```
survey.py  →  classify.py  →  retarget.sh  →  audit.py
                                         ↘  jump_audit.py flips
                                            jump_audit.py segments  →  exclude_ood.py
```

`flips` 不读 pkl，分区一出来就可以和批量转换同时跑。

## 1. 运动学扫描

只读 BVH 里分区要用的关节，降到 30 fps，每个片段一行指标。全量大约 6 分钟（20 进程）。改分区阈值不用重扫。

```bash
python scripts/bones_seed/survey.py \
  --out /home/zerenluo/Space/bfm_datasets/bone-seed_retarget/survey/survey_raw.parquet \
  --workers 20
```

## 2. 分区

按扫描结果和工作空间分成三区，写出 `survey/bones_seed_zones.parquet` 和 `green.txt`、`yellow.txt`、`red.txt`。批量转换只读绿、黄两个名单。

- **红**：g1d 用不了。非直立、地形、跳跃出了工作空间，或贴地：Hips 低于 0.55 m（比升降最低还低 20 多 cm，坐地、跪坐、深蹲）的帧超过 1%，和黄区同一个容忍度。
- **黄**：能用，但有一部分帧超出工作空间（升降行程、后仰、底盘速度）。
- **绿**：几乎每帧都在工作空间内。升降速度只记录，不决定分区。

```bash
python scripts/bones_seed/classify.py \
  --survey /home/zerenluo/Space/bfm_datasets/bone-seed_retarget/survey/survey_raw.parquet \
  --out_dir /home/zerenluo/Space/bfm_datasets/bone-seed_retarget/survey
```

改阈值后只重跑这一步，看各区条数和时长，再开批量转换。

## 3. 批量转换

`VERSION` 不存在时必须是下一个编号。已有的版本会跳过转过的片段，包括已经进 OOD 的。面板单独开着，转换结束或 Ctrl-C 都不关它；同一个版本不能开第二份。

```bash
VERSION=v1 NOTE="first run" scripts/bones_seed/retarget.sh
VERSION=v1 scripts/bones_seed/retarget.sh          # 中断后续跑
scripts/bones_seed/retarget.sh --panel             # 只开面板
```

面板里跟进度、播放片段、和官方 G1 对比。绿区和黄区播放转换结果，红区只播放参考。转换过程中也可以右键「剔除OOD数据」；OOD 区里右键「恢复到原数据区」。

突跳名单来自面板的帧缓存（`vN/frame_counts.csv`，OOD 的在 `vN/ood/frame_counts.csv`）。面板开着就会扫；没开面板时：

```bash
python scripts/bones_seed/browse.py --build_cache v1
python scripts/bones_seed/browse.py --build_cache v1 --loop 60    # 每 60 秒更新一次，直到 Ctrl-C
```

版本在别的机器上转、本机通过挂载看时，面板加 `--cache_only`：只读那台机器维护的缓存，不在挂载上遍历 pkl（见 `scripts/hiphi/SOP.md` 第 7 节）。

## 4. 质量检查

全量查 NaN、帧数是否和重采样一致、关节范围、升降 / 关节 / 底盘速度。再按区、原因、类别抽 2% 重算 IK 目标，看连杆跟踪误差和自碰。OOD 里的片段不在数据集里，不查。

```bash
python scripts/bones_seed/audit.py \
  --run_dir /home/zerenluo/Space/bfm_datasets/bone-seed_retarget/v1 \
  --out_dir /home/zerenluo/Space/bfm_datasets/bone-seed_retarget/v1/audit
```

写出 `full.parquet` 和 `sample.parquet`。

## 5. 突跳归因

两张表，剔除时都要。

`flips` 扫绿区和黄区的参考动作：上臂和前臂一起转、手几乎不动的步数。可以只扫某几天；已有其他日期的行会保留。

```bash
python scripts/bones_seed/jump_audit.py flips
python scripts/bones_seed/jump_audit.py flips --dates 230112
```

写出 `survey/ref_twist_flips.csv`。

`segments` 给这个版本里手臂关节超过 20 rad/s 的每一段归因，对照官方 G1。片段可以在 `vN/bvh` 或 `vN/ood`。先有第 3 步的帧缓存。

```bash
python scripts/bones_seed/jump_audit.py segments --version v1
```

写出 `vN/audit/jump_segments.csv`。原因按这个顺序：

| cause | 含义 | 剔除 |
| --- | --- | --- |
| ref_fails | 官方 G1 这段也跟不上 | 是，连镜像 |
| ref_bug | 参考数据翻转或目标瞬移 | 是，连镜像 |
| joint_limits | 卡在收窄的肩 / 肘限位 | 是 |
| wrist_cap | 转腕被逐帧上限挤到肩上 | 是 |
| fast_motion | 人手臂确实这么快，连杆跟着目标 | 留 |

## 6. 剔除 OOD

先看名单，再移动。规则：一次采集里某演员有一半以上扫过的片段在翻转，他这次采集的绿、黄片段整组剔除；上表里要剔除的突跳段也剔除。已经在 OOD 的跳过。

```bash
python scripts/bones_seed/exclude_ood.py --version v1
python scripts/bones_seed/exclude_ood.py --version v1 --apply
```

pkl 从 `vN/bvh/<rel>.pkl` 移到 `vN/ood/bvh/<rel>.pkl`，记在 `vN/ood/ood.csv`。之后再跑这个版本不会把它们转回去。面板 OOD 区可以单条恢复。

面板里手动放回去的片段没有单独记录。规则没变就再跑 `--apply`，会再次剔除。先跑不带 `--apply` 的那条看名单。

剔除后再跑一遍第 4 步，质量报告就是留下的数据集。
