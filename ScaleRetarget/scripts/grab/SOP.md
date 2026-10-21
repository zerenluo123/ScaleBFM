# GRAB → g1d

原始数据 `grab`（下载的 10 个受试者压缩包 `zips/grab__s1.zip` … `grab__s10.zip`）只读。解压到 `grab_unpacked/s1 … s10/`
（每条一个 npz，120 fps，含身体、手、物体、桌子），产物都在 `grab_retarget`：`bvh/` 是转成 loader 格式的 pkl，`survey/` 是分区和
参考扫描，`vN/` 是一次批量转换。本机路径在 `/home/zerenluo/Space/bfm_datasets/` 下，换机器用环境变量 `GRAB`、`GRAB_UNPACKED`、
`GRAB_OUT` 指过去（`dataset.py`）。命令都在 `ScaleRetarget` 目录下、`conda activate scaleretarget` 之后执行。

流程和 BONES-SEED、HiPHI、LAFAN1 一样（`scripts/hiphi/SOP.md`），代码共用 `scaleretarget/pipeline/`。不同的地方：

- 一条是 `rel = s1/{物体}_{动作}[_N][_Retake]`，如 `s1/cup_drink_1`。没有镜像，一个受试者算一次采集（`session`）。
- AMASS 里的 GRAB 子集是同一批采集又拟合了一遍，不用它，用这份原版（`scripts/amass/` 跳过 `GRAB.tar.bz2`）。
- **体型**：SMPL-X 三个数据集（GRAB、OMOMO、AMASS）都和原版标准 G1 一样，用拟合到 G1 站姿的体型（`use_optimized_shape`，
  `unitree_g1d.yaml` 里 `smplx_shape_robot` 指向 G1：G1-D 上身和 G1 一样、没有腿），片子自己的体型参数不用。但根节点不缩放
  （对应表 `root_keeps_human_scale`），骨盆高度和位置还是人自己的，升降跟着人的骨盆走。5 条试转手腕误差中位 1.8–5.5 cm
  （标准 G1 1.8–3.8 cm）；改用片子自己的体型 + 原版比例表是 5.8–8.8 cm，所以没用。
- 对应表 `grab_to_unitree_g1d.yaml`（`amass_`、`omomo_` 两份内容相同，原版三份本来就相同）从原版 `grab_to_unitree_g1.yaml`
  推出来，做法同 LAFAN：上身连杆、偏移照抄，腿去掉，pelvis 驱动 pelvis_proxy，权重是原版的 0.2 倍（手的位置权重 2.0），
  加上 g1d 的 `ik_solver`、`frame_smooth`、`joint_comfort`。
- 每个进程启动时拟合一次体型（6000 步，约 1 分钟），扫描、转换、检查都一样，之后不再算。
- 参考手臂跳变规则同 HiPHI；SMPL-X 没有 BVH，只在 30 fps 下有翻转时才按原始帧率（120 fps）重读这一条比较（`smplx_survey.py`）。
- 「对比标准 G1」和突跳归因的对照，是当场用原版 `grab_to_unitree_g1.yaml` 转标准 G1（同一个 G1 体型）。

```
解压  →  convert.py  →  survey.py  →  classify.py  →  retarget.sh  →  audit.py
                                                            ↘  jump_audit.py flips
                                                               jump_audit.py segments  →  exclude_ood.py
```

## 0. 解压、转格式

```bash
mkdir -p $GRAB_UNPACKED && for z in $GRAB/zips/grab__s*.zip; do unzip -q -n $z -d $GRAB_UNPACKED; done
python scripts/grab/convert.py --workers 32
```

`convert.py` 用 `scaleretarget/utils/convert_grab_to_smplx.py` 把每条的身体、手的姿态、帧率、性别写成
`grab_retarget/bvh/s1/{名字}.pkl`，已有的跳过。1335 条约 1 GB，16 进程约 20 秒。

## 1. 运动学扫描

```bash
python scripts/grab/survey.py --out $GRAB_OUT/survey/survey_raw.parquet --workers 64
```

本机 20 进程 12 分钟（其中约 1 分钟是每个进程拟合体型）。

## 2. 分区

规则和其他数据集一样。GRAB 是站着在桌边抓东西（pass、lift、inspect、drink……），文件名里的动作词没有一条命中文字规则。

```bash
python scripts/grab/classify.py --survey $GRAB_OUT/survey/survey_raw.parquet --out_dir $GRAB_OUT/survey
```

写出 `survey/grab_zones.parquet`（和 `.csv`）、`green.txt`、`yellow.txt`、`red.txt`。2026-10 的结果（共 1335 条 3.76 h）：

| 区 | 条数 | 时长 |
|---|---|---|
| 绿 | 1027 | 2.88 h |
| 黄（后仰超限 302、升降到底 4） | 306 | 0.87 h |
| 红（贴地 2） | 2 | 0.01 h |

后仰超限是人站姿本身往后靠，和受试者强相关（s6 七成的条、s9 2%），机器人在 -2.5° 停住，差别很小，所以照常转换。
参考手臂跳变 0 条。

## 3. 批量转换

```bash
VERSION=v1 NOTE="first run" NUM_WORKERS=64 scripts/grab/retarget.sh
```

面板「受试者」一列是 s1–s10，「类别」是动作（pass、lift……），描述是物体和动作。

本机 8 条试转检查：帧数、关节范围、升降和转向速度都对，手腕误差中位 3.1–5.2 cm、p95 5.0–7.1 cm，肩 6–8 cm（LAFAN 16–18 cm），
只有吃苹果那条拇指碰到头（人本来就把手放到嘴边）。

## 4–6. 质量检查、突跳归因、剔除 OOD

```bash
D=grab OUT=$GRAB_OUT V=v1
python scripts/$D/audit.py --run_dir $OUT/$V --out_dir $OUT/$V/audit --workers 64
python scripts/$D/jump_audit.py flips --workers 64
python scripts/$D/jump_audit.py segments --version $V --workers 64
python scripts/$D/exclude_ood.py --version $V               # 只打印名单
python scripts/$D/exclude_ood.py --version $V --apply
python scripts/$D/browse.py --build_cache $V
```

`jump_audit.py flips` 按受试者筛用 `--subjects s1 s4`。

## 7. 在服务器上跑

服务器、tmux、本机挂载看面板见 HiPHI SOP 第 7 节。数据在 `~/humanoid_wbc/bfm_datasets/grab/zips`，`source ~/humanoid_wbc/bfm_env.sh`
设好 `GRAB`、`GRAB_UNPACKED`、`GRAB_OUT` 之后跑第 0–3 步（第 3 步在 tmux 里）。

本机看面板：

```bash
R=~/mnt/hkeai/humanoid_wbc/bfm_datasets
GRAB_OUT=$R/grab_retarget python scripts/grab/browse.py --version v1 --cache_only
```
