# Embody3D → g1d

原始数据 `embody3d`（官方下载的 7 个 SMPL-X 压缩包 `embody_3d__20251016__{大类}_smplx.zip`，约 197 GB）只读。7 个大类解开是
16 个子集（MULTIPERSON 是 emotions、location、polyadic、icebreakers，HANDS 是 hand_single、hand_duo、hand_contact、
contact_touch……），一个子集下每段采集一个文件夹，采集里每个参与者一个文件夹，存逐帧的 SMPL-X 数组（30 fps，Y 朝上）。原版用全部
16 个子集，这里也是。产物都在 `embody3d_retarget`：`bvh/` 是每人每段采集一个 pkl，`survey/`、`vN/` 同其他数据集。换机器用环境变量
`EMBODY3D`、`EMBODY3D_OUT`（`dataset.py`）。

流程、体型、对应表、参考跳变都和 GRAB 一样（`scripts/grab/SOP.md`，SMPL-X 的部分写在那里）。原版转标准 G1 的对应表和体型拟合表
跟 AMASS 的逐字相同，g1d 表 `embody3d_to_unitree_g1d.yaml` 也照 AMASS 的。不同的地方：

- 一条是 `rel = {子集}/{采集}/{参与者}`。采集名是 `c--{日期}--{时间}--{参与者们}--pilot--MotionPrior{N}--{任务}--{起}-{止}`；
  同一日期时间是同一组人的一次采集（面板的 session），「子数据集」一列和「类别」都是子集。
- 不在地面上：按原版 loader 转成 Z 朝上后，身体最低点在 z = 0 上面 0.2–0.4 m，每个人不一样（同一段采集的几个人也不同），
  站着骨盆 1.2–1.4 m。原版 README 用 `kinematic` 格式化器会调高度，g1d 用的基础格式化器不调。`config/loader/embody3d.yaml` 的
  `ground: True` 让 loader 把每条片段抬或降到它自己体型（300 个 betas，neutral）在追踪正常的帧上的网格最低点为 z = 0；抬过之后
  站着骨盆 0.88–1.04 m。
- 每条带逐帧的追踪标记（压缩包里的 `missing/`，1 是追踪正常），原版没用。有任何一帧追踪不正常的片段进红区「参考数据缺帧」
  （那一帧的姿态不是人的）。
- 文字是采集名里的任务（`LOCOMOTION_crouched_sneaking_hiding` → crouched sneaking hiding，去掉打头的大类和数字）；对话类只有
  场景编号（scene），只靠胯部高度、俯仰、速度分区。

## 0. 转格式

```bash
python scripts/embody3d/convert.py --workers 16
```

直接从压缩包读，不用解压。每个参与者写成 `embody3d_retarget/bvh/{rel}.pkl`：原版 `gather_embody3d.py` 收的身体部分（`pose_body`、
`root_orient`、`trans`），加上体型（一行，压缩包每帧重复存）、性别（neutral）、帧率（30）、追踪标记；手的姿态不要（loader 把手放平）。
已有的跳过。

## 1–2. 扫描、分区

```bash
python scripts/embody3d/survey.py --out $EMBODY3D_OUT/survey/survey_raw.parquet --workers 64
python scripts/embody3d/classify.py --survey $EMBODY3D_OUT/survey/survey_raw.parquet --out_dir $EMBODY3D_OUT/survey
```

## 3–6. 批量转换、检查、剔除 OOD

```bash
VERSION=v1 NOTE="first run" NUM_WORKERS=64 scripts/embody3d/retarget.sh    # 在 tmux 里跑
D=embody3d OUT=$EMBODY3D_OUT V=v1
python scripts/$D/audit.py --run_dir $OUT/$V --out_dir $OUT/$V/audit --workers 8
python scripts/$D/jump_audit.py flips --workers 8
python scripts/$D/jump_audit.py segments --version $V --workers 8
python scripts/$D/exclude_ood.py --version $V               # 只打印名单
python scripts/$D/exclude_ood.py --version $V --apply
python scripts/$D/browse.py --build_cache $V
```

`jump_audit.py flips` 按子集筛用 `--subsets locomotion charades`。服务器上检查步骤用 `--workers 8`（见 `scripts/amass/SOP.md`）。

本机用 7 个子集各一段采集的 13 条试过（`convert` → `survey` → `classify` → 转 3 条 → 检查）：手腕误差中位 2–5 cm，骨盆 < 1 cm；
双手交握、搭在身前时两手的拇指、腕部互相穿模（一条对话片段整段都是）。
