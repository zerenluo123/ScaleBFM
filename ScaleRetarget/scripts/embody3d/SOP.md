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

## 7. 抬高版（v2）：救回贴地、坐跪

v1 的「贴地」（任一帧胯部低于 0.55 m）、「坐跪」（胯部中位低于 0.70 m 或文字说坐、跪）整条进红区，一共 4221 条 / 100.3 h，
里面很多是跪着、坐着、坐在地上，上半身在做事。g1d 没有腿，升降到底时胯部停在 0.801 m（`workspace.LIFT_BASE`），人再往下，
手的目标也跟着往下，上半身就被拉变形。v2 在 retarget 时打开 `++robot.root_height_filter.min_height=0.801`：胯部低于 0.801 m
的帧，整个人往上平移到胯部 0.801 m 再平滑（`root_height_filter.py`），上半身相对胯部的姿态不变，等于把腿切掉。不开时和原来逐字节
一样；IK 没改。代价是跪、坐时手碰地面、桌面的接触关系没了（手跟着抬高 0.2–0.6 m）。

分区规则（`workspace.assign_raised_zones`，`classify.py --torso`），按整条定，不切段：

- 贴地、坐跪的片段抬高后保留，按抬高后的动作重新分黄绿区（绿区原因写「抬高救回」）；但只要有一帧胯部低于 0.35 m、上身前倾或后倒
  超过 45° 就留红区「趴躺伏地」。上身偏离直立 60° 以上的帧，胯部高度在 0.10–0.35 m（躺在地上）和 0.35–0.70 m（跪着往前趴）
  两堆，中间几乎没有，0.35 m 取的是这个空档。
- 所有保留的片段（原黄绿区也一样）都按抬高后的动作分区，并且：后仰超过 −19°（v1 黄绿区最差的 5%）连续超过 2 s，进红区
  「后仰过大」（例：坐在椅子上往后靠着讲课）；上身侧倾（偏离前后竖直面，g1d 没有这个关节）超过 25° 连续超过 2 s，进红区「侧倾过大」。
  阈值是对着面板里看过的 10 条定的：侧倾 20° 会误判两条好数据。

```bash
python scripts/embody3d/torso_survey.py --out $EMBODY3D_OUT/survey/torso_raw.parquet --workers 64      # 约 90 min
python scripts/embody3d/classify.py --survey $EMBODY3D_OUT/survey/survey_raw.parquet --out_dir $EMBODY3D_OUT/survey \
    --torso $EMBODY3D_OUT/survey/torso_raw.parquet
python scripts/embody3d/link_unraised.py --src v1 --dst v2            # 只打印
python scripts/embody3d/link_unraised.py --src v1 --dst v2 --apply
VERSION=v2 NOTE="raised: hips kept at or above the lowest lift" NUM_WORKERS=64 scripts/embody3d/retarget.sh \
    ++robot.root_height_filter.min_height=0.801
```

然后对 v2 照第 3–6 步检查、剔除 OOD。`link_unraised.py`：保留的片段里全程胯部不低于 0.801 m 的，抬高不起作用，结果和 v1
一样，直接从 v1 硬链接 pkl（v1 的 OOD 链接进 v2 的 OOD，原因一起带过来），v2 只转其余的；v1 不动。

| | 绿区 | 黄区 | 红区 |
|---|---|---|---|
| v1 分区 | 4022 条 / 39.1 h | 17069 条 / 333.8 h | 5676 条 / 128.0 h |
| v2 分区 | 5520 条 / 59.3 h（其中抬高救回 1220 条 / 17.9 h） | 18058 条 / 357.5 h | 3189 条 / 84.1 h |

v2 的红区：后仰过大 799 条 / 29.1 h，趴躺伏地 604 / 12.7，侧倾过大 331 / 14.5，其余原因同 v1。v1 黄绿区里因后仰、侧倾改进红区
的 351 条 / 15.9 h。v2 从 v1 链接 18621 条 / 309.3 h（945 条在 OOD），转 4957 条 / 107.6 h。

`jump_audit.py flips` 按子集筛用 `--subsets locomotion charades`。服务器上检查步骤用 `--workers 8`（见 `scripts/amass/SOP.md`）。

本机用 7 个子集各一段采集的 13 条试过（`convert` → `survey` → `classify` → 转 3 条 → 检查）：手腕误差中位 2–5 cm，骨盆 < 1 cm；
双手交握、搭在身前时两手的拇指、腕部互相穿模（一条对话片段整段都是）。
