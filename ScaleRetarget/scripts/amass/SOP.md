# AMASS → g1d

原始数据 `amass`（下载的 SMPL-X N 压缩包，一个子数据集一个 `{子集}.tar.bz2`）只读。`GRAB.tar.bz2` 不用（是 GRAB 原版又拟合了
一遍，用 `scripts/grab/` 的原版）；其余 22 个原样解到 `amass_retarget/bvh/{子集}/{受试者}/{名字}_stageii.npz`，这就是 loader
的格式，不用再转。产物都在 `amass_retarget`：`bvh/`、`survey/`、`vN/` 同其他数据集。换机器用环境变量 `AMASS`、`AMASS_OUT`
（`dataset.py`）。

流程、体型、对应表、参考跳变都和 GRAB 一样（`scripts/grab/SOP.md`，SMPL-X 的部分写在那里）。不同的地方：

- 一条是 `rel = {子集}/{受试者}/{名字}_stageii`，如 `KIT/3/walking_slow01_stageii`。一个子集的一个受试者算一次采集；面板
  「子数据集」一列和「类别」都是子集（CMU、KIT……）。每个序列旁边的 `_stagei.npz` 只有体型，loader 跳过（`exclude_content`）。
- 采集帧率按子集不同（60–250 fps），参考跳变按各自的原始帧率比较。
- 没有文字。文字规则读受试者文件夹名和文件名拆成的词，有的子集写了动作（ACCAD、KIT、EyesJapanDataset……），CMU 等只有编号，
  这些只靠胯部高度、俯仰、速度分区。站上高处靠「Hips 连续 1 秒以上高于升降顶端」进地形（CMU 的不平地面、攀爬，KIT/912 等）；
  EKUT 的名字是动作代码，`beam`/`beamn`（架高的平衡木）、`slp`/`slpd`（踏上台子）、`wsu*`/`wsd*`（上下楼梯）按地形算
  （`classify.py` 的 `EKUT_TERRAIN`），它的推、走路几条留着。
- DFaust 的 `trans` 不在地面上：脚底在 z = 0 下面约 0.61 m（其他子集整条片段的网格最低点都在 ±4 cm 内），不处理会整个子集都判成
  贴地。`config/loader/amass.yaml` 的 `ground_subsets: [DFaust]` 让 loader 把这些片段抬起来，使片段自己的体型（betas、性别）在
  整条片段上的网格最低点落在 z = 0；survey 走同一个 loader，扫描、分区、转换都是抬过的。

## 0. 解压

```bash
mkdir -p $AMASS_OUT/bvh
for t in $AMASS/*.tar.bz2; do [ "$(basename $t)" = GRAB.tar.bz2 ] || tar -I lbzip2 -xf $t -C $AMASS_OUT/bvh; done
```

## 1–2. 扫描、分区

```bash
python scripts/amass/survey.py --out $AMASS_OUT/survey/survey_raw.parquet --workers 64
python scripts/amass/classify.py --survey $AMASS_OUT/survey/survey_raw.parquet --out_dir $AMASS_OUT/survey
```

`classify.py` 另外按子集打印各区时长。本机用 SSM、CNRS、HumanEva 三个小子集（137 条）试过：CNRS 是快走、转身（7 秒 10–12 m），
大多是黄区「底盘超速」；4 条 CNRS 手腕在一两个原始帧里转 130–240 rad/s（手的标记丢了拟合出来的），按参考跳变进红区。

## 3–6. 批量转换、检查、剔除 OOD

```bash
VERSION=v1 NOTE="first run" NUM_WORKERS=64 scripts/amass/retarget.sh    # 在 tmux 里跑
D=amass OUT=$AMASS_OUT V=v1
python scripts/$D/audit.py --run_dir $OUT/$V --out_dir $OUT/$V/audit --workers 64
python scripts/$D/jump_audit.py flips --workers 64           # 只看参考数据；loader 没改过（只是重转）可以跳过
python scripts/$D/jump_audit.py segments --version $V --workers 64
python scripts/$D/exclude_ood.py --version $V               # 只打印名单
python scripts/$D/exclude_ood.py --version $V --apply
python scripts/$D/browse.py --build_cache $V
```

`jump_audit.py flips` 按子集筛用 `--subsets CMU KIT`。

服务器 hkeai-test 是共享的 k8s 容器，别人的进程多时总线程数逼近 pod 的上限，检查步骤开 16 个进程会 fork 失败
（`Resource temporarily unavailable`，`jump_audit.py segments` 报 `BrokenProcessPool`），用 `--workers 8`。

2026-10 v1（5c89f05）的结果：共 15927 条 56.83 h，绿 7967 / 25.64 h，黄 5162 / 17.41 h，红 2798 / 13.78 h（贴地 4.56 h、
参考数据跳变 3.36 h、坐跪 2.54 h、地形 1.49 h、非直立 1.07 h、跳跃 0.75 h）。转了 13375 条，没有超限位，参考翻转 0 条，手腕误差
中位 4.8 cm。OOD 927 条 5.46 h：突跳 683 条 4.63 h（卡在关节限位 258、动作太快 302、转腕上限 67、参考跳变 40、标准 G1 也转不好
16，集中在 BMLrub 和 DanceDB），转完后加了站上高处的地形规则又移进 244 条 0.83 h。最终 12448 条 38.52 h。

v2（b9f2070）是最终版本，v1 作废。加了 `ground_subsets` 后重扫 DFaust 129 条（`survey_raw.parquet` 里只替换这些行；其他子集
loader 输出不变，不用重扫）、全表重新分区：绿 8058 / 25.70 h，黄 5193 / 17.45 h，红 2676 / 13.69 h（DFaust 绿 91、黄 31、红 7）。
按绿、黄名单全量转了 13251 条，和 v1 比少了 246 条站上高处（现在是红区地形），多了 122 条 DFaust，共有的 13129 条转换结果逐位
相同。检查：手腕误差中位 4.7 cm，没有超限位；参考翻转沿用 v1 时的 `ref_twist_flips.csv`（0 条）。突跳 697 条 4.63 h 进 OOD
（卡在关节限位 260、动作太快 306、转腕上限 70、参考跳变 45、标准 G1 也转不好 16）。最终 12554 条 38.60 h（绿 25.13 h，黄 13.47 h）。
