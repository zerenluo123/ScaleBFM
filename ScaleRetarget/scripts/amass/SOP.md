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
  这些只靠胯部高度、俯仰、速度分区。

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
python scripts/$D/jump_audit.py flips --workers 64
python scripts/$D/jump_audit.py segments --version $V --workers 64
python scripts/$D/exclude_ood.py --version $V               # 只打印名单
python scripts/$D/exclude_ood.py --version $V --apply
python scripts/$D/browse.py --build_cache $V
```

`jump_audit.py flips` 按子集筛用 `--subsets CMU KIT`。
