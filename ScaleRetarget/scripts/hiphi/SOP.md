# HiPHI → g1d

下载的原始数据 `HiPHI`（`data/*.tar.zst`、`metadata/`）和解压后的 `HiPHI_unpacked` 都只读。产物都在 `HiPHI_retarget`：
`survey/` 是分区和参考扫描，`vN/` 是一次批量转换。本机路径在 `/home/zerenluo/Space/bfm_datasets/` 下，换机器用环境变量
`HIPHI`、`HIPHI_UNPACKED`、`HIPHI_OUT` 指过去（`dataset.py`）。命令都在 `ScaleRetarget` 目录下、`conda activate scaleretarget`
之后执行。

流程和 BONES-SEED（`scripts/bones_seed/SOP.md`）一样，代码共用 `scaleretarget/pipeline/`。不同的地方：

- 先解压。一个 part 解开约 26.8 GB，32 个 part 一共约 830 GB，本机放不下，全量在服务器上跑（第 7 节）。
- 片段名 `rel = {frame}/{motion_id}`，例如 `Adjusting/Adjusting-adjust_0001`；镜像是 `__mirror`。
- 每个演员有好几套骨架标定（一次采集一套）。loader 把每条片段的骨架缩放到 SOMA Uniform 的比例，前臂扭转挪到手上，
  再按这条片段站立时的脚踝高度上下平移（`config/loader/hiphi.yaml`）。
- 演员 A116–A132 的全部片段（35252 条里 346 条，原片、镜像各 173，别的演员没有）骨骼坐标系和别的不一样：
  躯干和脖子（Spine–Spine3、Neck–Head）绕竖轴差半圈，大腿、小腿绕自身轴差半圈，脚绕踝到脚尖的轴差半圈，两只上臂绕
  自身轴差四分之一圈。关节位置看不出来，但朝向目标错了：躯干目标朝后、肩肘差 90°（g1d 腰扭 150°、两臂交叉），膝盖和
  脚踝的朝向目标反了（标准 G1 两腿往里拧、交叉）。loader 按几何量出来转回统一约定（`canonical_frames`），关节位置不变。g1d 只用
  Hips、Spine2 和手臂，腿和脚的修正不改变它的结果。
- 没有官方 G1 结果。面板「对比标准 G1」和突跳归因的对照，都是当场把同一条片段转到标准 G1（`hiphi_to_unitree_g1.yaml`）。

```
unpack.py  →  survey.py  →  classify.py  →  retarget.sh  →  audit.py
                                                 ↘  jump_audit.py flips
                                                    jump_audit.py segments  →  exclude_ood.py
```

## 0. 解压

按 part 解压，并在 `HiPHI_unpacked/bvh/{frame}/{motion_id}.bvh` 建指向 BVH 的相对链接，批量转换就读这个目录。
已经解完的 part 跳过；磁盘剩余不够这个 part 加 `--reserve_gb`（默认 50）时停下。

```bash
python scripts/hiphi/unpack.py --parts 1 2 12
python scripts/hiphi/unpack.py --all
python scripts/hiphi/unpack.py --link_only     # 只重建链接
```

## 1. 运动学扫描

每条片段用 HiphiLoader 按批量转换的方式读成 30 fps，算工作空间指标；同时记下骨架标定（BVH 骨长的哈希）、帧间隔、loader 的上下平移量、
转回的骨骼坐标系（`turned_frames`），以及参考手臂跳变的帧数。先在 30 fps 上找候选：上臂扭转翻转（上臂、前臂都 > 40 rad/s，
手 < 20 rad/s）和手翻转（手 > 40 rad/s，上臂、前臂都 < 20 rad/s），阈值同突跳归因（`scaleretarget/utils/jump_audit.py`）；
再回到原始 90 fps 看这一步里转得快的那根骨头（上臂 / 手），超过 100 rad/s 才算跳变（`ref_pops`）。真实的快手（甩水、拍腕、
抖前臂）在原始帧率上平滑、不超过约 80 rad/s；捕捉毛刺是一到三个原始帧里手翻过 120–180°（100–300 rad/s，手臂不动）。
只扫已解压的片段。一条约 5 秒（单进程）：part 1、2、12 共 3334 条，20 进程 13 分钟。

```bash
python scripts/hiphi/survey.py \
  --out /home/zerenluo/Space/bfm_datasets/HiPHI_retarget/survey/survey_raw.parquet --workers 20
```

## 2. 分区

规则和 BONES-SEED 一样（工作空间指标定黄区，贴地、跳跃、地形、非直立定红区），文字规则按 HiPHI 改了：

- HiPHI 每条只有一个 Frame-LU（如 `Self_motion-crawl`）和一句很长的自然语言描述。描述里很多词是别的意思（“rolling shoulders”、
  “sit down onto the box”），所以地形、非直立只看 Frame-LU；描述只在去掉 rolling / lying / rise up 之后补查非直立的词；跳跃两者都看。
- 坐地、跪坐、深蹲靠胯部高度判成「贴地」：Hips 只要有一帧低于 0.55 m。HiPHI 一条约 1 分钟，常有做到一半坐到地上一段再站起来的，这样的片段也进红区。
- 「坐跪」的文字只看 Frame-LU（`Posture-sit`、`Change_posture-sit_down`、`Body_movement-kneel` 等），描述里的 chair、bench 多半是搬的东西。
- 读不了的片段进红区，原因「骨架不支持」：少数 BVH 是简化骨架，脊柱只有 Spine、Spine1 两节，没有躯干目标跟的 Spine2；3334 条里有 22 条。
- 参考手臂有任何一帧跳变（第 1 步的两种）的片段进红区，原因「参考数据跳变」：跳变在数据里，转成什么样都跟不了。

```bash
python scripts/hiphi/classify.py \
  --survey /home/zerenluo/Space/bfm_datasets/HiPHI_retarget/survey/survey_raw.parquet \
  --out_dir /home/zerenluo/Space/bfm_datasets/HiPHI_retarget/survey
```

写出 `survey/hiphi_zones.parquet`（和 `.csv`）、`green.txt`、`yellow.txt`、`red.txt`。只覆盖扫过的片段，解压了新的 part 后重跑第 1、2 步。

## 3. 批量转换

和 BONES-SEED 一样：`VERSION` 不存在时必须是下一个编号，续跑跳过转过的片段和 OOD 里的，配置文件改过就要开新版本。

```bash
VERSION=v1 NOTE="first run" NUM_WORKERS=18 scripts/hiphi/retarget.sh
VERSION=v1 scripts/hiphi/retarget.sh          # 中断后续跑
scripts/hiphi/retarget.sh --panel             # 只开面板
```

40 条（共 39 分钟动作）20 进程约 70 秒。

面板和 BONES-SEED 的一样。「演员」一列是 actor_id，「类别」是 Frame-LU。「对比标准 G1」现场转换，要等十几秒才出窗口。

没开面板时，帧缓存（突跳名单从这里来）：

```bash
python scripts/hiphi/browse.py --build_cache v1
```

## 4. 质量检查

```bash
python scripts/hiphi/audit.py \
  --run_dir /home/zerenluo/Space/bfm_datasets/HiPHI_retarget/v1 \
  --out_dir /home/zerenluo/Space/bfm_datasets/HiPHI_retarget/v1/audit
```

## 5. 突跳归因

```bash
python scripts/hiphi/jump_audit.py flips
python scripts/hiphi/jump_audit.py flips --actors A001 A003     # 只扫这几个演员
python scripts/hiphi/jump_audit.py segments --version v1
```

`segments` 只算有突跳的片段，每条当场转一次标准 G1 作对照，约 12 秒。原因表见 BONES-SEED SOP 第 5 节，`ref_fails` 这里是「标准 G1 也转不好」。

## 6. 剔除 OOD

```bash
python scripts/hiphi/exclude_ood.py --version v1
python scripts/hiphi/exclude_ood.py --version v1 --apply
```

“一次采集”是同一个演员的同一套骨架标定（分区表的 `session` 列，镜像片段和原片段算同一次）。其余同 BONES-SEED。

## 7. 在服务器上跑

服务器 `ssh hkeai-test`（47.79.124.13，端口 31124，用户 luozr），代码在 `~/humanoid_wbc/ScaleBFM`，数据在
`~/humanoid_wbc/bfm_datasets`（`HiPHI`、`HiPHI_unpacked`、`HiPHI_retarget`，BONES-SEED 的 `bones-seed`、`bone-seed_retarget`
也在这里）。224 核，和别人共用，转换用 64–96 进程。服务器没有图形界面：`retarget.sh` 发现没有显示器就不开面板，改成在后台
每 60 秒更新一次帧缓存（`vN/cache.log`），转换结束时再更新一次。面板在本机通过 sshfs 挂载只读缓存。

第一次准备（环境和数据只做一次）：

```bash
ssh hkeai-test
cd ~/humanoid_wbc/ScaleBFM && git pull
# 环境：Python 3.11 + ScaleRetarget，解压用的 zstd，下载用的 huggingface_hub。
# /etc/pip.conf 里有一个这台机器解析不了的源，pip 加 --isolated 绕开。
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n scaleretarget -c conda-forge --override-channels python=3.11 zstd pip -y && conda activate scaleretarget
python -m pip --isolated install -e ScaleRetarget "huggingface_hub[hf_xet]"
# 数据：两个数据集都要先在网页上申请权限，再 hf auth login（或拷本机的 ~/.cache/huggingface/token）。约 200 MB/s。
D=~/humanoid_wbc/bfm_datasets
hf download noitomrobotics/HiPHI --repo-type dataset --local-dir $D/HiPHI
hf download bones-studio/seed soma_uniform.tar.gz g1.tar.gz metadata/seed_metadata_v004.parquet \
  --repo-type dataset --local-dir $D/bones-seed
cd $D/bones-seed && tar -xzf soma_uniform.tar.gz && tar -xzf g1.tar.gz
```

之后每次都先 `source ~/humanoid_wbc/bfm_env.sh`：设好两个数据集的路径变量、激活环境、进 `ScaleRetarget` 目录。内容是

```bash
export BFM=~/humanoid_wbc/bfm_datasets
export BONES_SEED=$BFM/bones-seed BONES_SEED_OUT=$BFM/bone-seed_retarget
export HIPHI=$BFM/HiPHI HIPHI_UNPACKED=$BFM/HiPHI_unpacked HIPHI_OUT=$BFM/HiPHI_retarget
export LAFAN=$BFM/lafan1 LAFAN_OUT=$BFM/lafan1_retarget
export GRAB=$BFM/grab GRAB_UNPACKED=$BFM/grab_unpacked GRAB_OUT=$BFM/grab_retarget
export OMOMO=$BFM/omomo OMOMO_UNPACKED=$BFM/omomo_unpacked OMOMO_OUT=$BFM/omomo_retarget
export AMASS=$BFM/amass AMASS_OUT=$BFM/amass_retarget
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate scaleretarget
cd ~/humanoid_wbc/ScaleBFM/ScaleRetarget
```

线程数那一行不能省：这台机器（容器）限制所有用户加起来的线程总数（约 16k，别人的任务常占 8k 左右），numpy / torch 默认
按 224 核开线程池，每个进程约 130 个线程，64 个进程就撞上限。撞上时 MuJoCo 加载模型报 `Caught an unknown exception!`，
严重时 luozr 的所有进程连同 tmux 一起被杀。设成 1 以后每个进程约 1 个线程，两个数据集各开 96 进程同时跑也只有几百个线程。

然后：

```bash
python scripts/hiphi/unpack.py --all          # 一个 part 接一个 part；32 个 part 约 860 GB
python scripts/hiphi/survey.py --out $HIPHI_OUT/survey/survey_raw.parquet --workers 96
python scripts/hiphi/classify.py --survey $HIPHI_OUT/survey/survey_raw.parquet --out_dir $HIPHI_OUT/survey
VERSION=v1 NOTE="first run" NUM_WORKERS=64 scripts/hiphi/retarget.sh    # 建议在 tmux 里跑
```

本机看进度、播放：

```bash
mkdir -p ~/mnt/hkeai && sshfs hkeai-test:/home/luozr ~/mnt/hkeai -o reconnect,ServerAliveInterval=15,max_read=1048576,idmap=user
R=~/mnt/hkeai/humanoid_wbc/bfm_datasets
HIPHI=$R/HiPHI HIPHI_UNPACKED=$R/HiPHI_unpacked HIPHI_OUT=$R/HiPHI_retarget \
python scripts/hiphi/browse.py --version v1 --cache_only
```

`--cache_only` 只读服务器写的 `vN/frame_counts.csv` 和 `vN/ood/frame_counts.csv`，不在挂载上遍历 pkl；标题栏会显示「只读缓存」，
进度行显示缓存最后更新的时间。右键剔除 / 恢复照常（直接在挂载上移动文件，以 `vN/ood/ood.csv` 为准）。播放时 BVH 也走挂载读：
挂载读约 3 MB/s（不加 `max_read` 只有约 2 MB/s，开压缩没用），HiPHI 的 BVH 一条 20 MB 上下，点开要等 7 秒左右。
不用了 `fusermount -u ~/mnt/hkeai`；已经挂着时再挂一次会报 `failed to access mountpoint: Permission denied`，先卸再挂。

第 4–6 步在服务器上照常跑（同样先设路径），`--workers` 可以开到 64。

### 常用命令（各数据集通用）

服务器上长任务都放 tmux 里，断网、关终端不影响：

```bash
ssh hkeai-test
tmux new -s rt_hiphi              # 新开会话；已经有了就 tmux attach -t rt_hiphi
source ~/humanoid_wbc/bfm_env.sh
# ……在这里跑命令……
# Ctrl-b 再按 d：离开会话，任务继续跑；Ctrl-b [ 往上翻输出，q 退出
tmux ls                           # 有哪些会话
tmux attach -t rt_hiphi           # 回到会话
```

转换（第 3 步，`VERSION` 换成下一个编号）：

```bash
VERSION=v2 NOTE="说明这一版改了什么" NUM_WORKERS=64 scripts/hiphi/retarget.sh
VERSION=v2 NOTE="说明这一版改了什么" NUM_WORKERS=64 scripts/bones_seed/retarget.sh
VERSION=v1 NOTE="说明这一版改了什么" NUM_WORKERS=64 scripts/lafan/retarget.sh
VERSION=v1 NOTE="说明这一版改了什么" NUM_WORKERS=64 scripts/grab/retarget.sh     # omomo、amass 同样
```

转完以后的检查和剔除（第 4–6 步）。先设好是哪个数据集、哪个版本，下面几行原样复制：

```bash
D=hiphi OUT=$HIPHI_OUT V=v2               # BONES-SEED：D=bones_seed OUT=$BONES_SEED_OUT V=v1；LAFAN：D=lafan OUT=$LAFAN_OUT V=v1；GRAB：D=grab OUT=$GRAB_OUT V=v1（omomo、amass 同样）
python scripts/$D/audit.py --run_dir $OUT/$V --out_dir $OUT/$V/audit --workers 64
python scripts/$D/jump_audit.py flips --workers 64          # 只看参考数据；loader 没改过可以跳过
python scripts/$D/jump_audit.py segments --version $V --workers 64
python scripts/$D/exclude_ood.py --version $V               # 只打印名单，不动文件
```

看过名单（面板里也能逐条看）再移动，移完更新帧缓存，本机面板才看得到：

```bash
python scripts/$D/exclude_ood.py --version $V --apply
python scripts/$D/browse.py --build_cache $V
```

`--apply` 重跑是安全的：已经在 OOD 的跳过。但面板里手动恢复的片段没有记录，再 `--apply` 会被重新剔除。

本机看面板（在 `ScaleRetarget` 目录下；每条命令一行，换行要在行尾加 `\`）：

```bash
mkdir -p ~/mnt/hkeai && sshfs hkeai-test:/home/luozr ~/mnt/hkeai -o reconnect,ServerAliveInterval=15,max_read=1048576,idmap=user
R=~/mnt/hkeai/humanoid_wbc/bfm_datasets
HIPHI=$R/HiPHI HIPHI_UNPACKED=$R/HiPHI_unpacked HIPHI_OUT=$R/HiPHI_retarget python scripts/hiphi/browse.py --version v2 --cache_only
BONES_SEED=$R/bones-seed BONES_SEED_OUT=$R/bone-seed_retarget python scripts/bones_seed/browse.py --version v1 --cache_only
LAFAN=$R/lafan1 LAFAN_OUT=$R/lafan1_retarget python scripts/lafan/browse.py --version v1 --cache_only
GRAB_OUT=$R/grab_retarget python scripts/grab/browse.py --version v1 --cache_only
OMOMO_OUT=$R/omomo_retarget python scripts/omomo/browse.py --version v1 --cache_only
AMASS_OUT=$R/amass_retarget python scripts/amass/browse.py --version v1 --cache_only
fusermount -u ~/mnt/hkeai         # 不用了卸载
```

面板里右键剔除 / 恢复单条，直接改挂载上的文件，不用再跑别的。整批按规则剔除在服务器上跑 `exclude_ood.py`。

## 8. 重新标定 loader 和躯干目标的数

`config/loader/hiphi.yaml` 里的四组数，以及两张 HiPHI 对应表里 torso_link（Spine2）那一行的旋转和位置偏移，都可以重测，
脚本把测出的值和当前值并排打印（偏移要手动抄进 `hiphi_to_unitree_g1d.yaml` 和 `hiphi_to_unitree_g1.yaml`，两张表一样）：

```bash
python scripts/hiphi/calibrate_offsets.py
```

- `reference_distances`：loader 在每条动作的静止 T-pose 上把各段缩放到这些长度，所以同一个演员的每条动作缩放倍数一样，
  不随动作内容变。四肢和大腿根：SOMA Uniform 各段长度（随机 200 条 BONES-SEED 动作，帧上取 90 百分位；都是定长，换一批动作
  测出来完全一样）。肩和大腿根（Arm、UpLeg）在 SOMA 上从两髋关节中点量，因为 HiPHI 的 Hips 就在这个中点；从 SOMA 的 Hips
  量的话，HiPHI 的髋会被拉宽约一半。
- 胯到肩（Arm）随脊柱和肩胛动作变，不能直接拿 SOMA 的 90 百分位（抬手、耸肩的帧最长，用它会把走路时的上身拉长约 8%）。
  改成：T-pose 上取多长，HiPHI 走路时胯到肩的中位数才和 SOMA 走路一样（56 条 HiPHI 走路 vs 200 条 SOMA 走路，
  52.5 / 52.2 cm）。HiPHI 的 T-pose：四肢逐帧长度恒定，用实际值（和文件头略有出入）；脊柱、颈、头的长度求解器会按姿态
  压短最多约 1 cm（蜷缩类最短），同一演员不同动作差到 3 cm，所以用文件头的长度（约等于最长），每个演员一个定值。
  改了 Arm 之后躯干位置偏移跟着变，要再跑一遍脚本，第二遍 Arm 应该不变。
- `hips_offset`：SOMA 的 Hips 相对两髋关节中点的位置（同一批动作的中位数：上方 8.4 cm、后方 2.6 cm），换到 HiPHI Hips 的
  坐标系里。loader 缩放完后把 HiPHI 的 Hips 点挪到这里，机器人底盘和升降、以及上身缩放的原点就和 BONES-SEED 跟的是身上同一个点；
  不挪的话升降要低约 7 cm。
- `planted_ankle_height`：SOMA Uniform 直腿站立时脚踝高度的中位数（0.064 m）。
- `typical_planted_ankle_height`：HiPHI 同一测法的中位数，从第 1 步扫描结果算，给极少数从不直腿站立的片段用（0.072 m；移 Hips 之前是 0.103 m）。
- torso_link 旋转偏移：HiPHI 静止姿态的偏移再往前转一个角度，使走路时"胸口目标比躯干线（两髋关节中点 → 两肩中点）靠前多少"和
  SOMA Chest 一样（各取直立帧的中位数，2026-10 用 1/2/7/12/27 包的 56 条走路量出 +5.0°）。躯干线不从 Hips 起：
  两边 Hips 原本不在身上同一个点，从 Hips 起量会让两边差出约 3°。
- torso_link 位置偏移：让胸口目标相对两肩目标的位置和 SOMA 一样（随机动作的中位数，在胸口目标坐标系里量）。
- 选 Spine2 而不是 Spine4 的原因：Spine4 挂着两肩，抬手、耸肩时会跟着转，胸口目标约 1/5 的帧后仰超过 G1-D 腰的极限；
  Spine2 在前后弯、侧弯、扭腰时跟躯干的关系和 SOMA Chest 最接近。
