"""Browse a dataset's clips by g1d zone, follow the batch retarget progress, and play clips in MuJoCo.

Each batch run is a version folder <out>/vN (bvh/<rel>.pkl, run.json, config snapshot); the version
picker switches progress, list and playback between them. Converted green and yellow clips play the
retargeted pkl with the IK targets; red clips play only the reference targets. Progress counts the real
frames stored in each pkl and refreshes while the batch run is going; the arm joint speed columns are
computed from the same pkls, so they follow the selected version.
The reference comparison opens our retarget next to a standard G1 retarget of the same clip: the
dataset's official one if it ships one, otherwise ours, retargeted live. "对比版本" opens the same clip
from two versions.
Right-click "剔除OOD数据" moves converted green / yellow clips to the OOD zone: their pkl goes from
vN/bvh/ to vN/ood/bvh/ (logged in vN/ood/ood.csv), so the dataset folder no longer holds them and the
batch run does not convert them again; in the OOD zone "恢复到原数据区" moves them back.

With --cache_only the panel reads progress from the frame caches (frames.py) instead of walking the
version folder, for a version on a network mount whose caches build_cache --loop keeps current on the
machine that runs the batch; OOD membership then comes from vN/ood/ood.csv.
"""

import argparse
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton, QTableView, QVBoxLayout,
    QWidget,
)

from scaleretarget.pipeline.dataset import ROOT
from scaleretarget.pipeline.frames import arm_columns, build_cache, counters, list_versions, read_pkl, run_info
from scaleretarget.pipeline.workspace import REASON_EN
from scaleretarget.utils.ood import OOD_DIR, find_pkl, move_to_ood, read_ood_log, restore_from_ood

REFRESH_MS = 15000

ZONES = [("green", "绿区", "#2e9e44"), ("yellow", "黄区", "#d49a12"), ("red", "红区", "#c0392b"),
         ("ood", "OOD区", "#6e40c9")]
ZONE_LABEL = {k: l for k, l, _ in ZONES}
SPIKE_MODES = {"全部": None, "> 20 rad/s": ("frames_gt20", "> 20 rad/s"), "> 30 rad/s": ("frames_gt30", "> 30 rad/s")}
STRETCH_COLUMN = "description"

STYLE = """
QWidget { font-family: "Noto Sans CJK SC"; font-size: 10pt; color: #1f2328; }
QMainWindow, QWidget#central { background: #f4f6f8; }
QGroupBox { background: white; border: 1px solid #e1e4e8; border-radius: 12px; margin-top: 16px; padding: 12px 10px 8px 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 6px; color: #57606a; }
QPushButton { background: white; border: 1px solid #d0d7de; border-radius: 9px; padding: 5px 16px; }
QPushButton:hover { background: #f0f3f6; }
QPushButton:pressed { background: #e4e8ec; }
QLineEdit, QComboBox { background: white; border: 1px solid #d0d7de; border-radius: 9px; padding: 4px 10px; min-height: 22px; }
QLineEdit:focus, QComboBox:focus { border: 1px solid #0969da; }
QComboBox::drop-down { border: none; width: 22px; }
QProgressBar { background: #eaeef2; border: none; border-radius: 6px; max-height: 12px; min-height: 12px; }
QTableView { background: white; border: 1px solid #e1e4e8; border-radius: 12px; gridline-color: #f1f3f5;
             selection-background-color: #dbeafe; selection-color: #1f2328; alternate-background-color: #fafbfc; }
QHeaderView::section { background: #f6f8fa; border: none; border-bottom: 1px solid #e1e4e8; padding: 6px 8px; font-weight: 600; }
QStatusBar { color: #57606a; }
"""


def table_columns(dataset):
    return [
        ("filename", "文件", 300),
        ("group", dataset.group_label, 90),
        ("seconds", "时长(s)", 70),
        ("reason", "原因", 110),
        ("category", "类别", 170),
        ("description", "描述", 380),
        ("vmax", "手臂关节速度峰值", 120),
        ("spike_frames", "突跳帧", 190),
        ("status", "状态", 80),
    ]


class ClipModel(QAbstractTableModel):
    def __init__(self, df, columns):
        super().__init__()
        self.df = df
        self.columns = columns
        self.index_ = np.array([], dtype=int)
        self.spike_col = "frames_gt20"

    def set_rows(self, index):
        self.beginResetModel()
        self.index_ = np.asarray(index)
        self.endResetModel()

    def refresh_status(self):
        if len(self.index_):
            col = len(self.columns) - 1
            self.dataChanged.emit(self.index(0, 0), self.index(len(self.index_) - 1, col))

    def rowCount(self, parent=QModelIndex()):
        return len(self.index_)

    def columnCount(self, parent=QModelIndex()):
        return len(self.columns)

    def row(self, r):
        return self.df.loc[self.index_[r]]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        idx = self.index_[index.row()]
        col = self.columns[index.column()][0]
        if role == Qt.DisplayRole:
            if col == "status":
                if self.df.at[idx, "zone"] == "ood":
                    return f"剔除自{ZONE_LABEL[self.df.at[idx, 'orig_zone']]}"
                return status_text(self.df.at[idx, "zone"], self.df.at[idx, "converted"])
            if col == "reason" and self.df.at[idx, "zone"] == "ood":
                return self.df.at[idx, "ood_reason"] or self.df.at[idx, "reason"]
            if col == "seconds":
                return f"{self.df.at[idx, 'seconds']:.1f}"
            if col == "vmax":
                return f"{self.df.at[idx, 'vmax']:.1f} rad/s" if self.df.at[idx, "converted"] else ""
            if col == "spike_frames":
                return self.df.at[idx, self.spike_col]
            return str(self.df.at[idx, col])
        if role == Qt.ForegroundRole and self.df.at[idx, "zone"] != "red" and not self.df.at[idx, "converted"]:
            return QColor("#9aa1a9")
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            if self.columns[section][0] == "spike_frames":
                return "突跳帧 > 30 rad/s" if self.spike_col == "frames_gt30" else "突跳帧 > 20 rad/s"
            return self.columns[section][1]
        return None

    def sort(self, column, order=Qt.AscendingOrder):
        col = self.columns[column][0]
        key = {"status": "converted", "spike_frames": "vmax"}.get(col, col)
        rows = self.df.loc[self.index_, key]
        self.set_rows(rows.sort_values(ascending=order == Qt.AscendingOrder, kind="stable").index)


def status_text(zone, converted):
    if zone == "red":
        return "仅参考"
    return "已转换" if converted else "未转换"


class Browser(QMainWindow):
    def __init__(self, dataset, args):
        super().__init__()
        self.dataset = dataset
        self.args = args
        self.root = dataset.out
        self.reference = "官方 G1" if dataset.official_csv_root else "标准 G1"
        self.players = []
        self.columns = {}
        self.counter = None
        self.version = None
        self.scan_result = None
        self.scanning = False
        self.history = []
        self.zone = "green"
        # Bumped by every OOD move, so a scan that started before it does not undo it on screen.
        self.moves = 0

        df = dataset.load_zones()
        df["orig_zone"] = df.zone
        df["ood_reason"] = ""
        for col in ("group", "category", "description"):
            df[col] = df[col].fillna("").astype(str)
        self.df = df
        self.total_span = df.span_expected.sum()

        title = f"{dataset.name} → g1d"
        self.setWindowTitle(title + ("（只读缓存）" if args.cache_only else ""))
        self.resize(1500, 950)
        central = QWidget(objectName="central")
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.addLayout(self.build_version_bar())
        layout.addWidget(self.build_progress())
        layout.addLayout(self.build_toolbar())
        layout.addWidget(self.build_table(), stretch=1)
        self.setCentralWidget(central)

        versions = list_versions(self.root)
        if not versions:
            raise SystemExit(f"no version folders (v1, v2, ...) with a bvh/ folder under {self.root}")
        self.refresh_versions()
        self.set_version(args.version if args.version in versions else versions[-1])
        self.select_zone("green")
        self.poll_timer = QTimer(self, timeout=self.poll_scan, interval=500)
        self.poll_timer.start()

    # ---------- versions ----------

    def build_version_bar(self):
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 4, 0, 0)
        bar.addWidget(QLabel("结果版本"))
        self.version_box = QComboBox()
        self.version_box.setMinimumWidth(90)
        self.version_box.activated.connect(lambda _: self.set_version(self.version_box.currentText()))
        bar.addWidget(self.version_box)
        self.version_info = QLabel()
        self.version_info.setStyleSheet("color: #57606a;")
        bar.addWidget(self.version_info, stretch=1)
        bar.addWidget(QLabel("对比版本"))
        self.compare_box = QComboBox()
        self.compare_box.setMinimumWidth(90)
        bar.addWidget(self.compare_box)
        return bar

    def refresh_versions(self):
        versions = list_versions(self.root)
        for box in (self.version_box, self.compare_box):
            current = box.currentText()
            if [box.itemText(i) for i in range(box.count())] != versions:
                box.clear()
                box.addItems(versions)
            if current in versions:
                box.setCurrentText(current)
        if self.compare_box.currentText() == self.version and len(versions) > 1:
            self.compare_box.setCurrentText(next(v for v in reversed(versions) if v != self.version))

    def version_dir(self, version=None):
        return os.path.join(self.root, version or self.version)

    def set_version(self, version):
        info = run_info(self.version_dir(version))
        robot = info.get("robot", self.args.robot)
        if robot not in self.columns:
            self.columns[robot] = arm_columns(robot)
        self.version, self.robot = version, robot
        self.fps = self.args.fps or info.get("fps", 30)
        self.version_box.setCurrentText(version)
        self.refresh_versions()
        parts = [f"{self.fps} fps", robot, f"创建于 {info['created']}" if "created" in info else "", info.get("note", "")]
        self.version_info.setText("    ·    ".join(p for p in parts if p))
        self.progress_box.setTitle(f"批量转换进度    {self.version_dir()}")

        df = self.df
        df["frames_expected"] = self.dataset.expected_frames(df.span_expected, self.fps)
        df["frames_done"] = 0
        df["span_done"] = 0.0
        df["converted"] = False
        df["vmax"] = 0.0
        df["frames_gt20"] = df["frames_gt30"] = ""
        df["zone"] = df.orig_zone
        df["ood_reason"] = ""
        self.counter, self.ood_counter = counters(self.version_dir(), self.columns[robot], self.args.cache_only)
        self.history = []
        self.rate_label.setText(f"正在统计 {version} 已转换的 pkl ...")
        self.scan_result = None
        self.start_scan()
        if hasattr(self, "model"):
            self.populate()

    # ---------- layout ----------

    def build_progress(self):
        box = QGroupBox("批量转换进度")
        self.progress_box = box
        grid = QGridLayout(box)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(6)
        heads = ["", "已转时长 / 总时长", "已转帧数 / 总帧数（实际 pkl 帧数）", "已转占全部数据 / 该区占全部数据"]
        for col, head in enumerate(heads):
            label = QLabel(head)
            label.setStyleSheet("color: #57606a; font-weight: 600;")
            grid.addWidget(label, 0, col)
        self.zone_buttons, self.bars = {}, {}
        for row, (key, label, color) in enumerate(ZONES, start=1):
            button = QPushButton(label)
            button.setFixedSize(96, 40)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda _=False, k=key: self.select_zone(k))
            self.zone_buttons[key] = button
            grid.addWidget(button, row, 0)
            cells = []
            for col in range(1, 4):
                cell = QVBoxLayout()
                cell.setSpacing(3)
                bar = QProgressBar(textVisible=False, maximum=1000)
                bar.setStyleSheet(f"QProgressBar::chunk {{ background: {color}; border-radius: 6px; }}")
                text = QLabel()
                text.setStyleSheet("color: #3b434b;")
                cell.addWidget(bar)
                cell.addWidget(text)
                grid.addLayout(cell, row, col)
                cells.append((bar, text))
            self.bars[key] = cells
        for col in range(1, 4):
            grid.setColumnStretch(col, 1)
        self.rate_label = QLabel()
        self.rate_label.setStyleSheet("color: #57606a;")
        grid.addWidget(self.rate_label, len(ZONES) + 1, 0, 1, 4)
        return box

    def build_toolbar(self):
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 8, 0, 4)
        bar.addWidget(QLabel("搜索"))
        self.query = QLineEdit(placeholderText="文件名 / 类别 / 描述，回车筛选")
        self.query.setFixedWidth(300)
        self.query.returnPressed.connect(self.populate)
        bar.addWidget(self.query)
        bar.addSpacing(10)
        bar.addWidget(QLabel("原因"))
        self.reason = QComboBox()
        self.reason.setMinimumWidth(150)
        self.reason.activated.connect(lambda _: self.populate())
        bar.addWidget(self.reason)
        bar.addSpacing(10)
        bar.addWidget(QLabel("手臂关节突跳"))
        self.spike_mode = QComboBox()
        self.spike_mode.addItems(list(SPIKE_MODES))
        self.spike_mode.activated.connect(lambda _: self.populate())
        bar.addWidget(self.spike_mode)
        bar.addSpacing(10)
        self.only_converted = QCheckBox("只看已转换")
        self.only_converted.toggled.connect(lambda _: self.populate())
        bar.addWidget(self.only_converted)
        bar.addSpacing(10)
        for text, slot in [("刷新列表", self.populate), ("播放", self.play_selected),
                           (f"对比{self.reference}", self.compare_reference), ("对比版本", self.compare_versions)]:
            button = QPushButton(text)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(slot)
            bar.addWidget(button)
        bar.addStretch(1)
        return bar

    def build_table(self):
        columns = table_columns(self.dataset)
        self.model = ClipModel(self.df, columns)
        view = QTableView()
        view.setModel(self.model)
        view.setSortingEnabled(True)
        view.setAlternatingRowColors(True)
        view.setSelectionBehavior(QAbstractItemView.SelectRows)
        view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        view.setContextMenuPolicy(Qt.CustomContextMenu)
        view.customContextMenuRequested.connect(self.show_row_menu)
        view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        view.setShowGrid(False)
        view.verticalHeader().setVisible(False)
        view.verticalHeader().setDefaultSectionSize(26)
        header = view.horizontalHeader()
        for i, (col, _, width) in enumerate(columns):
            view.setColumnWidth(i, width)
            if col == STRETCH_COLUMN:
                header.setSectionResizeMode(i, QHeaderView.Stretch)
        header.setSortIndicatorShown(False)
        view.doubleClicked.connect(lambda _: self.play_selected())
        QShortcut(QKeySequence(Qt.Key_Return), view, activated=self.play_selected)
        self.view = view
        return view

    def style_zone_buttons(self):
        for key, _, color in ZONES:
            if key == self.zone:
                css = f"background: {color}; color: white; border: none;"
            else:
                css = f"background: white; color: {color}; border: 2px solid {color};"
            self.zone_buttons[key].setStyleSheet(
                f"QPushButton {{ {css} border-radius: 14px; font-size: 12pt; font-weight: 700; }}")

    # ---------- progress ----------

    def start_scan(self):
        if self.scanning:
            return
        self.scanning = True
        counter, ood_counter, version, moves = self.counter, self.ood_counter, self.version, self.moves

        def work():
            try:
                result = (counter.scan(), ood_counter.scan(), read_ood_log(self.version_dir(version)))
                self.scan_result = (version, moves, result)
            finally:
                self.scanning = False

        threading.Thread(target=work, daemon=True).start()

    def poll_scan(self):
        if self.scan_result is None:
            return
        (version, moves, result), self.scan_result = self.scan_result, None
        versions = list_versions(self.root)
        if self.version not in versions and versions:  # its folder was deleted
            self.set_version(versions[-1])
        elif version == self.version and moves == self.moves:
            self.apply_scan(*result)
            QTimer.singleShot(REFRESH_MS, self.start_scan)
        else:
            # Finished a scan of the previous version, or one from before an OOD move; scan again now.
            self.start_scan()
        self.refresh_versions()

    def apply_scan(self, entries, ood_entries, ood_log):
        df = self.df
        # The caches lag behind moves made here until the next rebuild; the log is written by the move itself.
        is_ood = df.rel.isin(ood_log.rel) if self.args.cache_only else df.rel.isin(ood_entries.keys())
        df["zone"] = df.orig_zone.where(~is_ood, "ood")
        df["ood_reason"] = df.rel.map(dict(zip(ood_log.rel, ood_log.reason))).fillna("").where(is_ood, "")
        entries = {**entries, **ood_entries}
        hit = df.rel.isin(entries.keys())
        rows = [entries[r] for r in df.rel[hit]]
        frames = pd.Series(0, index=df.index)
        fps = pd.Series(np.nan, index=df.index)
        df["vmax"] = 0.0
        df["frames_gt20"] = df["frames_gt30"] = ""
        if rows:
            frames[hit] = [r[0] for r in rows]
            fps[hit] = [r[1] for r in rows]
            df.loc[hit, "vmax"] = [r[3] for r in rows]
            df.loc[hit, "frames_gt20"] = [r[4] for r in rows]
            df.loc[hit, "frames_gt30"] = [r[5] for r in rows]
        df["converted"] = frames > 0
        df["frames_done"] = frames
        df["span_done"] = ((frames - 1) / fps).where(df.converted, 0.0)
        self.update_progress()

        if self.spike_mode.currentText() != "全部":
            self.populate()  # spike lists of newly converted clips change the filtered rows
        else:
            self.model.refresh_status()
            self.update_status_line()

    def update_progress(self):
        df = self.df
        for key, _, _ in ZONES:
            z = df[df.zone == key]
            (b1, t1), (b2, t2), (b3, t3) = self.bars[key]
            if key == "ood":
                self.update_ood_progress(z, b1, t1, b2, t2, b3, t3)
                continue
            span_total, span_done = z.span_expected.sum(), z.span_done.sum()
            frames_total, frames_done = int(z.frames_expected.sum()), int(z.frames_done.sum())
            share_total, share_done = span_total / self.total_span, span_done / self.total_span
            b1.setValue(round(1000 * span_done / max(span_total, 1e-9)))
            b2.setValue(round(1000 * frames_done / max(frames_total, 1)))
            b3.setValue(round(1000 * share_done / max(share_total, 1e-9)))
            note = "    红区不转换" if key == "red" else ""
            t1.setText(f"{span_done / 3600:.1f} h / {span_total / 3600:.1f} h    "
                       f"{span_done / max(span_total, 1e-9):.1%}{note}")
            t2.setText(f"{frames_done:,} / {frames_total:,}    {frames_done / max(frames_total, 1):.1%}    "
                       f"{int(z.converted.sum()):,} / {len(z):,} 段")
            t3.setText(f"{share_done:.1%} / {share_total:.1%}")

        now, done = time.time(), int(df.frames_done.sum())
        self.history = [(t, f) for t, f in self.history if now - t < 300] + [(now, done)]
        target = df.zone.isin(["green", "yellow"])
        remaining = int(df.frames_expected[target].sum() - df.frames_done[target].sum())
        t0, f0 = self.history[0]
        refresh = f"每 {REFRESH_MS // 1000} 秒自动刷新"
        if self.args.cache_only and os.path.isfile(self.counter.cache_path):
            stamp = time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(self.counter.cache_path)))
            refresh += f"（读缓存，缓存最后更新 {stamp}）"
        if now - t0 > 1 and done > f0:
            rate = (done - f0) / (now - t0)
            self.rate_label.setText(f"近 5 分钟速度 {rate:,.0f} 帧/秒    绿黄区还剩 {remaining:,} 帧    "
                                    f"预计 {remaining / rate / 3600:.1f} h    ·    {refresh}")
        else:
            idle = "    批量转换未在进行" if len(self.history) > 1 else ""
            self.rate_label.setText(f"绿黄区还剩 {remaining:,} 帧{idle}    ·    {refresh}")

    def update_ood_progress(self, z, b1, t1, b2, t2, b3, t3):
        """OOD clips against everything converted in green and yellow, including the OOD clips."""
        pool = self.df[self.df.zone.isin(["green", "yellow", "ood"]) & self.df.converted]
        span, frames = z.span_done.sum(), int(z.frames_done.sum())
        span_pool, frames_pool = max(pool.span_done.sum(), 1e-9), max(int(pool.frames_done.sum()), 1)
        share = span / self.total_span
        b1.setValue(round(1000 * span / span_pool))
        b2.setValue(round(1000 * frames / frames_pool))
        b3.setValue(round(1000 * share))
        t1.setText(f"{span / 3600:.2f} h    占绿黄区已转 {span / span_pool:.2%}")
        t2.setText(f"{frames:,}    {len(z):,} 段（原绿区 {int((z.orig_zone == 'green').sum()):,}，"
                   f"原黄区 {int((z.orig_zone == 'yellow').sum()):,}）")
        t3.setText(f"{share:.2%}    右键剔除 / 恢复，pkl 移到 {OOD_DIR}/bvh/")

    # ---------- list ----------

    def select_zone(self, zone):
        self.zone = zone
        self.style_zone_buttons()
        reasons = sorted(r for r in self.df.loc[self.df.zone == zone, self.reason_col()].unique() if r)
        self.reason.clear()
        self.reason.addItems(["全部", *reasons])
        self.populate()

    def reason_col(self):
        return "ood_reason" if self.zone == "ood" else "reason"

    def populate(self):
        df = self.df[self.df.zone == self.zone]
        if self.reason.currentText() not in ("", "全部"):
            df = df[df[self.reason_col()] == self.reason.currentText()]
        if self.only_converted.isChecked():
            df = df[df.converted]
        q = self.query.text().strip()
        if q:
            hay = df.filename + " " + df.category + " " + df.description
            df = df[hay.str.contains(q, case=False, regex=False)]
        mode = SPIKE_MODES[self.spike_mode.currentText()]
        self.model.spike_col = mode[0] if mode else "frames_gt20"
        self.model.headerDataChanged.emit(Qt.Horizontal, 0, self.model.columnCount() - 1)
        if mode:
            df = df[df[mode[0]] != ""]
            self.model.set_rows(df.sort_values("vmax", ascending=False).index)
        else:
            self.model.set_rows(df.sort_values("filename").index)
        self.update_status_line()

    def update_status_line(self):
        shown = self.model.index_
        converted = int(self.df.loc[shown, "converted"].sum()) if len(shown) else 0
        self.statusBar().showMessage(
            f"{self.version} · {ZONE_LABEL[self.zone]}：列出 {len(shown):,} 段，其中已转换 {converted:,}     |     "
            f"双击或回车播放，“对比{self.reference}”/“对比版本”同时开两个窗口；右键剔除到 OOD 区 / 从 OOD 区恢复（可多选）；"
            f"播放窗口里 空格 暂停，R 重播，←/→ 暂停并逐帧，↑ 跳到下一处突跳前 1 秒，C 视角跟随开关，T 机器人半透明")

    # ---------- playback ----------

    def selected_row(self):
        rows = self.view.selectionModel().selectedRows()
        return self.model.row(rows[0].row()) if rows else None

    # ---------- OOD ----------

    def show_row_menu(self, pos):
        index = self.view.indexAt(pos)
        if not index.isValid():
            return
        if not self.view.selectionModel().isRowSelected(index.row(), QModelIndex()):
            self.view.selectRow(index.row())
        rows = [self.model.row(i.row()) for i in self.view.selectionModel().selectedRows()]
        menu = QMenu(self)
        if self.zone == "ood":
            action = menu.addAction(f"恢复到原数据区（{len(rows)} 段）")
            action.triggered.connect(lambda: self.restore_rows(rows))
        elif self.zone in ("green", "yellow"):
            ready = [r for r in rows if r.converted]
            action = menu.addAction(f"剔除OOD数据（{len(ready)} 段）" if ready else "剔除OOD数据（只能剔除已转换的段）")
            action.setEnabled(bool(ready))
            action.triggered.connect(lambda: self.exclude_rows(ready))
        else:
            return
        menu.exec(self.view.viewport().mapToGlobal(pos))

    def exclude_rows(self, rows):
        done, failed = 0, []
        for row in rows:
            try:
                move_to_ood(self.version_dir(), row.rel, row.zone, "手动剔除")
            except OSError as e:  # e.g. the clip is being rewritten by an overwrite run
                failed.append(f"{row.filename}: {e}")
                continue
            self.df.at[row.name, "zone"] = "ood"
            self.df.at[row.name, "ood_reason"] = "手动剔除"
            done += 1
        self.after_move(f"已把 {done} 段移到 OOD 区（{self.version}/{OOD_DIR}/bvh/）", failed)

    def restore_rows(self, rows):
        done, failed = 0, []
        for row in rows:
            try:
                restore_from_ood(self.version_dir(), row.rel)
            except OSError as e:
                failed.append(f"{row.filename}: {e}")
                continue
            self.df.at[row.name, "zone"] = row.orig_zone
            self.df.at[row.name, "ood_reason"] = ""
            done += 1
        self.after_move(f"已把 {done} 段恢复到原数据区（{self.version}/bvh/）", failed)

    def after_move(self, message, failed):
        self.moves += 1
        self.update_progress()
        self.populate()
        self.statusBar().showMessage(message, 8000)
        if failed:
            QMessageBox.warning(self, "部分没有移动", "\n".join(failed[:20]))

    def spike_mode_cols(self):
        mode = SPIKE_MODES[self.spike_mode.currentText()] or SPIKE_MODES["> 20 rad/s"]
        return mode[0], mode[1]

    def player_cmd(self, row, robot, title):
        return [sys.executable, "-m", "scaleretarget.viewer.clip_player",
                "--source", self.dataset.bvh_path(row.rel),
                "--robot", robot, "--loader", self.dataset.loader, "--title", title]

    def version_cmd(self, row, version):
        """Player command for the pkl of a version, with its arm joint spikes marked; None if not converted."""
        pkl = find_pkl(self.version_dir(version), row.rel)
        if pkl is None:
            return None
        col, label = self.spike_mode_cols()
        if version == self.version:
            spikes = row[col]
        else:
            res = read_pkl(pkl, self.columns[self.robot])
            spikes = (res[3] if col == "frames_gt20" else res[4]) if res else ""
        cmd = self.player_cmd(row, self.robot, f"[{version}] [{row.zone}] {row.filename}  {REASON_EN.get(row.reason, row.reason)}")
        cmd += ["--pkl", pkl]
        # Config snapshot and robot overrides the version was run with, so the player draws the same IK targets.
        if os.path.isdir(os.path.join(self.version_dir(version), "config")):
            cmd += ["--config_snapshot", self.version_dir(version)]
        overrides = [o for o in run_info(self.version_dir(version)).get("overrides", [])
                     if o.lstrip("+").startswith("robot.")]
        if overrides:
            cmd += ["--override", *overrides]
        return cmd + (["--spikes", spikes, "--spike_label", f"arm joint speed {label}"] if spikes else [])

    def warn_not_converted(self, row, version):
        if row.zone == "red":
            QMessageBox.warning(self, "红区不转换", f"{row.filename} 在红区，批量 retarget 不转换，只能单独播放参考动作。")
            return
        QMessageBox.warning(
            self, "还没转换",
            f"{row.filename} 在 {version} 里还没有被批量转换到。\n\n请等批量 retarget 转到这一段后再播放；"
            f"可以先勾选“只看已转换”浏览已经转好的片段。")

    def play_selected(self):
        row = self.selected_row()
        if row is None:
            return
        if row.zone == "red":
            cmd = self.player_cmd(row, self.robot, f"[{row.zone}] {row.filename}  {REASON_EN.get(row.reason, row.reason)}")
        else:
            cmd = self.version_cmd(row, self.version)
            if cmd is None:
                return self.warn_not_converted(row, self.version)
        self.stop_player()
        self.launch(cmd)
        self.statusBar().showMessage(f"正在打开 {row.filename}（{self.version}，{status_text(row.zone, row.converted)}）...", 5000)

    def reference_cmd(self, row):
        """Player command for the standard G1 retarget of the clip, or None (after a warning) without one."""
        root = self.dataset.official_csv_root
        if root is None:
            return self.player_cmd(row, "unitree_g1", f"[standard G1, live] {row.filename}") + ["--live"]
        csv = os.path.join(root, row.rel + ".csv")
        if not os.path.isfile(csv):
            QMessageBox.warning(self, f"没有{self.reference}", f"{self.reference}数据里没有这一段：\n{csv}")
            return None
        return self.player_cmd(row, "unitree_g1", f"[official G1] {row.filename}") + ["--official_csv", csv]

    def compare_reference(self):
        """The selected version's retarget and the standard G1 retarget, in two windows."""
        row = self.selected_row()
        if row is None:
            return
        ours = self.version_cmd(row, self.version)
        if ours is None:
            return self.warn_not_converted(row, self.version)
        reference = self.reference_cmd(row)
        if reference is None:
            return
        self.stop_player()
        self.launch(ours)
        self.launch(reference)
        wait = "（标准 G1 现场转换，要等几十秒）" if self.dataset.official_csv_root is None else ""
        self.statusBar().showMessage(f"正在打开 {row.filename}：{self.version} + {self.reference}{wait} ...", 8000)

    def compare_versions(self):
        """The same clip from the selected version and the comparison version, in two windows."""
        row = self.selected_row()
        if row is None:
            return
        other = self.compare_box.currentText()
        if not other or other == self.version:
            QMessageBox.information(self, "对比版本", "请在右上角“对比版本”里选一个和当前结果版本不同的版本。")
            return
        cmds = []
        for version in (self.version, other):
            cmd = self.version_cmd(row, version)
            if cmd is None:
                return self.warn_not_converted(row, version)
            cmds.append(cmd)
        self.stop_player()
        for cmd in cmds:
            self.launch(cmd)
        self.statusBar().showMessage(f"正在打开 {row.filename}：{self.version} + {other} ...", 8000)

    def launch(self, cmd):
        self.players.append(subprocess.Popen(cmd, cwd=ROOT))

    def stop_player(self):
        for player in self.players:
            if player.poll() is None:
                player.terminate()
                try:
                    player.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    player.kill()
        self.players = []

    def closeEvent(self, event):
        self.stop_player()
        super().closeEvent(event)


def main(dataset, argv=None):
    ap = argparse.ArgumentParser(description=f"{dataset.name} → g1d: zones, batch retarget progress, playback")
    ap.add_argument("--version", default=None, help="version to open (default: the latest)")
    ap.add_argument("--fps", type=int, default=None, help="target fps; read from the version's run.json by default")
    ap.add_argument("--robot", default=dataset.robot, help="robot of versions whose run.json does not name one")
    ap.add_argument("--cache_only", action="store_true",
                    help="read progress from the frame caches only (version on a network mount, see --build_cache --loop)")
    ap.add_argument("--build_cache", default=None, metavar="VERSION",
                    help="fill the frame / speed caches of a version with many processes and exit")
    ap.add_argument("--workers", type=int, default=18, help="processes of --build_cache")
    ap.add_argument("--loop", type=float, default=0, metavar="SECONDS",
                    help="with --build_cache, rescan every SECONDS instead of exiting")
    args = ap.parse_args(argv)
    if args.build_cache:
        build_cache(dataset.version_dir(args.build_cache), args.robot, args.workers, args.loop)
        return
    app = QApplication(sys.argv)
    app.setFont(QFont("Noto Sans CJK SC", 10))
    app.setStyleSheet(STYLE)
    window = Browser(dataset, args)
    window.show()
    sys.exit(app.exec())
