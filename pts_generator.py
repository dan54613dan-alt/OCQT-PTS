# -*- coding: utf-8 -*-
"""Core implementation of Phenology-Constrained Time-Series Splicing (PTS).

PTS constructs annual rice time-series samples for underrepresented rice
cropping-system classes by combining observed early-, middle-, and late-season
phenological components under adaptive phenological-window, regional-background,
continuity, physical-plausibility, and novelty constraints.

The script reads only a real training dataset. Validation and independent test
samples must not be supplied. Input/output paths and generation limits can be
configured through environment variables; see README.md for the expected NPZ
schema and usage notes.

This is research code corresponding to the method described in the manuscript.
The proprietary training data and fitted reference libraries are not included.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# 1. 路径与配置
# =============================================================================

PROJECT_ROOT = Path(
    os.environ.get(
        "RICE_PROJECT_ROOT",
        str(Path.cwd()),
    )
)
REAL_TRAIN_NPZ = Path(
    os.environ.get(
        "PTS6_REAL_TRAIN_NPZ",
        str(PROJECT_ROOT / "data" / "real_training.npz"),
    )
)
OUTPUT_ROOT = Path(
    os.environ.get(
        "PTS6_OUTPUT_ROOT",
        str(PROJECT_ROOT / "outputs" / "pts_v6"),
    )
)

OUTPUT_CONFLICT_POLICY = os.environ.get(
    "PTS6_OUTPUT_CONFLICT_POLICY", "auto_backup"
)  # auto_backup | overwrite | error

SEED = int(os.environ.get("PTS6_SEED", "20260712"))
TARGET_CORE_A_PER_CELL = int(os.environ.get("PTS6_CORE_A_PER_CELL", "500"))
MAX_CORE_B_PER_CELL = int(os.environ.get("PTS6_CORE_B_PER_CELL", "150"))
MAX_ATTEMPT_MULTIPLIER = int(os.environ.get("PTS6_MAX_ATTEMPT_MULTIPLIER", "50"))

# 调试：0 表示全部 region×class；正整数表示只跑前 N 个单元。
DEBUG_MAX_CELLS = int(os.environ.get("PTS6_DEBUG_MAX_CELLS", "0"))
# 可选精确筛选，例如："Xianning:*;*:6;*:7"。
DEBUG_CELL_FILTER = os.environ.get("PTS6_DEBUG_CELL_FILTER", "").strip()

# 真实参考样本上限，避免统计阶段过慢。
MAX_REFERENCE_PER_CLASS = 6000
MAX_REFERENCE_PER_REGION = 6000
MAX_COMPONENTS_PER_POOL = 30000

# 拼接参数。
COMPONENT_HALF_WIDTH_DAYS = 50.0
NEUTRAL_HALF_WIDTH_DAYS = 46.0
# v5.1: prevent the same broad window from representing both middle and late.
SEASON_BOUNDARY_GAP_DAYS = 4.0
MIN_TARGET_WINDOW_DAYS = 24.0
BACKGROUND_NDVI_QUANTILE = 0.45
BACKGROUND_MIN_STEPS = 12
EDGE_BLEND_DAYS = 12.0
STYLE_ADAPT_STRENGTH = 0.38
STYLE_SCALE_MIN = 0.65
STYLE_SCALE_MAX = 1.45
CONFLICT_BASE_WEIGHT = 0.15
SAME_REGION_DONOR_PROB = 0.75
SMOOTH_WINDOW = 3

# 使用次数与重复限制。
MAX_BASE_USAGE = 35
MAX_PARENT_USAGE = 25
MAX_TEMPLATE_USAGE = 35
MAX_PARENT_COMBINATION_REPEAT = 1
MAX_COARSE_DUPLICATE_REPEAT = 2

# 质量门槛。
FORMULA_MAE_HARD = 5e-5
FEATURE_OUTLIER_RATE_CORE_A = 0.020
FEATURE_OUTLIER_RATE_CORE_B = 0.055
CLASS_DISTANCE_CORE_A = 2.10
CLASS_DISTANCE_CORE_B = 3.10
REGION_DISTANCE_CORE_A = 2.10
REGION_DISTANCE_CORE_B = 3.10
DESCRIPTOR_OUTLIER_CORE_A = 0.22
DESCRIPTOR_OUTLIER_CORE_B = 0.38
CONTINUITY_RATIO_CORE_A = 1.50
CONTINUITY_RATIO_CORE_B = 2.60
NEAR_COPY_NRMSE = 0.018
NEAR_COPY_CHANGE_FRACTION = 0.045
HARD_NOOP_NRMSE = 0.006
HARD_NOOP_CHANGE_FRACTION = 0.015
MIN_CORE_A_SCORE = 0.66
MIN_CORE_B_SCORE = 0.48

# 描述符稳健尺度下限。
DESCRIPTOR_SCALE_FLOOR = 0.05
SPECTRAL_SCALE_FLOOR = 1e-4
EPS = 1e-6


# v6 hierarchical adaptive-window configuration.
MAX_WINDOW_REFERENCE_SAMPLES = int(os.environ.get("PTS6_MAX_WINDOW_REFERENCE_SAMPLES", "2500"))
WINDOW_SMOOTH_STEPS = int(os.environ.get("PTS6_WINDOW_SMOOTH_STEPS", "7"))
WINDOW_PRIOR_EXPAND_DAYS = float(os.environ.get("PTS6_WINDOW_PRIOR_EXPAND_DAYS", "18"))
WINDOW_START_LEVEL = float(os.environ.get("PTS6_WINDOW_START_LEVEL", "0.18"))
WINDOW_END_LEVEL = float(os.environ.get("PTS6_WINDOW_END_LEVEL", "0.18"))
WINDOW_MIN_USE_SCORE = float(os.environ.get("PTS6_WINDOW_MIN_USE_SCORE", "0.42"))
WINDOW_CORE_A_SCORE = float(os.environ.get("PTS6_WINDOW_CORE_A_SCORE", "0.56"))
MIN_WINDOW_DAYS_BY_SEASON = {"early": 30.0, "middle": 35.0, "late": 35.0}
MIN_WINDOW_RAW_AMPLITUDE = float(os.environ.get("PTS6_MIN_WINDOW_RAW_AMPLITUDE", "0.045"))
MIN_WINDOW_NORM_AMPLITUDE = float(os.environ.get("PTS6_MIN_WINDOW_NORM_AMPLITUDE", "0.30"))
MIN_WINDOW_LSWI_AMPLITUDE = float(os.environ.get("PTS6_MIN_WINDOW_LSWI_AMPLITUDE", "0.04"))
MIN_DISTINCT_PEAK_GAP_DAYS = float(os.environ.get("PTS6_MIN_DISTINCT_PEAK_GAP_DAYS", "24"))
MERGED_ML_MAX_PEAK_GAP_DAYS = float(os.environ.get("PTS6_MERGED_ML_MAX_PEAK_GAP_DAYS", "38"))
VALLEY_MIN_DEPTH_NORM = float(os.environ.get("PTS6_VALLEY_MIN_DEPTH_NORM", "0.06"))
MAX_WINDOW_OVERLAP_DAYS = float(os.environ.get("PTS6_MAX_WINDOW_OVERLAP_DAYS", "18"))
WINDOW_BOUNDARY_GAP_DAYS = float(os.environ.get("PTS6_WINDOW_BOUNDARY_GAP_DAYS", "4"))

BOUNDARY_RATIO_CORE_A = float(os.environ.get("PTS6_BOUNDARY_RATIO_CORE_A", "1.55"))
BOUNDARY_RATIO_CORE_B = float(os.environ.get("PTS6_BOUNDARY_RATIO_CORE_B", "2.80"))
MIN_WINDOW_CONFIDENCE_CORE_A = float(os.environ.get("PTS6_MIN_WINDOW_CONFIDENCE_CORE_A", "0.48"))
MIN_WINDOW_CONFIDENCE_CORE_B = float(os.environ.get("PTS6_MIN_WINDOW_CONFIDENCE_CORE_B", "0.32"))

# Keep v5 environment aliases usable when a Colab cell still contains old names.
TARGET_CORE_A_PER_CELL = int(os.environ.get("PTS6_CORE_A_PER_CELL", os.environ.get("PTS5_CORE_A_PER_CELL", str(TARGET_CORE_A_PER_CELL))))
MAX_CORE_B_PER_CELL = int(os.environ.get("PTS6_CORE_B_PER_CELL", os.environ.get("PTS5_CORE_B_PER_CELL", str(MAX_CORE_B_PER_CELL))))
MAX_ATTEMPT_MULTIPLIER = int(os.environ.get("PTS6_MAX_ATTEMPT_MULTIPLIER", os.environ.get("PTS5_MAX_ATTEMPT_MULTIPLIER", str(MAX_ATTEMPT_MULTIPLIER))))
DEBUG_MAX_CELLS = int(os.environ.get("PTS6_DEBUG_MAX_CELLS", os.environ.get("PTS5_DEBUG_MAX_CELLS", str(DEBUG_MAX_CELLS))))
DEBUG_CELL_FILTER = os.environ.get("PTS6_DEBUG_CELL_FILTER", os.environ.get("PTS5_DEBUG_CELL_FILTER", DEBUG_CELL_FILTER)).strip()

CLASS_NAMES = {
    0: "no_rice",
    1: "early_rice",
    2: "middle_rice",
    3: "late_rice",
    4: "early_late_rice",
    5: "early_middle_rice",
    6: "middle_late_rice",
    7: "early_middle_late_rice",
}

CLASS_Y = {
    0: np.asarray([0, 0, 0], dtype=np.float32),
    1: np.asarray([1, 0, 0], dtype=np.float32),
    2: np.asarray([0, 1, 0], dtype=np.float32),
    3: np.asarray([0, 0, 1], dtype=np.float32),
    4: np.asarray([1, 0, 1], dtype=np.float32),
    5: np.asarray([1, 1, 0], dtype=np.float32),
    6: np.asarray([0, 1, 1], dtype=np.float32),
    7: np.asarray([1, 1, 1], dtype=np.float32),
}

CLASS_SEASONS = {
    1: ("early",),
    2: ("middle",),
    3: ("late",),
    4: ("early", "late"),
    5: ("early", "middle"),
    6: ("middle", "late"),
    7: ("early", "middle", "late"),
}

SEASON_SINGLE_CLASS = {"early": 1, "middle": 2, "late": 3}
SEASON_ORDER = {"early": 0, "middle": 1, "late": 2}
SEASONS = ("early", "middle", "late")

# 初始搜索范围只用于建立真实季相组件库；之后由真实峰值统计校准。
INITIAL_PEAK_SEARCH = {
    "early": (45.0, 205.0),
    "middle": (105.0, 300.0),
    "late": (175.0, 366.0),
}

REGION_TOKENS = [
    ("hengyang_middle_rice_2", "Hengyang_2"),
    ("hengyang_late_rice_2", "Hengyang_2"),
    ("hengyang_2", "Hengyang_2"),
    ("hengyang2", "Hengyang_2"),
    ("dianbai", "Dianbai"),
    ("hongan", "Hongan"),
    ("huangmei", "Huangmei"),
    ("yangxin", "Yangxin"),
    ("xianning", "Xianning"),
    ("hengyang", "Hengyang"),
    ("dawu", "Dawu"),
    ("maonan", "Maonan"),
    ("liushi", "Liushi"),
]

rng = np.random.default_rng(SEED)
random.seed(SEED)


# =============================================================================
# 2. 数据结构
# =============================================================================

@dataclass
class FeatureIndex:
    b4: int
    b8: int
    b11: int
    ndvi: int
    lswi: int
    spectral: Tuple[int, ...]


@dataclass
class RobustReference:
    names: Tuple[str, ...]
    median: np.ndarray
    scale: np.ndarray
    p01: np.ndarray
    p99: np.ndarray


@dataclass
class CandidateQuality:
    tier: str
    score: float
    formula_mae: float
    feature_outlier_rate: float
    class_distance: float
    class_outlier_rate: float
    region_distance: float
    region_outlier_rate: float
    continuity_ratio: float
    boundary_ratio: float
    window_confidence: float
    nearest_parent_nrmse: float
    change_fraction: float
    phenology_pass: bool
    peak_structure_pass: bool
    novelty_pass: bool
    hard_fail_reason: str


@dataclass
class AdaptiveWindow:
    season: str
    start_doy: float
    peak_doy: float
    end_doy: float
    shape_score: float
    level: str
    key: str
    reference_n: int
    need_review: bool
    review_reason: str
    merged_middle_late: bool = False
    fallback_used: bool = False

    @property
    def duration_days(self) -> float:
        return float(self.end_doy - self.start_doy)

    def shifted(self, delta: float, level: str, key: str, score_factor: float = 0.90) -> "AdaptiveWindow":
        start = float(np.clip(self.start_doy + delta, 1.0, 365.0))
        peak = float(np.clip(self.peak_doy + delta, 1.0, 365.0))
        end = float(np.clip(self.end_doy + delta, 2.0, 366.0))
        if end <= start:
            half = max(self.duration_days, MIN_WINDOW_DAYS_BY_SEASON[self.season]) / 2.0
            start = max(1.0, peak - half)
            end = min(366.0, peak + half)
        return AdaptiveWindow(
            season=self.season,
            start_doy=start,
            peak_doy=peak,
            end_doy=end,
            shape_score=float(np.clip(self.shape_score * score_factor, 0.0, 1.0)),
            level=level,
            key=key,
            reference_n=self.reference_n,
            need_review=self.need_review,
            review_reason=self.review_reason + ";shifted_to_target_region",
            merged_middle_late=self.merged_middle_late,
            fallback_used=self.fallback_used,
        )


@dataclass
class WindowLibrary:
    source_class: Dict[Tuple[str, int, str], AdaptiveWindow]
    region_class: Dict[Tuple[str, int, str], AdaptiveWindow]
    region_season: Dict[Tuple[str, str], AdaptiveWindow]
    global_class: Dict[Tuple[int, str], AdaptiveWindow]
    global_season: Dict[str, AdaptiveWindow]
    global_peak_priors: Dict[str, float]
    rows: List[Dict[str, object]]

    @staticmethod
    def usable(window: Optional[AdaptiveWindow], min_score: float = WINDOW_MIN_USE_SCORE) -> bool:
        return window is not None and window.shape_score >= min_score and window.end_doy > window.start_doy

    def resolve_source(self, source_file: str, region: str, class_id: int, season: str) -> AdaptiveWindow:
        candidates = [
            self.source_class.get((normalize_source_file(source_file), int(class_id), season)),
            self.region_class.get((region, int(class_id), season)),
            self.region_season.get((region, season)),
            self.global_class.get((int(class_id), season)),
            self.global_season.get(season),
        ]
        for window in candidates:
            if self.usable(window):
                return window
        raise KeyError(f"No usable source window: source={source_file}, region={region}, class={class_id}, season={season}")

    def resolve_region_season(self, region: str, season: str) -> AdaptiveWindow:
        for window in [self.region_season.get((region, season)), self.global_season.get(season)]:
            if self.usable(window):
                return window
        raise KeyError(f"No usable region-season window: region={region}, season={season}")

    def resolve_target(self, region: str, class_id: int, season: str) -> AdaptiveWindow:
        exact = self.region_class.get((region, int(class_id), season))
        if self.usable(exact):
            return exact

        class_global = self.global_class.get((int(class_id), season))
        region_season = self.region_season.get((region, season))
        global_season = self.global_season.get(season)
        if self.usable(class_global) and self.usable(region_season) and self.usable(global_season):
            delta = float(region_season.peak_doy - global_season.peak_doy)
            return class_global.shifted(
                delta,
                level="global_class_shifted_to_region",
                key=f"{region}|class{class_id}|{season}",
                score_factor=0.88,
            )
        for window in [region_season, class_global, global_season]:
            if self.usable(window):
                return window
        raise KeyError(f"No usable target window: region={region}, class={class_id}, season={season}")


# =============================================================================
# 3. 通用工具
# =============================================================================

def require_file(path: Path) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(path)


def prepare_output_dir(path: Path) -> Dict[str, str]:
    path = Path(path)
    info = {"action": "create_new", "backup_path": ""}
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return info

    nonempty = [
        p for p in path.rglob("*")
        if p.is_file() and p.stat().st_size > 0
    ]
    if not nonempty:
        info["action"] = "reuse_empty"
        return info

    if OUTPUT_CONFLICT_POLICY == "overwrite":
        shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        info["action"] = "overwrite"
        return info

    if OUTPUT_CONFLICT_POLICY == "error":
        raise FileExistsError(f"输出目录已存在且非空：{path}")

    if OUTPUT_CONFLICT_POLICY != "auto_backup":
        raise ValueError(
            f"未知 OUTPUT_CONFLICT_POLICY={OUTPUT_CONFLICT_POLICY}"
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(path.name + f"_backup_{stamp}")
    suffix = 1
    while backup.exists():
        backup = path.with_name(path.name + f"_backup_{stamp}_{suffix}")
        suffix += 1
    shutil.move(str(path), str(backup))
    path.mkdir(parents=True, exist_ok=True)
    info["action"] = "auto_backup"
    info["backup_path"] = str(backup)
    return info


def decode_value(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def decode_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    return np.asarray([decode_value(v) for v in arr], dtype=object)


def normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", decode_value(value).lower())


def normalize_source_file(value: object) -> str:
    return os.path.basename(
        decode_value(value).replace("\\", "/")
    ).strip().lower()


def infer_region_from_source_file(value: object) -> str:
    text = normalize_source_file(value)
    for token, region in REGION_TOKENS:
        if token in text:
            return region
    stem = Path(text).stem
    first = re.split(r"[_\-.]+", stem)[0].strip()
    return first if first else "Unknown"


def ensure_doy_matrix(doy: np.ndarray, n: int) -> np.ndarray:
    doy = np.asarray(doy, dtype=np.float32)
    if doy.ndim == 1:
        doy = np.broadcast_to(doy[None, :], (n, len(doy))).copy()
    if doy.shape[0] != n:
        raise ValueError(f"DOY形状异常：{doy.shape}, N={n}")
    if np.nanmax(doy) <= 1.5:
        doy = doy * 366.0
    return doy.astype(np.float32)


def resolve_class_id(data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "class_id" in data.files:
        return np.asarray(data["class_id"], dtype=np.int64).reshape(-1)
    if "Y" in data.files:
        y = np.asarray(data["Y"])
        lookup = {tuple(v.astype(int).tolist()): k for k, v in CLASS_Y.items()}
        return np.asarray(
            [lookup[tuple((row >= 0.5).astype(int).tolist())] for row in y],
            dtype=np.int64,
        )
    raise KeyError(f"无法解析类别字段：{data.files}")


def find_feature_index(
    feature_names_norm: Sequence[str],
    candidates: Sequence[str],
) -> Optional[int]:
    candidate_norm = {normalize_name(v) for v in candidates}
    for i, name in enumerate(feature_names_norm):
        if name in candidate_norm:
            return i
    return None


def resolve_feature_index(feature_names: np.ndarray, f: int) -> FeatureIndex:
    names = [normalize_name(v) for v in feature_names]
    b4 = find_feature_index(names, ["B4", "Red", "S2B4", "B4_Red"])
    b8 = find_feature_index(names, ["B8", "NIR", "S2B8", "B8_NIR"])
    b11 = find_feature_index(names, ["B11", "SWIR1", "S2B11", "B11_SWIR1"])
    ndvi = find_feature_index(names, ["NDVI"])
    lswi = find_feature_index(names, ["LSWI"])

    if ndvi is None:
        ndvi = f - 2
    if lswi is None:
        lswi = f - 1
    if b4 is None or b8 is None or b11 is None:
        raise RuntimeError(
            "无法识别 B4/B8/B11；feature_names="
            + str(feature_names.tolist())
        )

    spectral = tuple(i for i in range(f) if i not in {ndvi, lswi})
    if len(spectral) != 8:
        raise ValueError(f"期望8个原始波段，实际 spectral indices={spectral}")
    return FeatureIndex(
        b4=int(b4), b8=int(b8), b11=int(b11),
        ndvi=int(ndvi), lswi=int(lswi), spectral=spectral,
    )


def safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    den_safe = np.where(
        np.abs(den) < EPS,
        np.where(den < 0, -EPS, EPS),
        den,
    )
    return num / den_safe


def recompute_indices(X: np.ndarray, fi: FeatureIndex) -> np.ndarray:
    out = np.asarray(X, dtype=np.float32).copy()
    out[..., fi.ndvi] = safe_divide(
        out[..., fi.b8] - out[..., fi.b4],
        out[..., fi.b8] + out[..., fi.b4],
    )
    out[..., fi.lswi] = safe_divide(
        out[..., fi.b8] - out[..., fi.b11],
        out[..., fi.b8] + out[..., fi.b11],
    )
    out[..., fi.ndvi] = np.clip(out[..., fi.ndvi], -1.0, 1.0)
    out[..., fi.lswi] = np.clip(out[..., fi.lswi], -1.0, 1.0)
    return out.astype(np.float32)


def short_hash_array(array: np.ndarray, decimals: int = 3) -> str:
    arr = np.round(np.asarray(array, dtype=np.float32), decimals=decimals)
    arr = np.ascontiguousarray(arr)
    return hashlib.sha1(arr.view(np.uint8)).hexdigest()[:20]


def moving_average_2d(X: np.ndarray, window: int = 3) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if window <= 1:
        return X.copy()
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(X, ((pad_left, pad_right), (0, 0)), mode="edge")
    out = np.empty_like(X)
    for t in range(X.shape[0]):
        out[t] = padded[t:t + window].mean(axis=0)
    return out


def local_peak_index(
    doy: np.ndarray,
    values: np.ndarray,
    start: float,
    end: float,
    preferred_center: Optional[float] = None,
) -> Optional[int]:
    doy = np.asarray(doy, dtype=np.float32)
    v = np.asarray(values, dtype=np.float32)
    idx = np.where((doy >= start) & (doy <= end))[0]
    if len(idx) == 0:
        return None

    candidates = []
    for i in idx:
        if i == 0 or i == len(v) - 1:
            continue
        if v[i] >= v[i - 1] and v[i] > v[i + 1]:
            candidates.append(int(i))
    if not candidates:
        candidates = idx.astype(int).tolist()

    values_norm = v[candidates]
    amp = float(np.nanmax(values_norm) - np.nanmin(values_norm))
    amp = max(amp, 0.05)
    scores = values_norm.astype(np.float64)
    if preferred_center is not None:
        distance_penalty = np.abs(doy[candidates] - preferred_center) / 80.0
        scores = scores - 0.20 * amp * distance_penalty
    return int(candidates[int(np.argmax(scores))])


def taper_weights(doy_values: np.ndarray, start: float, end: float) -> np.ndarray:
    d = np.asarray(doy_values, dtype=np.float32)
    left = np.clip((d - start) / max(EDGE_BLEND_DAYS, 1.0), 0.0, 1.0)
    right = np.clip((end - d) / max(EDGE_BLEND_DAYS, 1.0), 0.0, 1.0)
    w = np.minimum(left, right)
    # smoothstep
    return (w * w * (3.0 - 2.0 * w)).astype(np.float32)


def robust_fit_reference(
    values: np.ndarray,
    names: Sequence[str],
    scale_floor: float = DESCRIPTOR_SCALE_FLOOR,
) -> RobustReference:
    values = np.asarray(values, dtype=np.float64)
    median = np.nanmedian(values, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    scale = np.maximum(q75 - q25, scale_floor)
    p01 = np.nanpercentile(values, 1, axis=0)
    p99 = np.nanpercentile(values, 99, axis=0)
    return RobustReference(
        names=tuple(names),
        median=median.astype(np.float32),
        scale=scale.astype(np.float32),
        p01=p01.astype(np.float32),
        p99=p99.astype(np.float32),
    )


def robust_distance(
    vector: np.ndarray,
    reference: RobustReference,
) -> Tuple[float, float, np.ndarray]:
    v = np.asarray(vector, dtype=np.float64)
    z = np.abs(v - reference.median) / np.maximum(reference.scale, EPS)
    finite = np.isfinite(z)
    if not finite.any():
        return float("inf"), 1.0, z
    mean_z = float(np.nanmean(z[finite]))
    outlier_rate = float(np.mean(z[finite] > 3.0))
    return mean_z, outlier_rate, z


def nrmse_and_change(
    candidate: np.ndarray,
    parent: np.ndarray,
    feature_scale: np.ndarray,
    spectral_indices: Sequence[int],
) -> Tuple[float, float]:
    a = candidate[:, spectral_indices].astype(np.float64)
    b = parent[:, spectral_indices].astype(np.float64)
    scale = np.maximum(feature_scale[np.asarray(spectral_indices)], 1e-4)
    diff_z = (a - b) / scale[None, :]
    nrmse = float(np.sqrt(np.mean(diff_z ** 2)))
    change_fraction = float(np.mean(np.abs(diff_z) > 0.20))
    return nrmse, change_fraction


# =============================================================================
# 4. 描述符与真实参考
# =============================================================================


# =============================================================================
# 4. Hierarchical adaptive windows, descriptors, and real references
# =============================================================================

def smooth_1d(values: np.ndarray, window: int = WINDOW_SMOOTH_STEPS) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) < 5 or window <= 1:
        return values.copy()
    w = min(int(window), len(values))
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return values.copy()
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(values, window_length=w, polyorder=min(2, w - 1), mode="interp").astype(np.float32)
    except Exception:
        pad = w // 2
        padded = np.pad(values, (pad, pad), mode="edge")
        kernel = np.ones(w, dtype=np.float32) / w
        return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def robust_norm_1d(values: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=np.float32)
    if mask is None:
        subset = values[np.isfinite(values)]
    else:
        subset = values[np.asarray(mask, dtype=bool) & np.isfinite(values)]
    if subset.size == 0:
        return np.zeros_like(values), 0.0, 1.0
    p10, p90 = np.percentile(subset, [10, 90])
    scale = max(float(p90 - p10), EPS)
    return np.clip((values - p10) / scale, -0.5, 1.5).astype(np.float32), float(p10), float(p90)


def derivatives_curvature(values: np.ndarray, doy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    d = np.asarray(doy, dtype=np.float64).copy()
    for i in range(1, len(d)):
        if d[i] <= d[i - 1]:
            d[i] = d[i - 1] + 1.0
    slope = np.gradient(values, d)
    accel = np.gradient(slope, d)
    curvature = accel / np.power(1.0 + slope ** 2, 1.5)
    return (
        np.nan_to_num(slope).astype(np.float32),
        np.nan_to_num(accel).astype(np.float32),
        np.nan_to_num(curvature).astype(np.float32),
    )


def aggregate_group_curve(
    indices: np.ndarray,
    X: np.ndarray,
    DOY: np.ndarray,
    fi: FeatureIndex,
) -> Tuple[np.ndarray, np.ndarray, int]:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        raise ValueError("empty_group")
    if len(indices) > MAX_WINDOW_REFERENCE_SAMPLES:
        indices = rng.choice(indices, MAX_WINDOW_REFERENCE_SAMPLES, replace=False)
    template = np.median(DOY[indices], axis=0).astype(np.float32)
    template = np.maximum.accumulate(template)
    for i in range(1, len(template)):
        if template[i] <= template[i - 1]:
            template[i] = template[i - 1] + 1.0
    template = np.clip(template, 1.0, 366.0)
    curves = np.empty((len(indices), len(template), X.shape[2]), dtype=np.float32)
    for row_i, idx in enumerate(indices):
        for f in range(X.shape[2]):
            curves[row_i, :, f] = np.interp(
                template,
                DOY[idx],
                X[idx, :, f],
                left=float(X[idx, 0, f]),
                right=float(X[idx, -1, f]),
            )
    median_curve = np.median(curves, axis=0).astype(np.float32)
    median_curve = recompute_indices(median_curve, fi)
    return template, median_curve, len(indices)


def estimate_global_peak_priors(
    X: np.ndarray,
    DOY: np.ndarray,
    class_id: np.ndarray,
    fi: FeatureIndex,
) -> Dict[str, float]:
    priors: Dict[str, float] = {}
    for season in SEASONS:
        cid = SEASON_SINGLE_CLASS[season]
        indices = np.where(class_id == cid)[0]
        if len(indices) == 0:
            raise RuntimeError(f"Missing real single-season class for {season}")
        if len(indices) > 12000:
            indices = rng.choice(indices, 12000, replace=False)
        peaks = []
        start, end = INITIAL_PEAK_SEARCH[season]
        for idx in indices:
            p = local_peak_index(DOY[idx], X[idx, :, fi.ndvi], start, end, None)
            if p is not None:
                peaks.append(float(DOY[idx, p]))
        if not peaks:
            raise RuntimeError(f"Cannot estimate global prior for {season}")
        priors[season] = float(np.median(peaks))
    return priors


def detect_candidate_peaks(
    ndvi_norm: np.ndarray,
    doy: np.ndarray,
    search_start: float,
    search_end: float,
) -> List[Dict[str, float]]:
    idx = np.where((doy >= search_start) & (doy <= search_end))[0]
    if len(idx) < 3:
        return []
    values = ndvi_norm[idx]
    amplitude = max(float(np.max(values) - np.min(values)), 0.05)
    prominence = max(0.08, 0.15 * amplitude)
    rows: List[Dict[str, float]] = []
    try:
        from scipy.signal import find_peaks
        median_step = max(float(np.median(np.diff(doy[idx]))), 1.0)
        min_steps = max(2, int(round(18.0 / median_step)))
        peaks, props = find_peaks(values, prominence=prominence, distance=min_steps)
        for j, p_local in enumerate(peaks):
            p = int(idx[p_local])
            rows.append({
                "index": p,
                "doy": float(doy[p]),
                "height": float(ndvi_norm[p]),
                "prominence": float(props["prominences"][j]),
            })
    except Exception:
        for local_i in range(1, len(idx) - 1):
            p = int(idx[local_i])
            if ndvi_norm[p] >= ndvi_norm[idx[local_i - 1]] and ndvi_norm[p] > ndvi_norm[idx[local_i + 1]]:
                rows.append({
                    "index": p,
                    "doy": float(doy[p]),
                    "height": float(ndvi_norm[p]),
                    "prominence": float(ndvi_norm[p] - np.min(values)),
                })
    if not rows:
        p = int(idx[int(np.argmax(values))])
        rows = [{"index": p, "doy": float(doy[p]), "height": float(ndvi_norm[p]), "prominence": 0.0}]
    return rows


def assign_group_peaks(
    expected_seasons: Sequence[str],
    ndvi_norm: np.ndarray,
    doy: np.ndarray,
    global_priors: Dict[str, float],
) -> Tuple[Dict[str, int], bool, str]:
    candidate_by_season: Dict[str, List[Dict[str, float]]] = {}
    for season in expected_seasons:
        start, end = INITIAL_PEAK_SEARCH[season]
        start = max(1.0, start - WINDOW_PRIOR_EXPAND_DAYS)
        end = min(366.0, end + WINDOW_PRIOR_EXPAND_DAYS)
        rows = detect_candidate_peaks(ndvi_norm, doy, start, end)
        rows = sorted(
            rows,
            key=lambda r: (
                r["height"] + 0.45 * r["prominence"]
                - 0.004 * abs(r["doy"] - global_priors[season])
            ),
            reverse=True,
        )[:6]
        candidate_by_season[season] = rows

    best = None
    best_score = -np.inf
    combinations = itertools.product(*[candidate_by_season[s] for s in expected_seasons])
    for combo in combinations:
        doys = [float(r["doy"]) for r in combo]
        if any(doys[i] >= doys[i + 1] for i in range(len(doys) - 1)):
            continue
        if any((doys[i + 1] - doys[i]) < MIN_DISTINCT_PEAK_GAP_DAYS for i in range(len(doys) - 1)):
            continue
        score = 0.0
        for season, row in zip(expected_seasons, combo):
            score += row["height"] + 0.45 * row["prominence"]
            score -= 0.004 * abs(row["doy"] - global_priors[season])
        if score > best_score:
            best_score = score
            best = combo
    if best is not None:
        return {s: int(r["index"]) for s, r in zip(expected_seasons, best)}, False, "distinct_peaks"

    # Real middle-late systems can contain a merged broad peak. Allow only the
    # middle-late pair to share a peak; early must remain distinct.
    merged = False
    assignments: Dict[str, int] = {}
    for season in expected_seasons:
        assignments[season] = int(candidate_by_season[season][0]["index"])
    if "middle" in expected_seasons and "late" in expected_seasons:
        pm = assignments["middle"]
        pl = assignments["late"]
        if abs(float(doy[pl]) - float(doy[pm])) <= MERGED_ML_MAX_PEAK_GAP_DAYS:
            shared = pm if ndvi_norm[pm] >= ndvi_norm[pl] else pl
            assignments["middle"] = shared
            assignments["late"] = shared
            merged = True
    return assignments, merged, "merged_middle_late_fallback" if merged else "independent_peak_fallback"


def threshold_window_indices(
    ndvi_norm: np.ndarray,
    doy: np.ndarray,
    peak_idx: int,
    search_start: float,
    search_end: float,
    season: str,
) -> Tuple[int, int]:
    idx = np.where((doy >= search_start) & (doy <= search_end))[0]
    left = idx[idx <= peak_idx]
    right = idx[idx >= peak_idx]
    low_left = left[ndvi_norm[left] <= WINDOW_START_LEVEL]
    low_right = right[(ndvi_norm[right] <= WINDOW_END_LEVEL) & (right > peak_idx)]
    start_idx = int(low_left[-1] + 1) if len(low_left) and low_left[-1] < peak_idx else int(left.min())
    end_idx = int(low_right[0] - 1) if len(low_right) else int(right.max())
    min_days = MIN_WINDOW_DAYS_BY_SEASON[season]
    min_idx, max_idx = int(idx.min()), int(idx.max())
    while float(doy[end_idx] - doy[start_idx]) < min_days:
        can_left = start_idx > min_idx
        can_right = end_idx < max_idx
        if can_left and can_right:
            if abs(doy[start_idx] - doy[peak_idx]) <= abs(doy[end_idx] - doy[peak_idx]):
                end_idx += 1
            else:
                start_idx -= 1
        elif can_left:
            start_idx -= 1
        elif can_right:
            end_idx += 1
        else:
            break
    return start_idx, end_idx


def find_valley_boundary(
    doy: np.ndarray,
    ndvi_norm: np.ndarray,
    left_peak_idx: int,
    right_peak_idx: int,
) -> Tuple[Optional[float], float, str]:
    if right_peak_idx <= left_peak_idx + 1:
        return None, 0.0, "no_interval"
    between = np.arange(left_peak_idx + 1, right_peak_idx)
    if len(between) == 0:
        return None, 0.0, "no_interval"
    valley_idx = int(between[int(np.argmin(ndvi_norm[between]))])
    peak_floor = min(float(ndvi_norm[left_peak_idx]), float(ndvi_norm[right_peak_idx]))
    depth = float(peak_floor - ndvi_norm[valley_idx])
    slope, accel, _ = derivatives_curvature(ndvi_norm, doy)
    lo = max(left_peak_idx, valley_idx - 2)
    hi = min(right_peak_idx, valley_idx + 2)
    sign_ok = bool(np.min(slope[lo:valley_idx + 1]) < 0 and np.max(slope[valley_idx:hi + 1]) > 0)
    curvature_ok = bool(accel[valley_idx] > -0.002)
    if depth >= VALLEY_MIN_DEPTH_NORM and (sign_ok or curvature_ok):
        return float(doy[valley_idx]), depth, "valley_or_inflection"
    return None, depth, "weak_valley"


def window_shape_metrics(
    season: str,
    doy: np.ndarray,
    ndvi_raw: np.ndarray,
    lswi_raw: np.ndarray,
    ndvi_norm: np.ndarray,
    start_idx: int,
    peak_idx: int,
    end_idx: int,
    prior_peak: float,
) -> Tuple[float, str]:
    idx = np.arange(start_idx, end_idx + 1)
    left = np.arange(start_idx, peak_idx + 1)
    right = np.arange(peak_idx, end_idx + 1)
    slope, accel, curvature = derivatives_curvature(ndvi_norm, doy)
    raw_amp = float(np.percentile(ndvi_raw[idx], 90) - np.percentile(ndvi_raw[idx], 10))
    norm_amp = float(np.max(ndvi_norm[idx]) - np.min(ndvi_norm[idx]))
    lswi_amp = float(np.percentile(lswi_raw[idx], 90) - np.percentile(lswi_raw[idx], 10))
    duration = float(doy[end_idx] - doy[start_idx])
    rise = float(np.max(slope[left])) if len(left) else 0.0
    fall = float(np.min(slope[right])) if len(right) else 0.0
    a_sign = np.sign(accel[idx])
    a_sign = a_sign[np.abs(accel[idx]) > 1e-4]
    inflections = int(np.sum(a_sign[1:] != a_sign[:-1])) if len(a_sign) > 1 else 0
    curv_energy = float(np.mean(np.abs(curvature[idx])))
    peak_distance = abs(float(doy[peak_idx]) - float(prior_peak))
    score = 0.0
    reasons = []
    if raw_amp >= MIN_WINDOW_RAW_AMPLITUDE: score += 0.17
    else: reasons.append(f"raw_amp={raw_amp:.3f}")
    if norm_amp >= MIN_WINDOW_NORM_AMPLITUDE: score += 0.14
    else: reasons.append(f"norm_amp={norm_amp:.3f}")
    if lswi_amp >= MIN_WINDOW_LSWI_AMPLITUDE: score += 0.06
    else: reasons.append(f"lswi_amp={lswi_amp:.3f}")
    if duration >= MIN_WINDOW_DAYS_BY_SEASON[season]: score += 0.12
    else: reasons.append(f"duration={duration:.1f}")
    if rise > 0: score += 0.13
    else: reasons.append("no_rise")
    if fall < 0: score += 0.13
    else: reasons.append("no_fall")
    if inflections >= 1: score += 0.10
    else: reasons.append("no_inflection")
    if start_idx < peak_idx < end_idx: score += 0.05
    else: reasons.append("edge_peak")
    if peak_distance <= 30: score += 0.07
    elif peak_distance <= 55: score += 0.035
    else: reasons.append(f"peak_shift={peak_distance:.0f}d")
    if np.isfinite(curv_energy) and curv_energy < 0.05: score += 0.03
    else: reasons.append("curvature_high")
    return float(np.clip(score, 0.0, 1.0)), ";".join(reasons) if reasons else "ok"


def adaptive_windows_for_group(
    indices: np.ndarray,
    expected_seasons: Sequence[str],
    X: np.ndarray,
    DOY: np.ndarray,
    fi: FeatureIndex,
    global_priors: Dict[str, float],
    level: str,
    key: str,
) -> Tuple[Dict[str, AdaptiveWindow], Dict[str, object]]:
    template_doy, median_curve, reference_n = aggregate_group_curve(indices, X, DOY, fi)
    ndvi = smooth_1d(median_curve[:, fi.ndvi])
    lswi = smooth_1d(median_curve[:, fi.lswi])
    ndvi_norm, _, _ = robust_norm_1d(ndvi)
    assignments, merged_ml, assignment_mode = assign_group_peaks(expected_seasons, ndvi_norm, template_doy, global_priors)

    raw_windows: Dict[str, Tuple[int, int, int]] = {}
    for season in expected_seasons:
        peak_idx = assignments[season]
        prior_start, prior_end = INITIAL_PEAK_SEARCH[season]
        start_idx, end_idx = threshold_window_indices(
            ndvi_norm,
            template_doy,
            peak_idx,
            max(1.0, prior_start - WINDOW_PRIOR_EXPAND_DAYS),
            min(366.0, prior_end + WINDOW_PRIOR_EXPAND_DAYS),
            season,
        )
        raw_windows[season] = (start_idx, peak_idx, end_idx)

    # Resolve adjacent distinct seasons by a real valley/inflection on the group
    # median curve. Midpoint is only a final fallback.
    boundary_meta = []
    expected = list(expected_seasons)
    for left, right in zip(expected[:-1], expected[1:]):
        ls, lp, le = raw_windows[left]
        rs, rp, re = raw_windows[right]
        if lp == rp and left == "middle" and right == "late" and merged_ml:
            peak_doy = float(template_doy[lp])
            raw_windows[left] = (ls, lp, min(le, int(np.searchsorted(template_doy, peak_doy + 18.0))))
            raw_windows[right] = (max(rs, int(np.searchsorted(template_doy, peak_doy - 18.0))), rp, re)
            boundary_meta.append({"pair": f"{left}-{right}", "mode": "merged_overlap", "depth": 0.0})
            continue
        if rp <= lp:
            continue
        boundary, depth, mode = find_valley_boundary(template_doy, ndvi_norm, lp, rp)
        if boundary is None:
            boundary = 0.5 * (float(template_doy[lp]) + float(template_doy[rp]))
            mode = "midpoint_fallback"
        left_end_doy = boundary - WINDOW_BOUNDARY_GAP_DAYS / 2.0
        right_start_doy = boundary + WINDOW_BOUNDARY_GAP_DAYS / 2.0
        le_new = min(le, int(np.searchsorted(template_doy, left_end_doy, side="right") - 1))
        rs_new = max(rs, int(np.searchsorted(template_doy, right_start_doy, side="left")))
        le_new = max(le_new, lp + 1)
        rs_new = min(rs_new, rp - 1)
        raw_windows[left] = (ls, lp, le_new)
        raw_windows[right] = (rs_new, rp, re)
        boundary_meta.append({"pair": f"{left}-{right}", "mode": mode, "depth": depth, "boundary_doy": boundary})

    windows: Dict[str, AdaptiveWindow] = {}
    for season in expected_seasons:
        start_idx, peak_idx, end_idx = raw_windows[season]
        score, reason = window_shape_metrics(
            season, template_doy, ndvi, lswi, ndvi_norm,
            start_idx, peak_idx, end_idx, global_priors[season],
        )
        fallback = False
        if score < 0.30 or end_idx <= start_idx:
            peak = float(template_doy[peak_idx])
            half = max(MIN_WINDOW_DAYS_BY_SEASON[season] / 2.0, 22.0)
            start_doy = max(1.0, peak - half)
            end_doy = min(366.0, peak + half)
            fallback = True
            reason += ";low_score_fallback_around_peak"
            score *= 0.85
        else:
            start_doy = float(template_doy[start_idx])
            end_doy = float(template_doy[end_idx])
        windows[season] = AdaptiveWindow(
            season=season,
            start_doy=start_doy,
            peak_doy=float(template_doy[peak_idx]),
            end_doy=end_doy,
            shape_score=score,
            level=level,
            key=key,
            reference_n=reference_n,
            need_review=bool(score < WINDOW_CORE_A_SCORE or fallback or assignment_mode != "distinct_peaks"),
            review_reason=f"{assignment_mode};{reason}",
            merged_middle_late=bool(merged_ml and season in {"middle", "late"}),
            fallback_used=fallback,
        )
    return windows, {
        "reference_n": reference_n,
        "assignment_mode": assignment_mode,
        "merged_middle_late": merged_ml,
        "boundaries": boundary_meta,
    }


def build_hierarchical_window_library(
    X: np.ndarray,
    DOY: np.ndarray,
    class_id: np.ndarray,
    regions: np.ndarray,
    source_file: np.ndarray,
    fi: FeatureIndex,
) -> WindowLibrary:
    priors = estimate_global_peak_priors(X, DOY, class_id, fi)
    source_map: Dict[Tuple[str, int, str], AdaptiveWindow] = {}
    region_class_map: Dict[Tuple[str, int, str], AdaptiveWindow] = {}
    region_season_map: Dict[Tuple[str, str], AdaptiveWindow] = {}
    global_class_map: Dict[Tuple[int, str], AdaptiveWindow] = {}
    global_season_map: Dict[str, AdaptiveWindow] = {}
    rows: List[Dict[str, object]] = []

    def add_group(indices, seasons, level, key, target_dict, key_builder):
        if len(indices) < 20:
            return
        try:
            windows, meta = adaptive_windows_for_group(
                np.asarray(indices, dtype=np.int64), seasons, X, DOY, fi, priors, level, key
            )
        except Exception as exc:
            rows.append({"level": level, "key": key, "status": f"failed:{type(exc).__name__}", "detail": str(exc)})
            return
        for season, window in windows.items():
            target_dict[key_builder(season)] = window
            rows.append({
                "level": level,
                "key": key,
                "season": season,
                "start_doy": window.start_doy,
                "peak_doy": window.peak_doy,
                "end_doy": window.end_doy,
                "duration_days": window.duration_days,
                "shape_score": window.shape_score,
                "need_review": window.need_review,
                "review_reason": window.review_reason,
                "merged_middle_late": window.merged_middle_late,
                "fallback_used": window.fallback_used,
                "reference_n": window.reference_n,
                "assignment_mode": meta.get("assignment_mode", ""),
                "boundary_meta_json": json.dumps(meta.get("boundaries", []), ensure_ascii=False),
                "status": "ok",
            })

    # Exact source-file × class groups.
    normalized_sources = np.asarray([normalize_source_file(v) for v in source_file], dtype=object)
    for src in sorted(set(normalized_sources.tolist())):
        idx_src = np.where(normalized_sources == src)[0]
        classes = sorted(set(class_id[idx_src].tolist()))
        for cid in classes:
            if cid <= 0:
                continue
            idx = idx_src[class_id[idx_src] == cid]
            seasons = CLASS_SEASONS[int(cid)]
            add_group(
                idx, seasons, "source_class", f"{src}|class{cid}", source_map,
                lambda season, src=src, cid=cid: (src, int(cid), season),
            )

    for region in sorted(set(regions.tolist())):
        for cid in range(1, 8):
            idx = np.where((regions == region) & (class_id == cid))[0]
            add_group(
                idx, CLASS_SEASONS[cid], "region_class", f"{region}|class{cid}", region_class_map,
                lambda season, region=region, cid=cid: (region, int(cid), season),
            )
        for season in SEASONS:
            containing = [cid for cid, seasons in CLASS_SEASONS.items() if season in seasons]
            idx = np.where((regions == region) & np.isin(class_id, containing))[0]
            add_group(
                idx, (season,), "region_season", f"{region}|{season}", region_season_map,
                lambda s, region=region: (region, s),
            )

    for cid in range(1, 8):
        idx = np.where(class_id == cid)[0]
        add_group(
            idx, CLASS_SEASONS[cid], "global_class", f"ALL|class{cid}", global_class_map,
            lambda season, cid=cid: (int(cid), season),
        )
    for season in SEASONS:
        containing = [cid for cid, seasons in CLASS_SEASONS.items() if season in seasons]
        idx = np.where(np.isin(class_id, containing))[0]
        add_group(
            idx, (season,), "global_season", f"ALL|{season}", global_season_map,
            lambda s: s,
        )

    library = WindowLibrary(
        source_class=source_map,
        region_class=region_class_map,
        region_season=region_season_map,
        global_class=global_class_map,
        global_season=global_season_map,
        global_peak_priors=priors,
        rows=rows,
    )
    for season in SEASONS:
        if not library.usable(global_season_map.get(season), min_score=0.25):
            raise RuntimeError(f"Global adaptive window unavailable for season={season}")
    return library


def window_to_tuple(window: AdaptiveWindow) -> Tuple[float, float]:
    return float(window.start_doy), float(window.end_doy)


def resolve_target_windows(
    library: WindowLibrary,
    target_region: str,
    target_class: int,
) -> Dict[str, AdaptiveWindow]:
    windows = {
        season: library.resolve_target(target_region, target_class, season)
        for season in CLASS_SEASONS[target_class]
    }
    expected = list(CLASS_SEASONS[target_class])
    # Windows produced jointly at region_class/global_class level already use
    # valleys. Only trim pathological residual overlap after hierarchical shift.
    for left, right in zip(expected[:-1], expected[1:]):
        wl, wr = windows[left], windows[right]
        overlap = wl.end_doy - wr.start_doy
        merged = wl.merged_middle_late and wr.merged_middle_late and left == "middle" and right == "late"
        if overlap > MAX_WINDOW_OVERLAP_DAYS and not merged:
            boundary = 0.5 * (wl.peak_doy + wr.peak_doy)
            windows[left] = AdaptiveWindow(**{
                **asdict(wl),
                "end_doy": max(wl.peak_doy + 2.0, boundary - WINDOW_BOUNDARY_GAP_DAYS / 2.0),
                "shape_score": wl.shape_score * 0.95,
                "need_review": True,
                "review_reason": wl.review_reason + ";overlap_trimmed",
            })
            windows[right] = AdaptiveWindow(**{
                **asdict(wr),
                "start_doy": min(wr.peak_doy - 2.0, boundary + WINDOW_BOUNDARY_GAP_DAYS / 2.0),
                "shape_score": wr.shape_score * 0.95,
                "need_review": True,
                "review_reason": wr.review_reason + ";overlap_trimmed",
            })
    return windows


def descriptor_names(fi: FeatureIndex) -> Tuple[str, ...]:
    names: List[str] = []
    for season in SEASONS:
        names.extend([
            f"{season}_peak_offset",
            f"{season}_ndvi_amp",
            f"{season}_ndvi_mean",
            f"{season}_lswi_amp",
            f"{season}_lswi_mean",
        ])
    names.extend([
        "ndvi_range", "lswi_range",
        "ndvi_total_variation", "lswi_total_variation",
        "ndvi_max_slope_30d", "lswi_max_slope_30d",
        "ndvi_curvature_q95_30d2", "lswi_curvature_q95_30d2",
    ])
    return tuple(names)


def describe_phenology(
    X: np.ndarray,
    doy: np.ndarray,
    fi: FeatureIndex,
    windows: Dict[str, AdaptiveWindow],
) -> np.ndarray:
    doy = np.asarray(doy, dtype=np.float32)
    ndvi = np.asarray(X[:, fi.ndvi], dtype=np.float64)
    lswi = np.asarray(X[:, fi.lswi], dtype=np.float64)
    values: List[float] = []
    for season in SEASONS:
        window = windows[season]
        idx = np.where((doy >= window.start_doy) & (doy <= window.end_doy))[0]
        if len(idx) < 2:
            values.extend([np.nan] * 5)
            continue
        nd = ndvi[idx]
        lw = lswi[idx]
        peak_local = int(np.argmax(nd))
        peak_doy = float(doy[idx[peak_local]])
        values.extend([
            peak_doy - float(window.peak_doy),
            float(np.max(nd) - np.min(nd)),
            float(np.mean(nd)),
            float(np.max(lw) - np.min(lw)),
            float(np.mean(lw)),
        ])
    dt = np.maximum(np.diff(doy).astype(np.float64), 1.0)
    ndvi_diff = np.diff(ndvi)
    lswi_diff = np.diff(lswi)
    ndvi_slope = ndvi_diff / dt * 30.0
    lswi_slope = lswi_diff / dt * 30.0
    ndvi_curv = np.diff(ndvi_slope) / np.maximum((dt[:-1] + dt[1:]) * 0.5, 1.0) * 30.0 if len(ndvi_slope) > 1 else np.asarray([0.0])
    lswi_curv = np.diff(lswi_slope) / np.maximum((dt[:-1] + dt[1:]) * 0.5, 1.0) * 30.0 if len(lswi_slope) > 1 else np.asarray([0.0])
    values.extend([
        float(np.max(ndvi) - np.min(ndvi)),
        float(np.max(lswi) - np.min(lswi)),
        float(np.sum(np.abs(ndvi_diff))),
        float(np.sum(np.abs(lswi_diff))),
        float(np.max(np.abs(ndvi_slope))) if len(ndvi_slope) else 0.0,
        float(np.max(np.abs(lswi_slope))) if len(lswi_slope) else 0.0,
        float(np.quantile(np.abs(ndvi_curv), 0.95)),
        float(np.quantile(np.abs(lswi_curv), 0.95)),
    ])
    return np.asarray(values, dtype=np.float32)


def style_descriptor_names(spectral_indices: Sequence[int]) -> Tuple[str, ...]:
    names: List[str] = []
    for idx in spectral_indices:
        names.extend([f"f{idx}_background_median", f"f{idx}_background_iqr"])
    return tuple(names)


def background_mask(ndvi: np.ndarray, excluded_windows: Optional[Sequence[AdaptiveWindow]] = None, doy: Optional[np.ndarray] = None) -> np.ndarray:
    ndvi = np.asarray(ndvi, dtype=np.float64)
    threshold = float(np.quantile(ndvi, BACKGROUND_NDVI_QUANTILE))
    mask = ndvi <= threshold
    if excluded_windows is not None and doy is not None:
        for window in excluded_windows:
            mask &= ~((doy >= window.start_doy) & (doy <= window.end_doy))
    if int(mask.sum()) < BACKGROUND_MIN_STEPS:
        order = np.argsort(ndvi)
        mask = np.zeros(len(ndvi), dtype=bool)
        mask[order[:min(BACKGROUND_MIN_STEPS, len(ndvi))]] = True
    return mask


def describe_region_style(X: np.ndarray, fi: FeatureIndex) -> np.ndarray:
    mask = background_mask(X[:, fi.ndvi])
    spectral = X[mask][:, fi.spectral].astype(np.float64)
    med = np.median(spectral, axis=0)
    iqr = np.percentile(spectral, 75, axis=0) - np.percentile(spectral, 25, axis=0)
    out = np.empty(len(fi.spectral) * 2, dtype=np.float32)
    out[0::2] = med.astype(np.float32)
    out[1::2] = iqr.astype(np.float32)
    return out


def continuity_vector(X: np.ndarray, doy: np.ndarray, fi: FeatureIndex) -> np.ndarray:
    doy = np.asarray(doy, dtype=np.float64)
    ndvi = X[:, fi.ndvi].astype(np.float64)
    lswi = X[:, fi.lswi].astype(np.float64)
    dt = np.maximum(np.diff(doy), 1.0)
    ndvi_diff = np.diff(ndvi); lswi_diff = np.diff(lswi)
    ndvi_slope = ndvi_diff / dt * 30.0; lswi_slope = lswi_diff / dt * 30.0
    ndvi_curv = np.diff(ndvi_slope) / np.maximum((dt[:-1] + dt[1:]) * 0.5, 1.0) * 30.0 if len(ndvi_slope) > 1 else np.asarray([0.0])
    lswi_curv = np.diff(lswi_slope) / np.maximum((dt[:-1] + dt[1:]) * 0.5, 1.0) * 30.0 if len(lswi_slope) > 1 else np.asarray([0.0])
    return np.asarray([
        np.max(np.abs(ndvi_diff)) if len(ndvi_diff) else 0.0,
        np.max(np.abs(lswi_diff)) if len(lswi_diff) else 0.0,
        np.max(np.abs(ndvi_slope)) if len(ndvi_slope) else 0.0,
        np.max(np.abs(lswi_slope)) if len(lswi_slope) else 0.0,
        np.quantile(np.abs(ndvi_curv), 0.95),
        np.quantile(np.abs(lswi_curv), 0.95),
    ], dtype=np.float32)


def boundary_vector(
    X: np.ndarray,
    doy: np.ndarray,
    fi: FeatureIndex,
    windows: Sequence[AdaptiveWindow],
) -> np.ndarray:
    ndvi = X[:, fi.ndvi].astype(np.float64)
    lswi = X[:, fi.lswi].astype(np.float64)
    d = np.asarray(doy, dtype=np.float64)
    slope_n, accel_n, _ = derivatives_curvature(ndvi, d)
    slope_l, accel_l, _ = derivatives_curvature(lswi, d)
    vals = [[], [], [], [], [], []]
    for window in windows:
        for boundary in [window.start_doy, window.end_doy]:
            i = int(np.argmin(np.abs(d - boundary)))
            if i <= 0 or i >= len(d) - 1:
                continue
            vals[0].append(abs(ndvi[i] - ndvi[i - 1]))
            vals[1].append(abs(lswi[i] - lswi[i - 1]))
            vals[2].append(abs(slope_n[i] - slope_n[i - 1]))
            vals[3].append(abs(slope_l[i] - slope_l[i - 1]))
            vals[4].append(abs(accel_n[i] - accel_n[i - 1]))
            vals[5].append(abs(accel_l[i] - accel_l[i - 1]))
    return np.asarray([max(v) if v else 0.0 for v in vals], dtype=np.float32)


def real_windows_for_sample(
    library: WindowLibrary,
    source_file: str,
    region: str,
    class_id: int,
) -> Dict[str, AdaptiveWindow]:
    windows: Dict[str, AdaptiveWindow] = {}
    for season in SEASONS:
        if class_id > 0 and season in CLASS_SEASONS[class_id]:
            windows[season] = library.resolve_source(source_file, region, class_id, season)
        else:
            windows[season] = library.resolve_region_season(region, season)
    return windows


def build_references(
    X: np.ndarray,
    DOY: np.ndarray,
    class_id: np.ndarray,
    regions: np.ndarray,
    source_file: np.ndarray,
    fi: FeatureIndex,
    library: WindowLibrary,
) -> Tuple[Dict[int, RobustReference], Dict[str, RobustReference], Dict[str, np.ndarray], Dict[str, np.ndarray], pd.DataFrame]:
    class_refs: Dict[int, RobustReference] = {}
    region_refs: Dict[str, RobustReference] = {}
    continuity_p95: Dict[str, np.ndarray] = {}
    boundary_p95: Dict[str, np.ndarray] = {}
    rows = []
    p_names = descriptor_names(fi)
    s_names = style_descriptor_names(fi.spectral)
    for cid in range(1, 8):
        indices = np.where(class_id == cid)[0]
        if len(indices) > MAX_REFERENCE_PER_CLASS:
            indices = rng.choice(indices, MAX_REFERENCE_PER_CLASS, replace=False)
        descriptors = []
        for i in indices:
            try:
                windows = real_windows_for_sample(library, source_file[i], str(regions[i]), cid)
                descriptors.append(describe_phenology(X[i], DOY[i], fi, windows))
            except Exception:
                continue
        if len(descriptors) < 20:
            raise RuntimeError(f"Insufficient adaptive real descriptors for class {cid}: N={len(descriptors)}")
        class_refs[cid] = robust_fit_reference(np.stack(descriptors), p_names, 0.04)
        rows.append({"reference_type": "class_phenology", "key": CLASS_NAMES[cid], "N": len(descriptors)})
    for region in sorted(set(regions.tolist())):
        indices = np.where((regions == region) & (class_id > 0))[0]
        if len(indices) > MAX_REFERENCE_PER_REGION:
            indices = rng.choice(indices, MAX_REFERENCE_PER_REGION, replace=False)
        styles, conts, bounds = [], [], []
        for i in indices:
            styles.append(describe_region_style(X[i], fi))
            conts.append(continuity_vector(X[i], DOY[i], fi))
            try:
                expected_windows = [
                    library.resolve_source(source_file[i], region, int(class_id[i]), s)
                    for s in CLASS_SEASONS[int(class_id[i])]
                ]
                bounds.append(boundary_vector(X[i], DOY[i], fi, expected_windows))
            except Exception:
                pass
        region_refs[region] = robust_fit_reference(np.stack(styles), s_names, 1e-4)
        continuity_p95[region] = np.maximum(
            np.percentile(np.stack(conts), 95, axis=0).astype(np.float32),
            np.asarray([0.03, 0.03, 0.03, 0.03, 0.005, 0.005], dtype=np.float32),
        )
        if bounds:
            boundary_p95[region] = np.maximum(
                np.percentile(np.stack(bounds), 95, axis=0).astype(np.float32),
                np.asarray([0.02, 0.02, 0.002, 0.002, 0.0005, 0.0005], dtype=np.float32),
            )
        else:
            boundary_p95[region] = continuity_p95[region].copy()
        rows.append({"reference_type": "region_style_continuity_boundary", "key": region, "N": len(indices)})
    return class_refs, region_refs, continuity_p95, boundary_p95, pd.DataFrame(rows)


# =============================================================================
# 5. Adaptive component pools and morphologically constrained splicing
# =============================================================================

def build_component_pools(
    X: np.ndarray,
    DOY: np.ndarray,
    class_id: np.ndarray,
    regions: np.ndarray,
    source_file: np.ndarray,
    fi: FeatureIndex,
    library: WindowLibrary,
) -> Tuple[Dict[Tuple[str, str], np.ndarray], Dict[str, np.ndarray], Dict[Tuple[int, str], float], pd.DataFrame]:
    same_region: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    global_pool: Dict[str, List[int]] = defaultdict(list)
    quality: Dict[Tuple[int, str], float] = {}
    for idx in range(len(X)):
        cid = int(class_id[idx])
        if cid <= 0:
            continue
        region = str(regions[idx])
        for season in CLASS_SEASONS[cid]:
            try:
                window = library.resolve_source(source_file[idx], region, cid, season)
            except Exception:
                continue
            widx = np.where((DOY[idx] >= window.start_doy) & (DOY[idx] <= window.end_doy))[0]
            if len(widx) < 7:
                continue
            amp = float(np.max(X[idx, widx, fi.ndvi]) - np.min(X[idx, widx, fi.ndvi]))
            if amp < max(0.10, MIN_WINDOW_RAW_AMPLITUDE):
                continue
            sample_score = float(np.clip(0.65 * window.shape_score + 0.35 * min(amp / 0.35, 1.0), 0.0, 1.0))
            quality[(idx, season)] = sample_score
            same_region[(region, season)].append(idx)
            global_pool[season].append(idx)
    rows = []
    same_arr = {}
    for key, values in same_region.items():
        values = sorted(set(values), key=lambda i: quality.get((i, key[1]), 0.0), reverse=True)
        arr = np.asarray(values[:MAX_COMPONENTS_PER_POOL], dtype=np.int64)
        same_arr[key] = arr
        rows.append({"pool_type": "same_region", "region": key[0], "season": key[1], "N": len(arr), "median_component_score": float(np.median([quality[(i, key[1])] for i in arr])) if len(arr) else np.nan})
    global_arr = {}
    for season, values in global_pool.items():
        values = sorted(set(values), key=lambda i: quality.get((i, season), 0.0), reverse=True)
        arr = np.asarray(values[:MAX_COMPONENTS_PER_POOL], dtype=np.int64)
        global_arr[season] = arr
        rows.append({"pool_type": "global", "region": "ALL", "season": season, "N": len(arr), "median_component_score": float(np.median([quality[(i, season)] for i in arr])) if len(arr) else np.nan})
    return same_arr, global_arr, quality, pd.DataFrame(rows)


def neutralize_window(spectral: np.ndarray, doy: np.ndarray, window: AdaptiveWindow) -> np.ndarray:
    out = spectral.copy()
    idx = np.where((doy >= window.start_doy) & (doy <= window.end_doy))[0]
    if len(idx) == 0:
        return out
    left = int(idx.min()) - 1
    right = int(idx.max()) + 1
    if left >= 0 and right < len(doy) and left < right:
        phi = np.linspace(0.0, 1.0, len(idx), dtype=np.float32)[:, None]
        neutral = (1.0 - phi) * out[left][None, :] + phi * out[right][None, :]
    else:
        outside = np.ones(len(doy), dtype=bool); outside[idx] = False
        neutral_base = np.median(out[outside], axis=0) if outside.sum() >= 3 else np.median(out, axis=0)
        neutral = np.repeat(neutral_base[None, :], len(idx), axis=0)
    weights = taper_weights(doy[idx], float(doy[idx[0]]), float(doy[idx[-1]]))[:, None]
    alpha = np.clip(0.94 * weights, 0.0, 1.0)
    out[idx] = (1.0 - alpha) * out[idx] + alpha * neutral
    return out.astype(np.float32)


def make_neutral_canvas(
    base_idx: int,
    X: np.ndarray,
    DOY: np.ndarray,
    class_id: np.ndarray,
    regions: np.ndarray,
    source_file: np.ndarray,
    fi: FeatureIndex,
    library: WindowLibrary,
) -> Tuple[np.ndarray, List[AdaptiveWindow]]:
    base_spectral = moving_average_2d(X[base_idx][:, fi.spectral], SMOOTH_WINDOW)
    base_class = int(class_id[base_idx])
    base_region = str(regions[base_idx])
    removed: List[AdaptiveWindow] = []
    out = base_spectral.copy()
    if base_class > 0:
        for season in CLASS_SEASONS[base_class]:
            try:
                window = library.resolve_source(source_file[base_idx], base_region, base_class, season)
            except Exception:
                window = library.resolve_region_season(base_region, season)
            out = neutralize_window(out, DOY[base_idx], window)
            removed.append(window)
    return out.astype(np.float32), removed


def style_adapt_segment(
    source_segment: np.ndarray,
    donor_full_spectral: np.ndarray,
    donor_full_ndvi: np.ndarray,
    donor_doy: np.ndarray,
    source_window: AdaptiveWindow,
    base_full_spectral: np.ndarray,
    base_full_ndvi: np.ndarray,
    base_doy: np.ndarray,
    target_windows: Sequence[AdaptiveWindow],
    same_region: bool,
) -> np.ndarray:
    donor_mask = background_mask(donor_full_ndvi, [source_window], donor_doy)
    base_mask = background_mask(base_full_ndvi, target_windows, base_doy)
    src_bg = donor_full_spectral[donor_mask]
    base_bg = base_full_spectral[base_mask]
    src_med = np.median(src_bg, axis=0, keepdims=True)
    src_iqr = np.maximum(np.percentile(src_bg, 75, axis=0, keepdims=True) - np.percentile(src_bg, 25, axis=0, keepdims=True), SPECTRAL_SCALE_FLOOR)
    base_med = np.median(base_bg, axis=0, keepdims=True)
    base_iqr = np.maximum(np.percentile(base_bg, 75, axis=0, keepdims=True) - np.percentile(base_bg, 25, axis=0, keepdims=True), SPECTRAL_SCALE_FLOOR)
    scale = np.clip(base_iqr / src_iqr, STYLE_SCALE_MIN, STYLE_SCALE_MAX)
    matched = (source_segment - src_med) * scale + base_med
    strength = STYLE_ADAPT_STRENGTH * (0.45 if same_region else 1.0)
    return ((1.0 - strength) * source_segment + strength * matched).astype(np.float32)


def warp_component_to_target(
    donor_spectral: np.ndarray,
    donor_doy: np.ndarray,
    source_window: AdaptiveWindow,
    target_doy: np.ndarray,
    target_window: AdaptiveWindow,
) -> Tuple[np.ndarray, np.ndarray]:
    source_idx = np.where((donor_doy >= source_window.start_doy) & (donor_doy <= source_window.end_doy))[0]
    target_idx = np.where((target_doy >= target_window.start_doy) & (target_doy <= target_window.end_doy))[0]
    if len(source_idx) < 3 or len(target_idx) < 3:
        raise ValueError("component_window_too_short")
    src_phase = (donor_doy[source_idx] - source_window.start_doy) / max(source_window.end_doy - source_window.start_doy, 1.0)
    tgt_phase = (target_doy[target_idx] - target_window.start_doy) / max(target_window.end_doy - target_window.start_doy, 1.0)
    donor_seg = donor_spectral[source_idx]
    warped = np.empty((len(target_idx), donor_spectral.shape[1]), dtype=np.float32)
    for j in range(donor_spectral.shape[1]):
        warped[:, j] = np.interp(tgt_phase, src_phase, donor_seg[:, j], left=float(donor_seg[0, j]), right=float(donor_seg[-1, j]))
    return warped, target_idx


def select_donor(
    region: str,
    season: str,
    same_region_pools: Dict[Tuple[str, str], np.ndarray],
    global_pools: Dict[str, np.ndarray],
    component_quality: Dict[Tuple[int, str], float],
    parent_usage: Counter,
    used_indices: set,
) -> Tuple[int, str]:
    local = same_region_pools.get((region, season), np.asarray([], dtype=np.int64))
    pools = []
    if len(local) and rng.random() < SAME_REGION_DONOR_PROB:
        pools.append((local, "same_region"))
    pools.append((global_pools.get(season, np.asarray([], dtype=np.int64)), "cross_region_or_global"))
    if len(local):
        pools.append((local, "same_region_fallback"))
    for pool, mode in pools:
        if len(pool) == 0:
            continue
        candidates = rng.choice(pool, size=min(len(pool), 120), replace=False)
        candidates = sorted(candidates.tolist(), key=lambda i: component_quality.get((int(i), season), 0.0) - 0.02 * parent_usage[int(i)], reverse=True)
        for idx in candidates:
            idx = int(idx)
            if idx in used_indices or parent_usage[idx] >= MAX_PARENT_USAGE:
                continue
            return idx, mode
    raise RuntimeError(f"No donor for region={region}, season={season}")


def generate_candidate(
    target_region: str,
    target_class: int,
    base_idx: int,
    X: np.ndarray,
    DOY: np.ndarray,
    class_id: np.ndarray,
    regions: np.ndarray,
    source_file: np.ndarray,
    fi: FeatureIndex,
    library: WindowLibrary,
    same_region_pools: Dict[Tuple[str, str], np.ndarray],
    global_pools: Dict[str, np.ndarray],
    component_quality: Dict[Tuple[int, str], float],
    parent_usage: Counter,
) -> Tuple[np.ndarray, Dict[str, object], List[int], Dict[str, AdaptiveWindow]]:
    target_doy = DOY[base_idx].astype(np.float32).copy()
    base_full = X[base_idx].astype(np.float32)
    canvas, removed_windows = make_neutral_canvas(base_idx, X, DOY, class_id, regions, source_file, fi, library)
    target_windows = resolve_target_windows(library, target_region, target_class)
    target_window_list = [target_windows[s] for s in CLASS_SEASONS[target_class]]

    value_accum = np.zeros_like(canvas, dtype=np.float32)
    weight_accum = np.zeros((len(target_doy), 1), dtype=np.float32)
    count_accum = np.zeros(len(target_doy), dtype=np.int16)
    donor_indices: List[int] = []
    donor_modes = {}; donor_regions = {}; source_windows = {}
    used = {int(base_idx)}
    window_scores = []

    for season in CLASS_SEASONS[target_class]:
        donor_idx, donor_mode = select_donor(target_region, season, same_region_pools, global_pools, component_quality, parent_usage, used)
        used.add(donor_idx); donor_indices.append(donor_idx)
        donor_region = str(regions[donor_idx])
        source_window = library.resolve_source(source_file[donor_idx], donor_region, int(class_id[donor_idx]), season)
        target_window = target_windows[season]
        donor_spectral = moving_average_2d(X[donor_idx][:, fi.spectral], SMOOTH_WINDOW)
        warped, target_idx = warp_component_to_target(donor_spectral, DOY[donor_idx], source_window, target_doy, target_window)
        warped = style_adapt_segment(
            warped,
            donor_spectral,
            X[donor_idx, :, fi.ndvi],
            DOY[donor_idx],
            source_window,
            base_full[:, fi.spectral],
            base_full[:, fi.ndvi],
            target_doy,
            target_window_list,
            same_region=(donor_region == target_region),
        )
        weights = taper_weights(target_doy[target_idx], target_window.start_doy, target_window.end_doy)[:, None]
        value_accum[target_idx] += warped * weights
        weight_accum[target_idx] += weights
        count_accum[target_idx] += 1
        donor_modes[season] = donor_mode
        donor_regions[season] = donor_region
        source_windows[season] = source_window
        window_scores.extend([source_window.shape_score, target_window.shape_score])

    candidate_spectral = canvas.copy()
    valid = weight_accum[:, 0] > 0
    if valid.any():
        component_average = value_accum / np.maximum(weight_accum, EPS)
        alpha = np.clip(weight_accum, 0.0, 1.0)
        alpha[count_accum >= 2] *= (1.0 - CONFLICT_BASE_WEIGHT)
        candidate_spectral = ((1.0 - alpha) * canvas + alpha * component_average).astype(np.float32)
    candidate = base_full.copy()
    candidate[:, fi.spectral] = candidate_spectral
    candidate = recompute_indices(candidate, fi)
    candidate = np.nan_to_num(candidate, nan=0.0, posinf=0.0, neginf=0.0)

    meta = {
        "synthetic_method": "hierarchical_adaptive_full_region_morph_v6",
        "removed_base_windows": {s: asdict(w) for s, w in zip(CLASS_SEASONS[int(class_id[base_idx])], removed_windows)} if int(class_id[base_idx]) > 0 else {},
        "source_windows": {s: asdict(w) for s, w in source_windows.items()},
        "target_windows": {s: asdict(w) for s, w in target_windows.items()},
        "donor_modes": donor_modes,
        "donor_source_regions": donor_regions,
        "overlap_step_n": int(np.sum(count_accum >= 2)),
        "window_confidence": float(min(window_scores)) if window_scores else 0.0,
        "merged_middle_late": bool(any(w.merged_middle_late for w in target_windows.values())),
    }
    return candidate.astype(np.float32), meta, donor_indices, target_windows


# =============================================================================
# 6. Per-sample no-teacher quality review
# =============================================================================

def phenology_presence_pass(
    descriptor: np.ndarray,
    target_class: int,
    class_reference: RobustReference,
) -> Tuple[bool, Dict[str, float]]:
    expected = set(CLASS_SEASONS[target_class])
    details: Dict[str, float] = {}
    passed = True
    for season_i, season in enumerate(SEASONS):
        peak_idx = season_i * 5
        amp_idx = peak_idx + 1
        amp = float(descriptor[amp_idx]); peak_offset = float(descriptor[peak_idx])
        ref_amp_low = float(class_reference.p01[amp_idx]); ref_amp_high = float(class_reference.p99[amp_idx])
        ref_peak_low = float(class_reference.p01[peak_idx]); ref_peak_high = float(class_reference.p99[peak_idx])
        if season in expected:
            amp_floor = max(0.09, ref_amp_low * 0.78)
            season_pass = amp >= amp_floor and ref_peak_low - 18.0 <= peak_offset <= ref_peak_high + 18.0
        else:
            amp_ceiling = max(0.17, ref_amp_high * 1.18)
            season_pass = amp <= amp_ceiling
        passed &= bool(season_pass)
        details[f"{season}_amp"] = amp
        details[f"{season}_peak_offset"] = peak_offset
        details[f"{season}_presence_pass"] = float(season_pass)
    return bool(passed), details


def count_prominent_peaks(doy: np.ndarray, ndvi: np.ndarray) -> List[float]:
    y = smooth_1d(ndvi, 5)
    yn, _, _ = robust_norm_1d(y)
    rows = detect_candidate_peaks(yn, doy, 1.0, 366.0)
    return sorted([float(r["doy"]) for r in rows if r["prominence"] >= 0.06 or r["height"] >= 0.55])


def peak_structure_pass(
    candidate: np.ndarray,
    doy: np.ndarray,
    fi: FeatureIndex,
    target_class: int,
    target_windows: Dict[str, AdaptiveWindow],
) -> Tuple[bool, Dict[str, float]]:
    peaks = count_prominent_peaks(doy, candidate[:, fi.ndvi])
    expected_n = len(CLASS_SEASONS[target_class])
    merged_allowed = target_class in {6, 7} and any(w.merged_middle_late for w in target_windows.values())
    min_expected = expected_n - (1 if merged_allowed else 0)
    max_expected = expected_n + 1
    count_ok = min_expected <= len(peaks) <= max_expected
    window_hits = 0
    for season in CLASS_SEASONS[target_class]:
        w = target_windows[season]
        if any(w.start_doy - 8.0 <= p <= w.end_doy + 8.0 for p in peaks):
            window_hits += 1
    required_hits = expected_n - (1 if merged_allowed else 0)
    pass_value = bool(count_ok and window_hits >= required_hits)
    return pass_value, {
        "prominent_peak_n": float(len(peaks)),
        "expected_peak_n": float(expected_n),
        "expected_window_hit_n": float(window_hits),
        "merged_middle_late_allowed": float(merged_allowed),
        "peak_structure_pass": float(pass_value),
    }


def feature_outlier_rate(candidate: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((candidate < lower[None, :]) | (candidate > upper[None, :])))


def evaluate_candidate(
    candidate: np.ndarray,
    target_doy: np.ndarray,
    target_class: int,
    target_region: str,
    target_windows: Dict[str, AdaptiveWindow],
    window_confidence: float,
    base_curve: np.ndarray,
    donor_curves_resampled: List[np.ndarray],
    fi: FeatureIndex,
    library: WindowLibrary,
    class_refs: Dict[int, RobustReference],
    region_refs: Dict[str, RobustReference],
    continuity_p95: Dict[str, np.ndarray],
    boundary_p95: Dict[str, np.ndarray],
    real_feature_lower: np.ndarray,
    real_feature_upper: np.ndarray,
    real_feature_iqr: np.ndarray,
) -> Tuple[CandidateQuality, Dict[str, float]]:
    def reject(reason: str, formula=np.inf, outlier=1.0):
        return CandidateQuality("Reject", 0.0, float(formula), float(outlier), np.inf, 1.0, np.inf, 1.0, np.inf, np.inf, float(window_confidence), 0.0, 0.0, False, False, False, reason), {}
    if not np.isfinite(candidate).all():
        return reject("non_finite")
    formula_candidate = recompute_indices(candidate, fi)
    formula_mae = float(np.mean(np.abs(formula_candidate[:, [fi.ndvi, fi.lswi]] - candidate[:, [fi.ndvi, fi.lswi]])))
    if formula_mae > FORMULA_MAE_HARD:
        return reject("derived_index_formula_failed", formula_mae)
    outlier_rate = feature_outlier_rate(candidate, real_feature_lower, real_feature_upper)
    all_windows = {season: target_windows.get(season, library.resolve_region_season(target_region, season)) for season in SEASONS}
    pheno = describe_phenology(candidate, target_doy, fi, all_windows)
    class_distance, class_outlier, _ = robust_distance(pheno, class_refs[target_class])
    phenology_pass, phenology_details = phenology_presence_pass(pheno, target_class, class_refs[target_class])
    structure_pass, structure_details = peak_structure_pass(candidate, target_doy, fi, target_class, target_windows)
    style = describe_region_style(candidate, fi)
    region_distance, region_outlier, _ = robust_distance(style, region_refs[target_region])
    continuity = continuity_vector(candidate, target_doy, fi)
    continuity_ratio = float(np.max(continuity / np.maximum(continuity_p95[target_region], EPS)))
    boundaries = boundary_vector(candidate, target_doy, fi, list(target_windows.values()))
    boundary_ratio = float(np.max(boundaries / np.maximum(boundary_p95[target_region], EPS)))
    parent_curves = [base_curve] + donor_curves_resampled
    novelty_metrics = [nrmse_and_change(candidate, p, real_feature_iqr, fi.spectral) for p in parent_curves]
    nearest_parent_nrmse = float(min(v[0] for v in novelty_metrics))
    nearest_idx = int(np.argmin([v[0] for v in novelty_metrics])); change_fraction = float(novelty_metrics[nearest_idx][1])
    hard_noop = nearest_parent_nrmse < HARD_NOOP_NRMSE and change_fraction < HARD_NOOP_CHANGE_FRACTION
    novelty_pass = not (nearest_parent_nrmse < NEAR_COPY_NRMSE and change_fraction < NEAR_COPY_CHANGE_FRACTION)

    class_score = float(np.clip(1.0 - class_distance / 3.6, 0.0, 1.0))
    region_score = float(np.clip(1.0 - region_distance / 3.6, 0.0, 1.0))
    continuity_score = float(np.clip(1.0 - (continuity_ratio - 1.0) / 2.6, 0.0, 1.0))
    boundary_score = float(np.clip(1.0 - (boundary_ratio - 1.0) / 2.8, 0.0, 1.0))
    novelty_score = float(np.clip((nearest_parent_nrmse - 0.005) / 0.08, 0.0, 1.0))
    physical_score = float(np.clip(1.0 - outlier_rate / 0.08, 0.0, 1.0))
    window_score = float(np.clip(window_confidence, 0.0, 1.0))
    score = (
        0.25 * class_score + 0.19 * region_score + 0.13 * continuity_score
        + 0.12 * boundary_score + 0.10 * float(phenology_pass)
        + 0.07 * float(structure_pass) + 0.06 * window_score
        + 0.04 * novelty_score + 0.04 * physical_score
    )
    hard_reason = ""
    if hard_noop: hard_reason = "hard_noop"
    elif outlier_rate > 0.12: hard_reason = "feature_domain_hard_fail"
    elif class_distance > 4.6: hard_reason = "class_phenology_hard_fail"
    elif region_distance > 4.6: hard_reason = "region_style_hard_fail"
    elif continuity_ratio > 4.2: hard_reason = "continuity_hard_fail"
    elif boundary_ratio > 4.5: hard_reason = "boundary_hard_fail"
    elif window_confidence < 0.20: hard_reason = "window_confidence_hard_fail"
    core_a = (
        not hard_reason and outlier_rate <= FEATURE_OUTLIER_RATE_CORE_A
        and class_distance <= CLASS_DISTANCE_CORE_A and class_outlier <= DESCRIPTOR_OUTLIER_CORE_A
        and region_distance <= REGION_DISTANCE_CORE_A and region_outlier <= DESCRIPTOR_OUTLIER_CORE_A
        and continuity_ratio <= CONTINUITY_RATIO_CORE_A and boundary_ratio <= BOUNDARY_RATIO_CORE_A
        and window_confidence >= MIN_WINDOW_CONFIDENCE_CORE_A
        and phenology_pass and structure_pass and novelty_pass and score >= MIN_CORE_A_SCORE
    )
    core_b = (
        not hard_reason and outlier_rate <= FEATURE_OUTLIER_RATE_CORE_B
        and class_distance <= CLASS_DISTANCE_CORE_B and class_outlier <= DESCRIPTOR_OUTLIER_CORE_B
        and region_distance <= REGION_DISTANCE_CORE_B and region_outlier <= DESCRIPTOR_OUTLIER_CORE_B
        and continuity_ratio <= CONTINUITY_RATIO_CORE_B and boundary_ratio <= BOUNDARY_RATIO_CORE_B
        and window_confidence >= MIN_WINDOW_CONFIDENCE_CORE_B
        and phenology_pass and score >= MIN_CORE_B_SCORE
    )
    if core_a: tier = "Core-A"
    elif core_b: tier = "Core-B"
    else:
        tier = "Reject"
        if not hard_reason:
            if not phenology_pass: hard_reason = "phenology_presence_failed"
            elif not structure_pass: hard_reason = "peak_structure_failed"
            elif not novelty_pass: hard_reason = "near_copy"
            elif window_confidence < MIN_WINDOW_CONFIDENCE_CORE_B: hard_reason = "window_confidence_low"
            else: hard_reason = "soft_quality_gate_failed"
    q = CandidateQuality(
        tier, float(score), formula_mae, outlier_rate,
        class_distance, class_outlier, region_distance, region_outlier,
        continuity_ratio, boundary_ratio, float(window_confidence),
        nearest_parent_nrmse, change_fraction,
        phenology_pass, structure_pass, novelty_pass, hard_reason,
    )
    extras = {
        **phenology_details, **structure_details,
        "continuity_ndvi_max_jump": float(continuity[0]),
        "continuity_lswi_max_jump": float(continuity[1]),
        "continuity_ndvi_max_slope": float(continuity[2]),
        "continuity_lswi_max_slope": float(continuity[3]),
        "continuity_ndvi_curv_q95": float(continuity[4]),
        "continuity_lswi_curv_q95": float(continuity[5]),
        "boundary_ndvi_value_jump": float(boundaries[0]),
        "boundary_lswi_value_jump": float(boundaries[1]),
        "boundary_ndvi_slope_jump": float(boundaries[2]),
        "boundary_lswi_slope_jump": float(boundaries[3]),
        "boundary_ndvi_accel_jump": float(boundaries[4]),
        "boundary_lswi_accel_jump": float(boundaries[5]),
    }
    return q, extras

# =============================================================================
# 7. 保存与诊断
# =============================================================================

def append_parent_slots(donor_indices: List[int]) -> Tuple[int, int, int]:
    slots = list(donor_indices[:3]) + [-1, -1, -1]
    return int(slots[0]), int(slots[1]), int(slots[2])


def save_subset_npz(
    path: Path,
    records: List[Dict[str, object]],
    feature_names: np.ndarray,
    tier: str,
) -> int:
    selected = [r for r in records if r["quality_tier"] == tier]
    if not selected:
        np.savez_compressed(
            path,
            X=np.empty((0, 72, len(feature_names)), dtype=np.float32),
            DOY=np.empty((0, 72), dtype=np.float32),
            class_id=np.empty((0,), dtype=np.int64),
            feature_names=feature_names,
        )
        return 0

    def arr(key: str, dtype=None):
        values = [r[key] for r in selected]
        return np.asarray(values, dtype=dtype)

    X_out = np.stack([r["X"] for r in selected]).astype(np.float32)
    DOY_out = np.stack([r["DOY"] for r in selected]).astype(np.float32)
    class_out = arr("class_id", np.int64)
    Y_out = np.stack([CLASS_Y[int(v)] for v in class_out]).astype(np.float32)
    parent_a = arr("parent_a_dataset_index", np.int64)
    parent_b = arr("parent_b_dataset_index", np.int64)
    parent_c = arr("parent_c_dataset_index", np.int64)

    np.savez_compressed(
        path,
        X=X_out,
        DOY=DOY_out,
        DOY_input=DOY_out,
        time_mask=np.ones(DOY_out.shape, dtype=np.float32),
        Y=Y_out,
        class_id=class_out,
        synthetic_flag=np.ones(len(selected), dtype=np.int8),
        target_region=arr("target_region", object),
        target_region_id=arr("target_region_id", np.int64),
        target_subtype=np.full(len(selected), -1, dtype=np.int64),
        synthetic_method=arr("synthetic_method", object),
        target_doy_template_dataset_index=arr(
            "target_doy_template_dataset_index", np.int64
        ),
        base_dataset_index=arr("base_dataset_index", np.int64),
        parent_a_dataset_index=parent_a,
        parent_b_dataset_index=parent_b,
        parent_c_dataset_index=parent_c,
        parent_early_dataset_index=arr("parent_early_dataset_index", np.int64),
        parent_middle_dataset_index=arr("parent_middle_dataset_index", np.int64),
        parent_late_dataset_index=arr("parent_late_dataset_index", np.int64),
        quality_tier=arr("quality_tier", object),
        quality_score=arr("quality_score", np.float32),
        class_descriptor_distance=arr("class_distance", np.float32),
        region_style_distance=arr("region_distance", np.float32),
        continuity_ratio=arr("continuity_ratio", np.float32),
        boundary_ratio=arr("boundary_ratio", np.float32),
        window_confidence=arr("window_confidence", np.float32),
        nearest_parent_nrmse=arr("nearest_parent_nrmse", np.float32),
        change_fraction_from_nearest_parent=arr("change_fraction", np.float32),
        candidate_id=arr("candidate_id", np.int64),
        feature_names=feature_names,
        class_id_to_name=np.asarray(
            [CLASS_NAMES[i] for i in range(8)], dtype=object
        ),
        generation_version=np.asarray(["PTS-v6-hierarchical-adaptive-full-region"], dtype=object),
        no_rice_strategy=np.asarray(
            ["real_global_negative_anchor_not_synthesized"], dtype=object
        ),
    )
    return len(selected)


def make_coverage_heatmap(summary: pd.DataFrame, output: Path) -> None:
    if summary.empty:
        return
    pivot = summary.pivot(
        index="target_region", columns="class_name", values="core_a_N"
    ).fillna(0)
    ordered_classes = [CLASS_NAMES[i] for i in range(1, 8)]
    pivot = pivot.reindex(columns=ordered_classes).fillna(0)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.7 * len(pivot) + 2)))
    im = ax.imshow(pivot.to_numpy(), aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("PTS-v6 Core-A coverage: target region × rice system")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{int(pivot.iloc[i, j]):,}", ha="center", va="center")
    fig.colorbar(im, ax=ax, label="Core-A N")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_region_median_curves(
    records: List[Dict[str, object]],
    fi: FeatureIndex,
    output_dir: Path,
) -> None:
    core_a = [r for r in records if r["quality_tier"] == "Core-A"]
    if not core_a:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    regions = sorted(set(str(r["target_region"]) for r in core_a))
    for region in regions:
        fig, ax = plt.subplots(figsize=(11, 6))
        for cid in range(1, 8):
            part = [
                r for r in core_a
                if r["target_region"] == region and int(r["class_id"]) == cid
            ]
            if not part:
                continue
            sample = part[: min(300, len(part))]
            # 同一地区可能存在多个真实DOY模板；先插值到统一绘图轴，仅用于诊断。
            grid = np.linspace(1, 366, 72)
            curves = []
            for r in sample:
                curves.append(np.interp(grid, r["DOY"], r["X"][:, fi.ndvi]))
            ax.plot(grid, np.median(np.stack(curves), axis=0), label=CLASS_NAMES[cid])
        ax.set_title(f"PTS-v6 median NDVI by class — {region}")
        ax.set_xlabel("DOY")
        ax.set_ylabel("NDVI")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{region}_median_ndvi_by_class.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)



# =============================================================================
# 8. Main
# =============================================================================

def main() -> None:
    require_file(REAL_TRAIN_NPZ)
    output_info = prepare_output_dir(OUTPUT_ROOT)
    print("=" * 120)
    print("PTS-v6 Hierarchical-Adaptive Full-Region Generator")
    print("Real Train :", REAL_TRAIN_NPZ)
    print("Output     :", OUTPUT_ROOT)
    print("Action     :", output_info)
    print("Seed       :", SEED)
    print("Core-A/cell:", TARGET_CORE_A_PER_CELL)
    print("Core-B/cell:", MAX_CORE_B_PER_CELL)
    print("=" * 120)

    print("[1/11] Loading Real Train...")
    with np.load(REAL_TRAIN_NPZ, allow_pickle=True) as data:
        X = np.asarray(data["X"], dtype=np.float32)
        DOY = ensure_doy_matrix(data["DOY"] if "DOY" in data.files else data["DOY_input"], len(X))
        class_id = resolve_class_id(data)
        source_file = decode_array(data["source_file"]) if "source_file" in data.files else np.asarray(["unknown"] * len(X), dtype=object)
        source_index = np.asarray(data["source_index_in_file"], dtype=np.int64).reshape(-1) if "source_index_in_file" in data.files else np.arange(len(X), dtype=np.int64)
        region_id_raw = np.asarray(data["region_id"], dtype=np.int64).reshape(-1) if "region_id" in data.files else np.full(len(X), -1, dtype=np.int64)
        feature_names = decode_array(data["feature_names"]) if "feature_names" in data.files else np.asarray([f"f{i}" for i in range(X.shape[2])], dtype=object)
    if X.ndim != 3 or X.shape[1:] != (72, 10):
        raise ValueError(f"Expected X=[N,72,10], got {X.shape}")
    if DOY.shape != X.shape[:2] or not np.isfinite(X).all() or not np.isfinite(DOY).all():
        raise ValueError("Real Train X/DOY invalid")
    fi = resolve_feature_index(feature_names, X.shape[2])
    X = recompute_indices(X, fi)
    regions = np.asarray([infer_region_from_source_file(v) for v in source_file], dtype=object)
    target_regions = sorted(set(regions[class_id > 0].tolist()))
    region_to_id = {r: int(np.median(region_id_raw[regions == r])) if np.any(regions == r) else i for i, r in enumerate(target_regions)}
    print("X shape       :", X.shape)
    print("Feature names :", feature_names.tolist())
    print("Indices       :", asdict(fi))
    print("Target regions:", target_regions)
    print("Class counts  :", dict(Counter(class_id.tolist())))

    print("[2/11] Building hierarchical adaptive windows...")
    library = build_hierarchical_window_library(X, DOY, class_id, regions, source_file, fi)
    window_df = pd.DataFrame(library.rows)
    window_df.to_csv(OUTPUT_ROOT / "01_hierarchical_adaptive_windows.csv", index=False, encoding="utf-8-sig")
    print("Global peak priors:", library.global_peak_priors)
    print("Usable windows:", int((window_df.get("status", pd.Series(dtype=str)) == "ok").sum()))

    print("[3/11] Building adaptive class/region references...")
    class_refs, region_refs, continuity_p95, boundary_p95, reference_df = build_references(
        X, DOY, class_id, regions, source_file, fi, library
    )
    reference_df.to_csv(OUTPUT_ROOT / "02_reference_summary.csv", index=False, encoding="utf-8-sig")

    sample_n = min(len(X), 50000)
    bound_idx = rng.choice(len(X), sample_n, replace=False)
    flat = X[bound_idx].reshape(-1, X.shape[2])
    q001, q999 = np.percentile(flat, [0.1, 99.9], axis=0)
    q25, q75 = np.percentile(flat, [25, 75], axis=0)
    real_iqr = np.maximum(q75 - q25, 1e-4).astype(np.float32)
    feature_lower = (q001 - 0.20 * real_iqr).astype(np.float32)
    feature_upper = (q999 + 0.20 * real_iqr).astype(np.float32)
    feature_lower[fi.ndvi] = max(feature_lower[fi.ndvi], -1.0); feature_upper[fi.ndvi] = min(feature_upper[fi.ndvi], 1.0)
    feature_lower[fi.lswi] = max(feature_lower[fi.lswi], -1.0); feature_upper[fi.lswi] = min(feature_upper[fi.lswi], 1.0)

    print("[4/11] Building adaptive component banks...")
    same_region_pools, global_pools, component_quality, pool_df = build_component_pools(
        X, DOY, class_id, regions, source_file, fi, library
    )
    pool_df.to_csv(OUTPUT_ROOT / "03_component_pool_summary.csv", index=False, encoding="utf-8-sig")

    base_pools = {region: np.where((regions == region) & (class_id > 0))[0] for region in target_regions}
    for region, pool in base_pools.items():
        if len(pool) == 0:
            raise RuntimeError(f"Region {region} has no real rice base samples")

    print("[5/11] Preparing full region × class cells...")
    cells = [(region, cid) for region in target_regions for cid in range(1, 8)]
    if DEBUG_CELL_FILTER:
        rules = []
        for token in DEBUG_CELL_FILTER.replace(",", ";").split(";"):
            token = token.strip()
            if not token: continue
            if ":" not in token:
                raise ValueError("PTS6_DEBUG_CELL_FILTER format: Xianning:*;*:6")
            rr, cc = [v.strip() for v in token.split(":", 1)]
            rules.append((rr, cc))
        def selected(cell):
            region, cid = cell
            return any((rr in {"*", region}) and (cc == "*" or int(cc) == cid) for rr, cc in rules)
        cells = [c for c in cells if selected(c)]
        if not cells: raise ValueError(f"No cells matched {DEBUG_CELL_FILTER}")
    if DEBUG_MAX_CELLS > 0:
        cells = cells[:DEBUG_MAX_CELLS]
    pd.DataFrame([{"target_region": r, "class_id": c, "class_name": CLASS_NAMES[c], "target_core_a_N": TARGET_CORE_A_PER_CELL, "max_core_b_N": MAX_CORE_B_PER_CELL} for r, c in cells]).to_csv(
        OUTPUT_ROOT / "04_generation_plan_region_class.csv", index=False, encoding="utf-8-sig"
    )

    accepted_records: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, object]] = []
    reject_examples: List[Dict[str, object]] = []
    rejection_counter = Counter(); base_usage = Counter(); parent_usage = Counter(); template_usage = Counter(); parent_combination_usage = Counter(); coarse_hash_usage = Counter()
    candidate_id = 0; cell_summary_rows = []

    print("[6/11] Generating and reviewing candidates...")
    for cell_i, (target_region, target_class) in enumerate(cells, 1):
        core_a_count = 0; core_b_count = 0; attempts = 0
        max_attempts = TARGET_CORE_A_PER_CELL * MAX_ATTEMPT_MULTIPLIER
        local_reject = Counter(); base_pool = base_pools[target_region]
        print("-" * 120)
        print(f"Cell {cell_i}/{len(cells)}: {target_region} × {CLASS_NAMES[target_class]}")
        while core_a_count < TARGET_CORE_A_PER_CELL and attempts < max_attempts:
            attempts += 1; candidate_id += 1
            eligible = [int(i) for i in rng.permutation(base_pool)[:min(len(base_pool), 180)] if base_usage[int(i)] < MAX_BASE_USAGE and template_usage[int(i)] < MAX_TEMPLATE_USAGE]
            if not eligible:
                local_reject["base_usage_exhausted"] += 1; rejection_counter["base_usage_exhausted"] += 1; continue
            base_idx = eligible[0]
            try:
                candidate, meta, donor_indices, target_windows = generate_candidate(
                    target_region, target_class, base_idx, X, DOY, class_id, regions, source_file, fi,
                    library, same_region_pools, global_pools, component_quality, parent_usage,
                )
            except Exception as exc:
                reason = f"generation_failed:{type(exc).__name__}"
                local_reject[reason] += 1; rejection_counter[reason] += 1
                if len(reject_examples) < 5000:
                    reject_examples.append({"candidate_id": candidate_id, "target_region": target_region, "class_id": target_class, "reason": reason, "detail": str(exc)})
                continue
            combination_key = (target_region, target_class, base_idx, *sorted(donor_indices))
            if parent_combination_usage[combination_key] >= MAX_PARENT_COMBINATION_REPEAT:
                local_reject["duplicate_parent_combination"] += 1; rejection_counter["duplicate_parent_combination"] += 1; continue
            coarse_hash = short_hash_array(candidate[:, [fi.ndvi, fi.lswi]], 2)
            if coarse_hash_usage[coarse_hash] >= MAX_COARSE_DUPLICATE_REPEAT:
                local_reject["coarse_curve_duplicate"] += 1; rejection_counter["coarse_curve_duplicate"] += 1; continue
            donor_resampled = []
            for idx in donor_indices:
                resampled = np.empty_like(candidate)
                for j in range(X.shape[2]):
                    resampled[:, j] = np.interp(DOY[base_idx], DOY[idx], X[idx, :, j], left=float(X[idx, 0, j]), right=float(X[idx, -1, j]))
                donor_resampled.append(recompute_indices(resampled, fi))
            quality, extras = evaluate_candidate(
                candidate, DOY[base_idx], target_class, target_region, target_windows, meta["window_confidence"],
                X[base_idx], donor_resampled, fi, library, class_refs, region_refs, continuity_p95, boundary_p95,
                feature_lower, feature_upper, real_iqr,
            )
            if quality.tier == "Reject":
                reason = quality.hard_fail_reason or "reject"
                local_reject[reason] += 1; rejection_counter[reason] += 1
                if len(reject_examples) < 5000:
                    reject_examples.append({"candidate_id": candidate_id, "target_region": target_region, "class_id": target_class, "reason": reason, "quality_score": quality.score, "class_distance": quality.class_distance, "region_distance": quality.region_distance, "continuity_ratio": quality.continuity_ratio, "boundary_ratio": quality.boundary_ratio, "window_confidence": quality.window_confidence})
                continue
            if quality.tier == "Core-B" and core_b_count >= MAX_CORE_B_PER_CELL:
                local_reject["core_b_quota_full"] += 1; rejection_counter["core_b_quota_full"] += 1; continue
            base_usage[base_idx] += 1; template_usage[base_idx] += 1
            for idx in donor_indices: parent_usage[idx] += 1
            parent_combination_usage[combination_key] += 1; coarse_hash_usage[coarse_hash] += 1
            parent_by_season = {s: -1 for s in SEASONS}
            for season, idx in zip(CLASS_SEASONS[target_class], donor_indices): parent_by_season[season] = int(idx)
            pa, pb, pc = append_parent_slots(donor_indices)
            record = {
                "candidate_id": candidate_id, "X": candidate.astype(np.float32), "DOY": DOY[base_idx].astype(np.float32).copy(),
                "class_id": target_class, "class_name": CLASS_NAMES[target_class], "target_region": target_region,
                "target_region_id": region_to_id[target_region], "quality_tier": quality.tier, "quality_score": quality.score,
                "formula_mae": quality.formula_mae, "feature_outlier_rate": quality.feature_outlier_rate,
                "class_distance": quality.class_distance, "class_outlier_rate": quality.class_outlier_rate,
                "region_distance": quality.region_distance, "region_outlier_rate": quality.region_outlier_rate,
                "continuity_ratio": quality.continuity_ratio, "boundary_ratio": quality.boundary_ratio,
                "window_confidence": quality.window_confidence,
                "nearest_parent_nrmse": quality.nearest_parent_nrmse, "change_fraction": quality.change_fraction,
                "phenology_pass": quality.phenology_pass, "peak_structure_pass": quality.peak_structure_pass,
                "novelty_pass": quality.novelty_pass, "synthetic_method": meta["synthetic_method"],
                "base_dataset_index": base_idx, "target_doy_template_dataset_index": base_idx,
                "parent_a_dataset_index": pa, "parent_b_dataset_index": pb, "parent_c_dataset_index": pc,
                "parent_early_dataset_index": parent_by_season["early"], "parent_middle_dataset_index": parent_by_season["middle"], "parent_late_dataset_index": parent_by_season["late"],
                "base_source_file": str(source_file[base_idx]), "base_source_index": int(source_index[base_idx]),
                "donor_indices_json": json.dumps(donor_indices), "donor_modes_json": json.dumps(meta["donor_modes"]),
                "donor_regions_json": json.dumps(meta["donor_source_regions"]),
                "source_windows_json": json.dumps(meta["source_windows"], ensure_ascii=False),
                "target_windows_json": json.dumps(meta["target_windows"], ensure_ascii=False),
                "removed_base_windows_json": json.dumps(meta["removed_base_windows"], ensure_ascii=False),
                "merged_middle_late": meta["merged_middle_late"], "overlap_step_n": meta["overlap_step_n"],
                "doy_pattern_id": short_hash_array(DOY[base_idx], 2), "coarse_curve_hash": coarse_hash,
                **extras,
            }
            accepted_records.append(record); manifest_rows.append({k: v for k, v in record.items() if k not in {"X", "DOY"}})
            if quality.tier == "Core-A": core_a_count += 1
            else: core_b_count += 1
            if (core_a_count + core_b_count) % 100 == 0:
                print(f"  accepted={core_a_count + core_b_count}, Core-A={core_a_count}, Core-B={core_b_count}, attempts={attempts}")
        cell_summary_rows.append({
            "target_region": target_region, "class_id": target_class, "class_name": CLASS_NAMES[target_class],
            "target_core_a_N": TARGET_CORE_A_PER_CELL, "core_a_N": core_a_count, "core_b_N": core_b_count,
            "attempt_N": attempts, "core_a_fill_rate": core_a_count / max(TARGET_CORE_A_PER_CELL, 1),
            "accept_rate": (core_a_count + core_b_count) / max(attempts, 1), "top_reject_reasons": json.dumps(local_reject.most_common(12)),
        })
        print(f"  DONE: Core-A={core_a_count}/{TARGET_CORE_A_PER_CELL}, Core-B={core_b_count}, attempts={attempts}")

    print("[7/11] Saving NPZ, manifests, and rejection reports...")
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = OUTPUT_ROOT / "pts_v6_hierarchical_adaptive_accepted_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    cell_summary_df = pd.DataFrame(cell_summary_rows)
    cell_summary_path = OUTPUT_ROOT / "05_region_class_generation_summary.csv"
    cell_summary_df.to_csv(cell_summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(reject_examples).to_csv(OUTPUT_ROOT / "06_reject_examples.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"reason": k, "N": v} for k, v in rejection_counter.most_common()]).to_csv(OUTPUT_ROOT / "07_rejection_reason_summary.csv", index=False, encoding="utf-8-sig")
    core_a_path = OUTPUT_ROOT / "pts_v6_hierarchical_adaptive_core_a_synthetic.npz"
    core_b_path = OUTPUT_ROOT / "pts_v6_hierarchical_adaptive_core_b_synthetic.npz"
    core_a_n = save_subset_npz(core_a_path, accepted_records, feature_names, "Core-A")
    core_b_n = save_subset_npz(core_b_path, accepted_records, feature_names, "Core-B")

    print("[8/11] Anti-shortcut acceptance checks...")
    core_a_manifest = manifest_df[manifest_df["quality_tier"] == "Core-A"].copy() if not manifest_df.empty else pd.DataFrame()
    if len(core_a_manifest):
        coverage = core_a_manifest.groupby(["target_region", "class_id", "class_name"]).size().reset_index(name="core_a_N")
        region_correct = sum(part["class_id"].value_counts().max() for _, part in core_a_manifest.groupby("target_region"))
        region_only_accuracy = float(region_correct / len(core_a_manifest))
        doy_correct = sum(part["class_id"].value_counts().max() for _, part in core_a_manifest.groupby("doy_pattern_id"))
        doy_template_accuracy = float(doy_correct / len(core_a_manifest))
    else:
        coverage = pd.DataFrame(columns=["target_region", "class_id", "class_name", "core_a_N"])
        region_only_accuracy = np.nan; doy_template_accuracy = np.nan
    coverage.to_csv(OUTPUT_ROOT / "08_core_a_region_class_coverage.csv", index=False, encoding="utf-8-sig")
    make_coverage_heatmap(coverage, OUTPUT_ROOT / "core_a_region_class_coverage_heatmap.png")

    print("[9/11] Writing diagnostic plots...")
    plot_region_median_curves(accepted_records, fi, OUTPUT_ROOT / "figures_median_ndvi_by_region")

    print("[10/11] Writing lineage-use tables...")
    pd.DataFrame([{"dataset_index": idx, "usage_N": n, "class_id": int(class_id[idx]), "class_name": CLASS_NAMES[int(class_id[idx])], "region": str(regions[idx]), "source_file": str(source_file[idx]), "source_index_in_file": int(source_index[idx])} for idx, n in parent_usage.most_common()]).to_csv(OUTPUT_ROOT / "09_parent_usage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"dataset_index": idx, "usage_N": n, "class_id": int(class_id[idx]), "class_name": CLASS_NAMES[int(class_id[idx])], "region": str(regions[idx]), "source_file": str(source_file[idx]), "source_index_in_file": int(source_index[idx])} for idx, n in base_usage.most_common()]).to_csv(OUTPUT_ROOT / "10_base_usage.csv", index=False, encoding="utf-8-sig")

    print("[11/11] Writing summary...")
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "generator": "PTS hierarchical-adaptive generator",
        "real_train_npz": str(REAL_TRAIN_NPZ), "output_root": str(OUTPUT_ROOT),
        "target_regions": target_regions, "target_classes": {i: CLASS_NAMES[i] for i in range(1, 8)},
        "no_rice_strategy": "real global negative anchor; no synthetic no_rice",
        "target_core_a_per_cell": TARGET_CORE_A_PER_CELL, "core_a_N": core_a_n, "core_b_N": core_b_n,
        "planned_cell_N": len(cells), "full_expected_core_a_N": len(cells) * TARGET_CORE_A_PER_CELL,
        "region_only_majority_accuracy_on_core_a": region_only_accuracy,
        "expected_region_only_random_accuracy": 1.0 / 7.0,
        "doy_template_majority_accuracy_on_core_a": doy_template_accuracy,
        "global_peak_priors": library.global_peak_priors,
        "configuration": {
            "hierarchical_window_levels": ["source_class", "region_class", "region_season", "global_class_shifted_to_region", "global_season"],
            "window_min_use_score": WINDOW_MIN_USE_SCORE, "window_core_a_score": WINDOW_CORE_A_SCORE,
            "valley_min_depth_norm": VALLEY_MIN_DEPTH_NORM, "merged_ml_max_gap_days": MERGED_ML_MAX_PEAK_GAP_DAYS,
            "style_adapt_strength": STYLE_ADAPT_STRENGTH, "same_region_donor_probability": SAME_REGION_DONOR_PROB,
            "quality_teacher_used": False,
        },
        "checks": {
            "all_sources_from_real_train": True, "real_val_read": False, "external_test_read": False,
            "derived_indices_recomputed_inside_generator": True, "region_class_full_coverage_targeted": True,
            "source_and_target_windows_adaptive": True, "midpoint_only_used_as_final_boundary_fallback": True,
        },
        "files": {"core_a_npz": str(core_a_path), "core_b_npz": str(core_b_path), "accepted_manifest": str(manifest_path), "window_library": str(OUTPUT_ROOT / "01_hierarchical_adaptive_windows.csv"), "cell_summary": str(cell_summary_path)},
        "output_conflict_action": output_info,
    }
    summary_path = OUTPUT_ROOT / "pts_v6_hierarchical_adaptive_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# PTS Hierarchical-Adaptive Generation Report", "",
        f"- Real Train：`{REAL_TRAIN_NPZ}`", f"- 目标地区：{', '.join(target_regions)}",
        "- 目标类别：class 1—7；class 0不人工拼接", f"- Core-A：**{core_a_n:,}**", f"- Core-B：**{core_b_n:,}**",
        f"- region-only majority accuracy：**{region_only_accuracy:.4f}**", f"- 7类随机参考：**{1/7:.4f}**",
        f"- DOY-template majority accuracy：**{doy_template_accuracy:.4f}**", "",
        "## v6核心升级", "",
        "1. source×class×season、region×class×season、region×season、global×class×season 和 global×season 五级自适应窗口库。",
        "2. 多季窗口优先使用真实谷值、斜率转折和曲率拐点分界；中点仅作最后回退。",
        "3. 背景只移除 base 原有真实季相，避免将全年所有潜在窗口机械削平。",
        "4. 区域风格适配使用低植被和非目标窗口背景，且只操作8个原始波段；NDVI/LSWI最终强制重算。",
        "5. Core-A逐样本同时通过物候、峰结构、区域背景、边界连续性、血缘和新颖度审查，不使用教师分类器。", "",
        "## 主要文件", "", f"- `{core_a_path.name}`", f"- `{core_b_path.name}`", f"- `{manifest_path.name}`",
        "- `01_hierarchical_adaptive_windows.csv`", "- `05_region_class_generation_summary.csv`", "- `08_core_a_region_class_coverage.csv`",
    ]
    (OUTPUT_ROOT / "generation_report.md").write_text("\n".join(report), encoding="utf-8")
    print("=" * 120)
    print("DONE")
    print("Core-A N              :", core_a_n)
    print("Core-B N              :", core_b_n)
    print("Region-only accuracy  :", region_only_accuracy)
    print("DOY-template accuracy :", doy_template_accuracy)
    print("Core-A NPZ            :", core_a_path)
    print("Window library        :", OUTPUT_ROOT / "01_hierarchical_adaptive_windows.csv")
    print("Summary               :", summary_path)
    print("=" * 120)


if __name__ == "__main__":
    main()
