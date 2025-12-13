import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

font_path = "NotoSansKR-Regular.ttf"
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    rcParams["font.family"] = "Noto Sans KR"
    rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="회계 이상 스크리닝", layout="wide")

EPS = 1e-9


def _parse_number_series(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace("\u00a0", "", regex=False)
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(s, errors="coerce")


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    aliases = {
        "company": ["company", "회사명", "법인명"],
        "year": ["year", "결산연도", "연도"],
        "industry": ["industry", "업종", "산업"],
        "sales": ["sales", "매출액", "수익"],
        "ar": ["ar", "accounts_receivable", "매출채권"],
        "inventory": ["inventory", "재고자산"],
        "total_assets": ["total_assets", "자산총계", "총자산"],
        "ocf": ["ocf", "영업활동현금흐름", "영업현금흐름"],
        "net_income": ["net_income", "당기순이익"],
    }

    col_map = {}
    for canonical, cands in aliases.items():
        for c in cands:
            if c in df.columns:
                col_map[c] = canonical
                break
    df = df.rename(columns=col_map)

    required = ["company", "year", "sales", "ar", "inventory", "total_assets", "ocf", "net_income"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}\n현재 컬럼: {list(df.columns)}")

    if "industry" not in df.columns:
        df["industry"] = "미지정"

    return df


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    den2 = den.copy()
    small = den2.abs() < EPS
    den2[small] = np.sign(den2[small]).replace(0, 1) * EPS
    return num / den2


def _compute_metrics(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_columns(df_raw)

    for col in ["sales", "ar", "inventory", "total_assets", "ocf", "net_income"]:
        df[col] = _parse_number_series(df[col])

    df = df.reset_index(drop=True)
    df["row_id"] = df.index + 1
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    df["ar_to_sales"] = _safe_ratio(df["ar"], df["sales"] + EPS)
    df["inv_to_sales"] = _safe_ratio(df["inventory"], df["sales"] + EPS)
    df["ocf_to_ni"] = _safe_ratio(df["ocf"], df["net_income"])
    df["tata"] = _safe_ratio(df["net_income"] - df["ocf"], df["total_assets"] + EPS)

    df = df.sort_values(["company", "year"])
    df["sales_yoy"] = df.groupby("company")["sales"].pct_change().fillna(0.0) * 100.0
    return df


def _group_key_cols(group_mode: str):
    if group_mode == "year":
        return ["year"]
    if group_mode == "year_industry":
        return ["year", "industry"]
    return []


def _build_group_stats(df: pd.DataFrame, group_mode: str, cols: list[str]):
    keys = _group_key_cols(group_mode)

    overall = {}
    for c in cols:
        m = float(df[c].mean())
        s = float(df[c].std(ddof=0))
        overall[c] = (m, s if (s and not np.isnan(s)) else 0.0)

    if not keys:
        return {"__overall__": overall}, keys

    stats = {}
    grouped = df.groupby(keys, dropna=False)
    for k, g in grouped:
        local = {}
        for c in cols:
            m = float(g[c].mean())
            s = float(g[c].std(ddof=0))
            local[c] = (m, s if (s and not np.isnan(s)) else 0.0)
        stats[k] = local

    stats["__overall__"] = overall
    return stats, keys


def _get_stat(stats: dict, key_tuple, col: str):
    if key_tuple in stats:
        m, s = stats[key_tuple][col]
        return m, s
    m, s = stats["__overall__"][col]
    return m, s


def _percentile_from_sorted(sorted_ref: np.ndarray, values: np.ndarray):
    sorted_ref = np.asarray(sorted_ref)
    values = np.asarray(values)
    idx = np.searchsorted(sorted_ref, values, side="right")
    return idx / max(len(sorted_ref), 1)


@dataclass
class ReferenceModel:
    group_mode: str
    metrics: list
    stats: dict
    key_cols: list
    scaler: RobustScaler
    iso: IsolationForest
    ref_mscore_sorted: np.ndarray
    ref_iso_sorted: np.ndarray


@st.cache_data(show_spinner=False)
def fit_reference(df_ref_raw: pd.DataFrame, group_mode: str, contamination: float) -> ReferenceModel:
    df_ref = _compute_metrics(df_ref_raw)

    metrics = ["ar_to_sales", "inv_t_]()
