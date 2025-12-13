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
        "cogs": ["cogs", "매출원가", "원가"],
        "sga": ["sga", "판매관리비", "판관비"],
        "op_income": ["op_income", "영업이익", "영업손익"],
        "dep": ["dep", "감가상각비", "감가상각"],
        "ar": ["ar", "accounts_receivable", "매출채권"],
        "inventory": ["inventory", "재고자산"],
        "total_assets": ["total_assets", "자산총계", "총자산"],
        "total_liab": ["total_liab", "부채총계", "총부채"],
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

    optional_defaults = {
        "cogs": np.nan,
        "sga": np.nan,
        "op_income": np.nan,
        "dep": np.nan,
        "total_liab": np.nan,
    }
    for k, v in optional_defaults.items():
        if k not in df.columns:
            df[k] = v

    return df


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    den2 = den.copy()
    small = den2.abs() < EPS
    den2[small] = np.sign(den2[small]).replace(0, 1) * EPS
    return num / den2


def _zscore_group(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = df.copy()
    for c in cols:
        m = g[c].mean()
        s = g[c].std(ddof=0)
        if s is None or s == 0 or np.isnan(s):
            g[c + "_z"] = 0.0
        else:
            g[c + "_z"] = (g[c] - m) / s
    return g


def _pct_rank(x: pd.Series) -> pd.Series:
    return x.rank(pct=True)


@dataclass
class PipelineParams:
    group_mode: str
    contamination: float
    w_linear: float
    w_iso: float


@st.cache_data(show_spinner=False)
def run_pipeline_cached(df_raw: pd.DataFrame, params: PipelineParams):
    return run_pipeline(
        df_raw=df_raw,
        group_mode=params.group_mode,
        contamination=params.contamination,
        w_linear=params.w_linear,
        w_iso=params.w_iso,
    )


def run_pipeline(
    df_raw: pd.DataFrame,
    group_mode: str = "year_industry",
    contamination: float = 0.10,
    w_linear: float = 1.0,
    w_iso: float = 1.0,
):
    df = _ensure_columns(df_raw)

    num_cols = [
        "sales", "cogs", "sga", "op_income", "dep",
        "ar", "inventory", "total_assets", "total_liab",
        "ocf", "net_income",
    ]
    for col in num_cols:
        df[col] = _parse_number_series(df[col])

    df = df.reset_index(drop=True)
    df["row_id"] = df.index + 1
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    df["ar_to_sales"] = _safe_ratio(df["ar"], df["sales"] + EPS)
    df["inv_to_sales"] = _safe_ratio(df["inventory"], df["sales"] + EPS)
    df["ocf_to_ni"] = _safe_ratio(df["ocf"], df["net_income"])
    df["tata"] = _safe_ratio(df["net_income"] - df["ocf"], df["total_assets"] + EPS)

    df["cogs_to_sales"] = _safe_ratio(df["cogs"], df["sales"] + EPS)
    df["sga_to_sales"] = _safe_ratio(df["sga"], df["sales"] + EPS)
    df["opm"] = _safe_ratio(df["op_income"], df["sales"] + EPS)
    df["dep_to_assets"] = _safe_ratio(df["dep"], df["total_assets"] + EPS)
    df["liab_to_assets"] = _safe_ratio(df["total_liab"], df["total_assets"] + EPS)

    df = df.sort_values(["company", "year"])
    df["sales_yoy"] = df.groupby("company")["sales"].pct_change().fillna(0.0) * 100.0

    metrics = [
        "ar_to_sales",
        "inv_to_sales",
        "tata",
        "ocf_to_ni",
        "cogs_to_sales",
        "sga_to_sales",
        "opm",
        "dep_to_assets",
        "liab_to_assets",
    ]

    for c in metrics:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    if group_mode == "year":
        df = df.groupby("year", group_keys=False).apply(_zscore_group, cols=metrics)
    elif group_mode == "year_industry":
        df = df.groupby(["year", "industry"], group_keys=False).apply(_zscore_group, cols=metrics)
    else:
        df = _zscore_group(df, metrics)

    z_cols = [m + "_z" for m in metrics]
    for c in z_cols:
        if c in df.columns:
            df[c] = df[c].clip(-5, 5)

    z = {m: df.get(m + "_z", pd.Series(0, index=df.index)).fillna(0.0) for m in metrics}

    df["linear_raw"] = (
        z["ar_to_sales"]
        + z["inv_to_sales"]
        + z["tata"]
        - z["ocf_to_ni"]
        + z["cogs_to_sales"]
        + z["sga_to_sales"]
        - z["opm"]
        + z["dep_to_assets"]
        + z["liab_to_assets"]
    )

    df["linear_norm"] = _pct_rank(df["linear_raw"].fillna(df["linear_raw"].median()))

    X = df[metrics].fillna(0.0).values
    try:
        scaler = RobustScaler()
        Xs = scaler.fit_transform(X)
        iso = IsolationForest(contamination=contamination, random_state=42)
        iso.fit(Xs)
        iso_raw = -iso.decision_function(Xs)
        df["iso_raw"] = iso_raw
        df["iso_score"] = _pct_rank(pd.Series(iso_raw)).values
    except Exception:
        df["iso_raw"] = 0.0
        df["iso_score"] = 0.0

    df["score_linear_part"] = w_linear * df["linear_norm"]
    df["score_iso_part"] = w_iso * df["iso_score"]
    df["flag_score"] = df["score_linear_part"] + df["score_iso_part"]

    z_mat = np.column_stack([z[m].values for m in metrics])
    abs_mat = np.abs(z_mat)
    idx = np.argsort(-abs_mat, axis=1)

    names = np.array(metrics, dtype=object)
    r = np.arange(len(df))

    df["_top1_metric"] = names[idx[:, 0]]
    df["_top1_z"] = z_mat[r, idx[:, 0]]
    df["_top2_metric"] = names[idx[:, 1]()]()
