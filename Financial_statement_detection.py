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

    metrics = ["ar_to_sales", "inv_to_sales", "tata", "ocf_to_ni"]
    stats, key_cols = _build_group_stats(df_ref, group_mode, metrics)

    z_ar = np.zeros(len(df_ref))
    z_inv = np.zeros(len(df_ref))
    z_tata = np.zeros(len(df_ref))
    z_ocf = np.zeros(len(df_ref))

    if key_cols:
        key_series = df_ref[key_cols].apply(lambda r: tuple(r.values.tolist()), axis=1)
    else:
        key_series = pd.Series([None] * len(df_ref), index=df_ref.index)

    for i, (idx, row) in enumerate(df_ref.iterrows()):
        k = key_series.loc[idx] if key_cols else None
        m, s = _get_stat(stats, k, "ar_to_sales")
        z_ar[i] = 0.0 if s == 0 else (row["ar_to_sales"] - m) / s
        m, s = _get_stat(stats, k, "inv_to_sales")
        z_inv[i] = 0.0 if s == 0 else (row["inv_to_sales"] - m) / s
        m, s = _get_stat(stats, k, "tata")
        z_tata[i] = 0.0 if s == 0 else (row["tata"] - m) / s
        m, s = _get_stat(stats, k, "ocf_to_ni")
        z_ocf[i] = 0.0 if s == 0 else (row["ocf_to_ni"] - m) / s

    z_ar = np.clip(z_ar, -5, 5)
    z_inv = np.clip(z_inv, -5, 5)
    z_tata = np.clip(z_tata, -5, 5)
    z_ocf = np.clip(z_ocf, -5, 5)

    mscore_ref = z_ar + z_inv + z_tata - z_ocf
    ref_mscore_sorted = np.sort(mscore_ref.astype(float))

    X_ref = df_ref[metrics].fillna(0.0).values
    scaler = RobustScaler()
    Xs_ref = scaler.fit_transform(X_ref)

    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(Xs_ref)
    iso_raw_ref = -iso.decision_function(Xs_ref)
    ref_iso_sorted = np.sort(np.asarray(iso_raw_ref).astype(float))

    return ReferenceModel(
        group_mode=group_mode,
        metrics=metrics,
        stats=stats,
        key_cols=key_cols,
        scaler=scaler,
        iso=iso,
        ref_mscore_sorted=ref_mscore_sorted,
        ref_iso_sorted=ref_iso_sorted,
    )


def score_with_reference(df_raw: pd.DataFrame, ref: ReferenceModel, w_beneish: float, w_iso: float) -> pd.DataFrame:
    df = _compute_metrics(df_raw)

    if ref.key_cols:
        key_series = df[ref.key_cols].apply(lambda r: tuple(r.values.tolist()), axis=1)
    else:
        key_series = pd.Series([None] * len(df), index=df.index)

    z_ar = np.zeros(len(df))
    z_inv = np.zeros(len(df))
    z_tata = np.zeros(len(df))
    z_ocf = np.zeros(len(df))

    for i, (idx, row) in enumerate(df.iterrows()):
        k = key_series.loc[idx] if ref.key_cols else None

        m, s = _get_stat(ref.stats, k, "ar_to_sales")
        z_ar[i] = 0.0 if s == 0 else (row["ar_to_sales"] - m) / s

        m, s = _get_stat(ref.stats, k, "inv_to_sales")
        z_inv[i] = 0.0 if s == 0 else (row["inv_to_sales"] - m) / s

        m, s = _get_stat(ref.stats, k, "tata")
        z_tata[i] = 0.0 if s == 0 else (row["tata"] - m) / s

        m, s = _get_stat(ref.stats, k, "ocf_to_ni")
        z_ocf[i] = 0.0 if s == 0 else (row["ocf_to_ni"] - m) / s

    z_ar = np.clip(z_ar, -5, 5)
    z_inv = np.clip(z_inv, -5, 5)
    z_tata = np.clip(z_tata, -5, 5)
    z_ocf = np.clip(z_ocf, -5, 5)

    mscore_raw = z_ar + z_inv + z_tata - z_ocf
    mscore_norm = _percentile_from_sorted(ref.ref_mscore_sorted, mscore_raw)

    X = df[ref.metrics].fillna(0.0).values
    Xs = ref.scaler.transform(X)
    iso_raw = -ref.iso.decision_function(Xs)
    iso_score = _percentile_from_sorted(ref.ref_iso_sorted, iso_raw)

    df["ar_to_sales_z"] = z_ar
    df["inv_to_sales_z"] = z_inv
    df["tata_z"] = z_tata
    df["ocf_to_ni_z"] = z_ocf

    df["mscore_raw"] = mscore_raw
    df["mscore_norm"] = mscore_norm
    df["iso_score"] = iso_score

    df["score_beneish_part"] = w_beneish * df["mscore_norm"]
    df["score_iso_part"] = w_iso * df["iso_score"]
    df["flag_score"] = df["score_beneish_part"] + df["score_iso_part"]

    df = df.sort_values("flag_score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def explain_top_row(row: pd.Series) -> str:
    parts = {
        "Beneish(간이)": float(row.get("score_beneish_part", 0.0)),
        "IsolationForest": float(row.get("score_iso_part", 0.0)),
    }
    driver = max(parts, key=parts.get)

    sub = {
        "AR/Sales(z)": float(row.get("ar_to_sales_z", 0.0)),
        "Inv/Sales(z)": float(row.get("inv_to_sales_z", 0.0)),
        "TATA(z)": float(row.get("tata_z", 0.0)),
        "OCF/NI(z)": float(row.get("ocf_to_ni_z", 0.0)),
    }
    top2 = sorted(sub.items(), key=lambda x: abs(x[1]), reverse=True)[:2]

    return (
        f"- 주도 요인: {driver}\n"
        f"- 점수: Beneish {parts['Beneish(간이)']:.3f} + ISO {parts['IsolationForest']:.3f} = {float(row.get('flag_score', 0.0)):.3f}\n"
        f"- 동종 대비 편차(절대값 Top2): " + ", ".join([f"{k}({v:+.2f})" for k, v in top2])
    )


st.sidebar.header("설정")

group_mode_ui = st.sidebar.radio("그룹 기준", ["연도", "연도+산업", "전체"])
group_mode_key = {"연도": "year", "연도+산업": "year_industry", "전체": "all"}[group_mode_ui]

contamination = st.sidebar.slider("ISO 민감도(contamination)", 0.01, 0.30, 0.10, 0.01)
w_beneish = st.sidebar.slider("Beneish 비중", 0.0, 3.0, 1.0, 0.1)
w_iso = st.sidebar.slider("ISO 비중", 0.0, 3.0, 1.0, 0.1)

st.sidebar.markdown("---")
rule = st.sidebar.radio("출력 규칙", ["OR(추천)", "AND(엄격)"])
allowed_fp = st.sidebar.number_input("정상(기준)에서 허용 후보 수", min_value=0, max_value=30, value=0, step=1)
top_n = st.sidebar.slider("표시 Top-N", 1, 30, 10, 1)

st.title("회계 이상 스크리닝 (기준고정·0건 가능)")

colA, colB = st.columns(2)
with colA:
    ref_file = st.file_uploader("기준(정상, 회계사 검증) 파일 업로드 (권장)", type=["csv", "xlsx"], key="ref")
with colB:
    test_file = st.file_uploader("분석할 파일 업로드", type=["csv", "xlsx"], key="test")

if test_file is None:
    st.stop()

def _read_file(f):
    return pd.read_csv(f) if f.name.lower().endswith(".csv") else pd.read_excel(f)

df_test_raw = _read_file(test_file)

use_ref = ref_file is not None
if use_ref:
    df_ref_raw = _read_file(ref_file)
    try:
        ref_model = fit_reference(df_ref_raw, group_mode_key, contamination)
    except Exception as e:
        st.error(f"기준 파일 처리 오류: {e}")
        st.stop()

    try:
        df_ref_scored = score_with_reference(df_ref_raw, ref_model, w_beneish, w_iso)
        df_scored = score_with_reference(df_test_raw, ref_model, w_beneish, w_iso)
    except Exception as e:
        st.error(f"스코어링 오류: {e}")
        st.stop()

    if allowed_fp == 0:
        thr_score = float(df_ref_scored["flag_score"].max()) + 1e-9
    else:
        thr_score = float(df_ref_scored["flag_score"].nlargest(int(allowed_fp)).min())

    thr_b = float(df_ref_scored["mscore_norm"].quantile(0.95))
    thr_i = float(df_ref_scored["iso_score"].quantile(0.95))

    base = df_scored["flag_score"] >= thr_score
    cond_b = df_scored["mscore_norm"] >= thr_b
    cond_i = df_scored["iso_score"] >= thr_i

    if rule.startswith("OR"):
        mask = base & (cond_b | cond_i)
    else:
        mask = base & cond_b & cond_i

    df_view = df_scored[mask].copy()

    ref_base = df_ref_scored["flag_score"] >= thr_score
    ref_b = df_ref_scored["mscore_norm"] >= thr_b
    ref_i = df_ref_scored["iso_score"] >= thr_i
    if rule.startswith("OR"):
        ref_mask = ref_base & (ref_b | ref_i)
    else:
        ref_mask = ref_base & ref_b & ref_i

    st.caption(
        f"기준고정 모드 | 기준 통과: {int(ref_mask.sum())} / {len(df_ref_scored)} | "
        f"분석 통과: {int(mask.sum())} / {len(df_scored)} | "
        f"컷오프: flag_score>{thr_score:.4f} (정상허용={allowed_fp}), "
        f"B≥p95({thr_b:.3f}), ISO≥p95({thr_i:.3f}), 규칙={rule.split('(')[0]}"
    )
else:
    st.warning("기준 파일이 없어서 파일 내부 기준으로만 계산합니다. (발표용으론 기준 파일 업로드 권장)")
    try:
        df_scored = _compute_metrics(df_test_raw)
    except Exception as e:
        st.error(f"전처리 오류: {e}")
        st.stop()

    metrics = ["ar_to_sales", "inv_to_sales", "tata", "ocf_to_ni"]
    stats, key_cols = _build_group_stats(df_scored, group_mode_key, metrics)

    if key_cols:
        key_series = df_scored[key_cols].apply(lambda r: tuple(r.values.tolist()), axis=1)
    else:
        key_series = pd.Series([None] * len(df_scored), index=df_scored.index)

    z_ar = np.zeros(len(df_scored))
    z_inv = np.zeros(len(df_scored))
    z_tata = np.zeros(len(df_scored))
    z_ocf = np.zeros(len(df_scored))

    for i, (idx, row) in enumerate(df_scored.iterrows()):
        k = key_series.loc[idx] if key_cols else None
        m, s = _get_stat(stats, k, "ar_to_sales"); z_ar[i] = 0.0 if s == 0 else (row["ar_to_sales"] - m) / s
        m, s = _get_stat(stats, k, "inv_to_sales"); z_inv[i] = 0.0 if s == 0 else (row["inv_to_sales"] - m) / s
        m, s = _get_stat(stats, k, "tata"); z_tata[i] = 0.0 if s == 0 else (row["tata"] - m) / s
        m, s = _get_stat(stats, k, "ocf_to_ni"); z_ocf[i] = 0.0 if s == 0 else (row["ocf_to_ni"] - m) / s

    z_ar = np.clip(z_ar, -5, 5)
    z_inv = np.clip(z_inv, -5, 5)
    z_tata = np.clip(z_tata, -5, 5)
    z_ocf = np.clip(z_ocf, -5, 5)

    mscore_raw = z_ar + z_inv + z_tata - z_ocf
    m_norm = (mscore_raw - np.nanmin(mscore_raw)) / (np.nanmax(mscore_raw) - np.nanmin(mscore_raw) + EPS)

    scaler = RobustScaler()
    Xs = scaler.fit_transform(df_scored[metrics].fillna(0.0).values)
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(Xs)
    iso_raw = -iso.decision_function(Xs)
    iso_norm = (iso_raw - np.min(iso_raw)) / (np.max(iso_raw) - np.min(iso_raw) + EPS)

    df_scored["ar_to_sales_z"] = z_ar
    df_scored["inv_to_sales_z"] = z_inv
    df_scored["tata_z"] = z_tata
    df_scored["ocf_to_ni_z"] = z_ocf
    df_scored["mscore_raw"] = mscore_raw
    df_scored["mscore_norm"] = m_norm
    df_scored["iso_score"] = iso_norm
    df_scored["score_beneish_part"] = w_beneish * df_scored["mscore_norm"]
    df_scored["score_iso_part"] = w_iso * df_scored["iso_score"]
    df_scored["flag_score"] = df_scored["score_beneish_part"] + df_scored["score_iso_part"]
    df_scored = df_scored.sort_values("flag_score", ascending=False).reset_index(drop=True)
    df_scored["rank"] = np.arange(1, len(df_scored) + 1)

    thr_score = st.sidebar.slider("절대 컷오프(flag_score)", 0.0, 2.0, 0.90, 0.05)
    thr_b = st.sidebar.slider("Beneish 최소(mscore_norm)", 0.0, 1.0, 0.80, 0.05)
    thr_i = st.sidebar.slider("ISO 최소(iso_score)", 0.0, 1.0, 0.60, 0.05)

    base = df_scored["flag_score"] >= thr_score
    cond_b = df_scored["mscore_norm"] >= thr_b
    cond_i = df_scored["iso_score"] >= thr_i

    if rule.startswith("OR"):
        mask = base & (cond_b | cond_i)
    else:
        mask = base & cond_b & cond_i

    df_view = df_scored[mask].copy()
    st.caption(
        f"내부기준 모드 | 통과: {int(mask.sum())} / {len(df_scored)} | "
        f"flag_score≥{thr_score:.2f}, Beneish≥{thr_b:.2f}, ISO≥{thr_i:.2f}, 규칙={rule.split('(')[0]}"
    )

if df_view.empty:
    st.info("현재 기준에서는 ‘추가 점검 후보’가 없습니다.")
    st.stop()

df_top = df_view.head(top_n).copy()

tab1, tab2 = st.tabs(["🔍 후보 리스트 & Top3 이유", "📊 점수 분포"])

with tab1:
    show_cols = [
        "rank", "company", "year", "industry",
        "flag_score", "score_beneish_part", "score_iso_part",
        "mscore_raw", "mscore_norm", "iso_score",
        "ar_to_sales", "inv_to_sales", "tata", "ocf_to_ni"
    ]
    show_cols = [c for c in show_cols if c in df_top.columns]
    st.dataframe(df_top[show_cols], use_container_width=True, height=380)

    top_k = min(3, len(df_top))
    for i in range(top_k):
        r = df_top.iloc[i]
        with st.expander(f"#{int(r['rank'])} {r['company']} ({int(r['year'])})", expanded=(i == 0)):
            st.markdown(explain_top_row(r))
            comp_df = pd.DataFrame(
                {"component": ["Beneish(간이)", "IsolationForest"], "score_part": [r["score_beneish_part"], r["score_iso_part"]]}
            ).set_index("component")
            st.bar_chart(comp_df)

with tab2:
    fig1, ax1 = plt.subplots()
    ax1.hist(df_scored["flag_score"].values, bins=20)
    ax1.set_title("flag_score 분포")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.hist(df_scored["mscore_norm"].values, bins=20)
    ax2.set_title("Beneish(간이) 백분위(mscore_norm) 분포")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    ax3.hist(df_scored["iso_score"].values, bins=20)
    ax3.set_title("ISO 백분위(iso_score) 분포")
    st.pyplot(fig3)
