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
    w_beneish: float
    w_iso: float


@st.cache_data(show_spinner=False)
def run_pipeline_cached(df_raw: pd.DataFrame, params: PipelineParams):
    return run_pipeline(
        df_raw=df_raw,
        group_mode=params.group_mode,
        contamination=params.contamination,
        w_beneish=params.w_beneish,
        w_iso=params.w_iso,
    )


def run_pipeline(
    df_raw: pd.DataFrame,
    group_mode: str = "year_industry",
    contamination: float = 0.10,
    w_beneish: float = 1.0,
    w_iso: float = 1.0,
):
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

    metrics = ["ar_to_sales", "inv_to_sales", "tata", "ocf_to_ni"]

    if group_mode == "year":
        df = df.groupby("year", group_keys=False).apply(_zscore_group, cols=metrics)
    elif group_mode == "year_industry":
        df = df.groupby(["year", "industry"], group_keys=False).apply(_zscore_group, cols=metrics)
    else:
        df = _zscore_group(df, metrics)

    for c in ["ar_to_sales_z", "inv_to_sales_z", "tata_z", "ocf_to_ni_z"]:
        if c in df.columns:
            df[c] = df[c].clip(-5, 5)

    z_ar = df.get("ar_to_sales_z", pd.Series(0, index=df.index))
    z_inv = df.get("inv_to_sales_z", pd.Series(0, index=df.index))
    z_tata = df.get("tata_z", pd.Series(0, index=df.index))
    z_ocf = df.get("ocf_to_ni_z", pd.Series(0, index=df.index))

    df["mscore_raw"] = z_ar + z_inv + z_tata - z_ocf

    df["c_ar"] = z_ar
    df["c_inv"] = z_inv
    df["c_tata"] = z_tata
    df["c_ocfneg"] = -z_ocf

    df["mscore_norm"] = _pct_rank(df["mscore_raw"].fillna(df["mscore_raw"].median()))

    iso_features = ["ar_to_sales", "inv_to_sales", "tata", "ocf_to_ni"]
    X = df[iso_features].fillna(0.0).values

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

    df["score_beneish_part"] = w_beneish * df["mscore_norm"]
    df["score_iso_part"] = w_iso * df["iso_score"]
    df["flag_score"] = df["score_beneish_part"] + df["score_iso_part"]

    df_scored = df.sort_values("flag_score", ascending=False).reset_index(drop=True)
    df_scored["rank"] = np.arange(1, len(df_scored) + 1)
    return df_scored


def explain_row(row: pd.Series) -> str:
    parts = {
        "Beneish(간이)": float(row.get("score_beneish_part", 0.0)),
        "IsolationForest": float(row.get("score_iso_part", 0.0)),
    }
    driver = max(parts, key=parts.get)

    sub = {
        "AR/Sales": float(row.get("c_ar", 0.0)),
        "Inv/Sales": float(row.get("c_inv", 0.0)),
        "TATA": float(row.get("c_tata", 0.0)),
        "OCF/NI(낮음)": float(row.get("c_ocfneg", 0.0)),
    }
    top2 = sorted(sub.items(), key=lambda x: x[1], reverse=True)[:2]

    return (
        f"- 주도 요인: {driver}\n"
        f"- 점수: Beneish {parts['Beneish(간이)']:.3f} + ISO {parts['IsolationForest']:.3f} = {float(row.get('flag_score', 0.0)):.3f}\n"
        f"- Beneish 기여(Top2): " + ", ".join([f"{k}({v:+.2f})" for k, v in top2])
    )


st.sidebar.header("설정")

group_mode_ui = st.sidebar.radio("그룹 표준화 기준", ["연도", "연도+산업", "전체"])
group_mode_key = {"연도": "year", "연도+산업": "year_industry", "전체": "all"}[group_mode_ui]

contamination = st.sidebar.slider("ISO 민감도(contamination)", 0.01, 0.30, 0.10, 0.01)

w_beneish = st.sidebar.slider("Beneish 비중", 0.0, 3.0, 1.0, 0.1)
w_iso = st.sidebar.slider("ISO 비중", 0.0, 3.0, 1.0, 0.1)

st.sidebar.markdown("---")
rule = st.sidebar.radio("출력 규칙", ["OR(추천)", "AND(엄격)"])

allowed_k = st.sidebar.slider("허용 후보 수(K) (0이면 0건 가능)", 0, 30, 0, 1)
top_n = st.sidebar.slider("표시 Top-N", 1, 30, 10, 1)

st.sidebar.markdown("---")
manual = st.sidebar.toggle("수동 임계값 사용", value=False)
if manual:
    thr_score = st.sidebar.slider("flag_score 최소", 0.0, 6.0, 2.50, 0.05)
    thr_b = st.sidebar.slider("Beneish(백분위) 최소", 0.0, 1.0, 0.98, 0.01)
    thr_i = st.sidebar.slider("ISO(백분위) 최소", 0.0, 1.0, 0.98, 0.01)

st.title("회계 이상 스크리닝")

uploaded = st.file_uploader("CSV 또는 Excel 업로드", type=["csv", "xlsx"])
if uploaded is None:
    st.stop()

df_raw = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)

params = PipelineParams(group_mode=group_mode_key, contamination=contamination, w_beneish=w_beneish, w_iso=w_iso)

try:
    df_scored = run_pipeline_cached(df_raw, params)
except Exception as e:
    st.error(f"처리 중 오류: {e}")
    st.stop()

if not manual:
    if allowed_k == 0:
        thr_score = float(df_scored["flag_score"].max()) + 1e-9
    else:
        thr_score = float(df_scored["flag_score"].nlargest(int(allowed_k)).min())

    thr_b = float(df_scored["mscore_norm"].quantile(0.95))
    thr_i = float(df_scored["iso_score"].quantile(0.95))

base = df_scored["flag_score"] >= thr_score
cond_b = df_scored["mscore_norm"] >= thr_b
cond_i = df_scored["iso_score"] >= thr_i

if rule.startswith("OR"):
    mask = base & (cond_b | cond_i)
else:
    mask = base & cond_b & cond_i

df_view = df_scored[mask].copy()

st.caption(
    f"통과: {df_view.shape[0]} / {df_scored.shape[0]} | "
    f"컷오프: flag_score≥{thr_score:.4f}, Beneish≥p95({thr_b:.2f}), ISO≥p95({thr_i:.2f}) | 규칙={rule.split('(')[0]} | K={allowed_k}"
)

tab1, tab2 = st.tabs(["🔍 후보 리스트 & Top3 이유", "🌡️ 동종 그룹 열지도"])

with tab1:
    if df_view.empty:
        st.info("현재 기준에서는 추가 점검 후보가 없습니다.")
        st.stop()

    df_top = df_view.head(top_n).copy()

    show_cols = [
        "rank",
        "company",
        "year",
        "industry",
        "flag_score",
        "score_beneish_part",
        "score_iso_part",
        "mscore_raw",
        "mscore_norm",
        "iso_score",
        "ar_to_sales",
        "inv_to_sales",
        "tata",
        "ocf_to_ni",
    ]
    show_cols = [c for c in show_cols if c in df_top.columns]
    st.dataframe(df_top[show_cols], use_container_width=True, height=380)

    top_k = min(3, len(df_top))
    for i in range(top_k):
        r = df_top.iloc[i]
        with st.expander(f"#{int(r['rank'])} {r['company']} ({int(r['year'])})", expanded=(i == 0)):
            st.markdown(explain_row(r))
            comp_df = pd.DataFrame(
                {"component": ["Beneish(간이)", "IsolationForest"], "score_part": [r["score_beneish_part"], r["score_iso_part"]]}
            ).set_index("component")
            st.bar_chart(comp_df)

with tab2:
    years = sorted(df_scored["year"].dropna().unique())
    industries = sorted(df_scored["industry"].dropna().unique())

    if not years or not industries:
        st.info("열지도를 만들 데이터가 부족합니다.")
        st.stop()

    sel_year = st.selectbox("연도 선택", years, key="peer_year")
    sel_ind = st.selectbox("산업 선택", industries, key="peer_ind")

    subset = df_scored[(df_scored["year"] == sel_year) & (df_scored["industry"] == sel_ind)].copy()
    if subset.empty:
        st.info("해당 연도·산업 조합에 데이터가 없습니다.")
        st.stop()

    companies = subset["company"].unique().tolist()
    sel_comp = st.selectbox("기준 회사 선택", companies, key="peer_comp")

    subset["size_metric"] = np.log1p(subset["total_assets"].fillna(0.0))
    subset["growth_metric"] = subset["sales_yoy"].fillna(0.0)
    subset["profit_metric"] = (subset["net_income"] / (subset["sales"] + EPS)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for c in ["size_metric", "growth_metric", "profit_metric"]:
        m = subset[c].mean()
        s = subset[c].std(ddof=0) or EPS
        subset[c + "_z"] = (subset[c] - m) / s

    focus = subset[subset["company"] == sel_comp].copy()
    if focus.empty:
        st.info("선택한 회사 데이터가 없습니다.")
        st.stop()

    focus_row = focus.iloc[0]
    f_vec = np.array([float(focus_row["size_metric_z"]), float(focus_row["growth_metric_z"]), float(focus_row["profit_metric_z"])])

    subset["peer_dist"] = subset.apply(
        lambda r: np.linalg.norm(np.array([r["size_metric_z"], r["growth_metric_z"], r["profit_metric_z"]]) - f_vec),
        axis=1,
    )

    k = st.slider("동종 그룹 크기", 3, min(10, subset.shape[0]), min(5, subset.shape[0]))
    peer = subset.nsmallest(k, "peer_dist").copy()

    metrics = ["ar_to_sales", "inv_to_sales", "tata", "ocf_to_ni", "mscore_raw", "iso_score", "flag_score"]
    metrics = [m for m in metrics if m in peer.columns]

    if not metrics:
        st.info("열지도로 보여줄 지표가 없습니다.")
        st.stop()

    peer_z = peer.copy()
    for m in metrics:
        mm = peer[m].mean()
        ss = peer[m].std(ddof=0) or EPS
        peer_z[m + "_z_peer"] = (peer[m] - mm) / ss

    z_cols = [m + "_z_peer" for m in metrics]
    z_vals = peer_z[z_cols].values
    labels = [f"{r['company']}_{int(r['year'])}" for _, r in peer.iterrows()]

    fig, ax = plt.subplots(figsize=(1.2 * len(metrics), 0.55 * len(peer) + 1))
    im = ax.imshow(z_vals, aspect="auto", cmap="coolwarm")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("동종 그룹 내 지표 편차 (z-score)")
    st.pyplot(fig)
