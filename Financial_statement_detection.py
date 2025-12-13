import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

font_path = "NotoSansKR-Regular.ttf"
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    rcParams["font.family"] = "Noto Sans KR"
    rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="회계 이상 탐지 대시보드 · (Benford 제외)", layout="wide")

EPS = 1e-9


def reset_session_for_new_file(filename: str):
    st.session_state["uploaded_name"] = filename
    st.session_state["base_top_ids"] = None
    st.session_state["base_params"] = None


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


def _safe_ratio(num: pd.Series, den: pd.Series, eps: float = EPS) -> pd.Series:
    den2 = den.copy()
    small = den2.abs() < eps
    den2[small] = np.sign(den2[small]).replace(0, 1) * eps
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


def _percentile_rank(x: pd.Series) -> pd.Series:
    return x.rank(pct=True) * 100.0


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

    df["beneish_c_ar"] = z_ar
    df["beneish_c_inv"] = z_inv
    df["beneish_c_tata"] = z_tata
    df["beneish_c_ocfneg"] = -z_ocf

    m = df["mscore_raw"].fillna(0.0).values
    m_norm = (m - np.nanmin(m)) / (np.nanmax(m) - np.nanmin(m) + EPS)
    df["mscore_norm"] = m_norm

    iso_features = ["ar_to_sales", "inv_to_sales", "tata", "ocf_to_ni"]
    X = df[iso_features].fillna(0.0).values

    try:
        scaler = RobustScaler()
        Xs = scaler.fit_transform(X)

        iso = IsolationForest(contamination=contamination, random_state=42)
        iso.fit(Xs)

        iso_raw = -iso.decision_function(Xs)
        iso_raw = np.asarray(iso_raw)
        iso_norm = (iso_raw - iso_raw.min()) / (iso_raw.max() - iso_raw.min() + EPS)
    except Exception:
        iso_norm = np.zeros(df.shape[0])

    df["iso_score"] = iso_norm

    df["score_beneish_part"] = w_beneish * df["mscore_norm"]
    df["score_iso_part"] = w_iso * df["iso_score"]
    df["flag_score"] = df["score_beneish_part"] + df["score_iso_part"]

    df_scored = df.sort_values("flag_score", ascending=False).reset_index(drop=True)
    df_scored["rank"] = np.arange(1, len(df_scored) + 1)

    meta = {"group_mode": group_mode, "contamination": contamination, "w_beneish": w_beneish, "w_iso": w_iso}
    return df_scored, meta


def explain_top_row(row: pd.Series) -> dict:
    parts = {
        "Beneish(간이)": float(row.get("score_beneish_part", 0.0)),
        "IsolationForest": float(row.get("score_iso_part", 0.0)),
    }
    driver = max(parts, key=parts.get)

    beneish_sub = {
        "AR/Sales 과다": float(row.get("beneish_c_ar", 0.0)),
        "Inv/Sales 과다": float(row.get("beneish_c_inv", 0.0)),
        "TATA(발생액) 과다": float(row.get("beneish_c_tata", 0.0)),
        "OCF/NI 낮음": float(row.get("beneish_c_ocfneg", 0.0)),
    }
    top_beneish = sorted(beneish_sub.items(), key=lambda x: x[1], reverse=True)[:2]

    msg = (
        f"- 주도 요인: {driver}\n"
        f"- 점수 구성: Beneish {parts['Beneish(간이)']:.3f} + ISO {parts['IsolationForest']:.3f} = {float(row.get('flag_score', 0.0)):.3f}\n"
        f"- Beneish 내부 상위 기여(Top2): " + ", ".join([f"{k}({v:+.2f})" for k, v in top_beneish])
    )
    return {"driver": driver, "parts": parts, "msg": msg}


st.sidebar.header("옵션")

group_mode_ui = st.sidebar.radio("그룹 표준화 기준", ["연도", "연도+산업", "전체"])
group_mode_key = {"연도": "year", "연도+산업": "year_industry", "전체": "all"}[group_mode_ui]

contamination = st.sidebar.slider("탐지 민감도(ISO contamination)", 0.01, 0.30, 0.10, 0.01)
top_n = st.sidebar.slider("Top-N(의심 후보 수)", 3, 30, 10, 1)

w_beneish = st.sidebar.slider("Beneish(간이) 비중", 0.0, 3.0, 1.0, 0.1)
w_iso = st.sidebar.slider("Isolation Forest 비중", 0.0, 3.0, 1.0, 0.1)

min_pct = st.sidebar.slider("정상 제외 기준(상위 백분위 이상만 표시)", 0.50, 0.99, 0.80, 0.01)

st.title("회계 이상 탐지 대시보드 · 강화판 (Benford 제외)")

uploaded = st.file_uploader("CSV 또는 Excel 업로드", type=["csv", "xlsx"])
if uploaded is None:
    st.stop()

if "uploaded_name" not in st.session_state or st.session_state["uploaded_name"] != uploaded.name:
    reset_session_for_new_file(uploaded.name)

df_raw = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)

params = PipelineParams(group_mode=group_mode_key, contamination=contamination, w_beneish=w_beneish, w_iso=w_iso)

try:
    df_scored, _ = run_pipeline_cached(df_raw, params)
except Exception as e:
    st.error(f"처리 중 오류: {e}")
    st.stop()

cut = float(df_scored["flag_score"].quantile(min_pct))
df_view = df_scored[df_scored["flag_score"] >= cut].copy()
if df_view.empty:
    df_view = df_scored.copy()

df_top = df_view.head(top_n).copy()

tab1, tab2 = st.tabs(["🔍 Top-N & 자동 설명", "🌡️ 동종 그룹 열지도"])

with tab1:
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
    st.dataframe(df_top[show_cols], use_container_width=True, height=360)

    top_k = min(3, len(df_top))
    if top_k > 0:
        for i in range(top_k):
            r = df_top.iloc[i]
            exp = explain_top_row(r)
            title = f"#{int(r['rank'])} {r['company']} ({int(r['year'])})"
            with st.expander(title, expanded=(i == 0)):
                st.markdown(exp["msg"])
                p = exp["parts"]
                comp_df = pd.DataFrame({"component": list(p.keys()), "score_part": list(p.values())}).set_index("component")
                st.bar_chart(comp_df)

                snap_cols = ["flag_score", "mscore_raw", "iso_score", "ar_to_sales", "inv_to_sales", "tata", "ocf_to_ni"]
                snap_cols = [c for c in snap_cols if c in df_top.columns]
                st.dataframe(pd.DataFrame([r[snap_cols]]), use_container_width=True)

                g = df_scored[(df_scored["year"] == r["year"]) & (df_scored["industry"] == r["industry"])].copy()
                if len(g) >= 5:
                    cols = ["ar_to_sales", "inv_to_sales", "tata", "ocf_to_ni", "mscore_raw", "iso_score"]
                    cols = [c for c in cols if c in g.columns]
                    pct = {}
                    for c in cols:
                        pct_series = _percentile_rank(g[c].fillna(g[c].median()))
                        idx = g.index[g["row_id"] == r["row_id"]]
                        if len(idx) > 0:
                            pct[c] = float(pct_series.loc[idx[0]])
                    if pct:
                        pct_df = pd.DataFrame([pct]).T.reset_index()
                        pct_df.columns = ["metric", "percentile_in_peer(%)"]
                        st.dataframe(pct_df, use_container_width=True, height=240)

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

    k = st.slider("동종 그룹 크기 (기준 회사 포함)", 3, min(10, subset.shape[0]), min(5, subset.shape[0]))
    peer = subset.nsmallest(k, "peer_dist").copy()

    metrics = ["ar_to_sales", "inv_to_sales", "tata", "ocf_to_ni", "mscore_raw", "iso_score", "flag_score"]
    metrics = [m for m in metrics if m in peer.columns]

    if metrics:
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
        st.pyplot(fig)
