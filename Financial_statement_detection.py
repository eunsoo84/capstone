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

    iso_features = metrics
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

    df["score_linear_part"] = w_linear * df["linear_norm"]
    df["score_iso_part"] = w_iso * df["iso_score"]
    df["flag_score"] = df["score_linear_part"] + df["score_iso_part"]

    df_scored = df.sort_values("flag_score", ascending=False).reset_index(drop=True)
    df_scored["rank"] = np.arange(1, len(df_scored) + 1)

    contrib = pd.DataFrame({m: z[m].values for m in metrics})
    df_scored["_top1_metric"] = contrib.abs().idxmax(axis=1)
    df_scored["_top1_z"] = contrib.lookup(contrib.index, df_scored["_top1_metric"])
    contrib2 = contrib.copy()
    for i in range(len(df_scored)):
        contrib2.loc[i, df_scored.loc[i, "_top1_metric"]] = 0.0
    df_scored["_top2_metric"] = contrib2.abs().idxmax(axis=1)
    df_scored["_top2_z"] = contrib2.lookup(contrib2.index, df_scored["_top2_metric"])
    contrib3 = contrib2.copy()
    for i in range(len(df_scored)):
        contrib3.loc[i, df_scored.loc[i, "_top2_metric"]] = 0.0
    df_scored["_top3_metric"] = contrib3.abs().idxmax(axis=1)
    df_scored["_top3_z"] = contrib3.lookup(contrib3.index, df_scored["_top3_metric"])

    return df_scored


def explain_row(row: pd.Series) -> str:
    parts = {
        "Linear(지표합)": float(row.get("score_linear_part", 0.0)),
        "IsolationForest": float(row.get("score_iso_part", 0.0)),
    }
    driver = max(parts, key=parts.get)

    t1m, t1z = str(row.get("_top1_metric", "")), float(row.get("_top1_z", 0.0))
    t2m, t2z = str(row.get("_top2_metric", "")), float(row.get("_top2_z", 0.0))
    t3m, t3z = str(row.get("_top3_metric", "")), float(row.get("_top3_z", 0.0))

    return (
        f"- 주도 요인: {driver}\n"
        f"- 점수: Linear {parts['Linear(지표합)']:.3f} + ISO {parts['IsolationForest']:.3f} = {float(row.get('flag_score', 0.0)):.3f}\n"
        f"- 동종 대비 편차 Top3: {t1m}({t1z:+.2f}), {t2m}({t2z:+.2f}), {t3m}({t3z:+.2f})"
    )


st.sidebar.header("설정")

group_mode_ui = st.sidebar.radio("그룹 표준화 기준", ["연도", "연도+산업", "전체"])
group_mode_key = {"연도": "year", "연도+산업": "year_industry", "전체": "all"}[group_mode_ui]

contamination = st.sidebar.slider("ISO 민감도(contamination)", 0.01, 0.30, 0.10, 0.01)

w_linear = st.sidebar.slider("지표합(Linear) 비중", 0.0, 3.0, 1.0, 0.1)
w_iso = st.sidebar.slider("ISO 비중", 0.0, 3.0, 1.0, 0.1)

st.sidebar.markdown("---")
rule = st.sidebar.radio("출력 규칙", ["OR(추천)", "AND(엄격)"])
allowed_k = st.sidebar.slider("허용 후보 수(K) (0이면 0건 가능)", 0, 30, 0, 1)
top_n = st.sidebar.slider("표시 Top-N", 1, 30, 10, 1)

st.sidebar.markdown("---")
p_cut = st.sidebar.slider("Beneish/ISO 퍼센타일 컷", 0.80, 0.99, 0.95, 0.01)

st.sidebar.markdown("---")
demo_only = st.sidebar.toggle("시연 회사만 보기", value=False)

st.title("회계 이상 스크리닝")

uploaded = st.file_uploader("CSV 또는 Excel 업로드", type=["csv", "xlsx"])
if uploaded is None:
    st.stop()

df_raw = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)

params = PipelineParams(group_mode=group_mode_key, contamination=contamination, w_linear=w_linear, w_iso=w_iso)

try:
    df_scored = run_pipeline_cached(df_raw, params)
except Exception as e:
    st.error(f"처리 중 오류: {e}")
    st.stop()

if allowed_k == 0:
    thr_score = float(df_scored["flag_score"].max()) + 1e-9
else:
    thr_score = float(df_scored["flag_score"].nlargest(int(allowed_k)).min())

thr_lin = float(df_scored["linear_norm"].quantile(p_cut))
thr_i = float(df_scored["iso_score"].quantile(p_cut))

base = df_scored["flag_score"] >= thr_score
cond_l = df_scored["linear_norm"] >= thr_lin
cond_i = df_scored["iso_score"] >= thr_i

if rule.startswith("OR"):
    mask = base & (cond_l | cond_i)
else:
    mask = base & cond_l & cond_i

df_view = df_scored[mask].copy()

if demo_only:
    all_companies = sorted(df_scored["company"].astype(str).unique().tolist())
    default_demo = [c for c in ["이마트", "대웅제약", "KG이니시스", "기아"] if c in all_companies]
    demo_companies = st.multiselect("시연 회사 선택(표시만 제한)", all_companies, default=default_demo)
    if demo_companies:
        df_view = df_view[df_view["company"].astype(str).isin(demo_companies)].copy()

st.caption(
    f"통과: {df_view.shape[0]} / {df_scored.shape[0]} | "
    f"컷오프: flag_score≥{thr_score:.4f}, Linear≥p{int(p_cut*100)}({thr_lin:.2f}), ISO≥p{int(p_cut*100)}({thr_i:.2f}) | 규칙={rule.split('(')[0]} | K={allowed_k}"
)

tab1, tab2 = st.tabs(["🔍 후보 리스트 & Top3 이유", "🌡️ 동종 그룹 열지도"])

with tab1:
    if df_view.empty:
        st.info("현재 기준에서는 추가 점검 후보가 없습니다.")
    else:
        df_top = df_view.head(top_n).copy()

        show_cols = [
            "rank",
            "company",
            "year",
            "industry",
            "flag_score",
            "score_linear_part",
            "score_iso_part",
            "linear_raw",
            "linear_norm",
            "iso_score",
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
        show_cols = [c for c in show_cols if c in df_top.columns]
        st.dataframe(df_top[show_cols], use_container_width=True, height=380)

        top_k = min(3, len(df_top))
        for i in range(top_k):
            r = df_top.iloc[i]
            with st.expander(f"#{int(r['rank'])} {r['company']} ({int(r['year'])})", expanded=(i == 0)):
                st.markdown(explain_row(r))
                comp_df = pd.DataFrame(
                    {"component": ["Linear(지표합)", "IsolationForest"], "score_part": [r["score_linear_part"], r["score_iso_part"]]}
                ).set_index("component")
                st.bar_chart(comp_df)

    st.markdown("---")
    st.subheader("회사/연도 조회(통과 여부 무관)")

    q_company = st.text_input("회사명(부분검색)", value="이마트")
    q_year = st.text_input("연도(선택)", value="2024")

    hit = df_scored[df_scored["company"].astype(str).str.contains(q_company, case=False, na=False)].copy()
    if q_year.strip():
        hit = hit[hit["year"].astype(str) == q_year.strip()].copy()

    if hit.empty:
        st.info("검색 결과가 없습니다.")
    else:
        hit = hit.sort_values("rank").head(30).copy()
        hit["gap_flag_score"] = hit["flag_score"] - thr_score
        hit["gap_linear"] = hit["linear_norm"] - thr_lin
        hit["gap_iso"] = hit["iso_score"] - thr_i

        cols2 = [
            "rank","company","year","industry",
            "flag_score","linear_norm","iso_score",
            "gap_flag_score","gap_linear","gap_iso",
            "ar_to_sales","inv_to_sales","tata","ocf_to_ni",
            "cogs_to_sales","sga_to_sales","opm","dep_to_assets","liab_to_assets",
            "_top1_metric","_top1_z","_top2_metric","_top2_z","_top3_metric","_top3_z"
        ]
        cols2 = [c for c in cols2 if c in hit.columns]
        st.dataframe(hit[cols2], use_container_width=True, height=280)

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

    metrics_hm = [
        "ar_to_sales",
        "inv_to_sales",
        "tata",
        "ocf_to_ni",
        "cogs_to_sales",
        "sga_to_sales",
        "opm",
        "dep_to_assets",
        "liab_to_assets",
        "linear_raw",
        "iso_score",
        "flag_score",
    ]
    metrics_hm = [m for m in metrics_hm if m in peer.columns]

    if not metrics_hm:
        st.info("열지도로 보여줄 지표가 없습니다.")
        st.stop()

    peer_z = peer.copy()
    for m in metrics_hm:
        mm = peer[m].mean()
        ss = peer[m].std(ddof=0) or EPS
        peer_z[m + "_z_peer"] = (peer[m] - mm) / ss

    z_cols = [m + "_z_peer" for m in metrics_hm]
    z_vals = peer_z[z_cols].values
    labels = [f"{r['company']}_{int(r['year'])}" for _, r in peer.iterrows()]

    fig, ax = plt.subplots(figsize=(1.2 * len(metrics_hm), 0.55 * len(peer) + 1))
    im = ax.imshow(z_vals, aspect="auto", cmap="coolwarm")
    ax.set_xticks(np.arange(len(metrics_hm)))
    ax.set_xticklabels(metrics_hm, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("동종 그룹 내 지표 편차 (z-score)")
    st.pyplot(fig)
