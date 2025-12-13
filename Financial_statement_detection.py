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


def _clip_series(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return s.clip(lower=lo, upper=hi)


def _robust_z(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    med = x.median()
    mad = (x - med).abs().median()
    denom = 1.4826 * mad
    if denom is None or denom == 0 or np.isnan(denom):
        return pd.Series(0.0, index=x.index)
    return (x - med) / denom


def _robust_z_group(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = df.copy()
    for c in cols:
        g[c + "_rz"] = _robust_z(g[c].replace([np.inf, -np.inf], np.nan).fillna(0.0))
        g[c + "_rz"] = g[c + "_rz"].clip(-6, 6)
    return g


def _pct_rank(x: pd.Series) -> pd.Series:
    return x.rank(pct=True)


def _pretty_metric(name: str) -> str:
    mapping = {
        "ar_to_sales": "AR/Sales",
        "inv_to_sales": "Inv/Sales",
        "tata": "TATA",
        "ocf_to_ni": "OCF/NI",
        "cogs_to_sales": "COGS/Sales",
        "sga_to_sales": "SGA/Sales",
        "opm": "OPM",
        "dep_to_assets": "Dep/Assets",
        "liab_to_assets": "Liab/Assets",
        "ar_to_sales_delta": "Δ AR/Sales",
        "inv_to_sales_delta": "Δ Inv/Sales",
        "tata_delta": "Δ TATA",
        "ocf_to_ni_delta": "Δ OCF/NI",
        "cogs_to_sales_delta": "Δ COGS/Sales",
        "sga_to_sales_delta": "Δ SGA/Sales",
        "opm_delta": "Δ OPM",
        "dep_to_assets_delta": "Δ Dep/Assets",
        "liab_to_assets_delta": "Δ Liab/Assets",
        "chg_ar_to_sales": "Δ(원본대비) AR/Sales",
        "chg_inv_to_sales": "Δ(원본대비) Inv/Sales",
        "chg_tata": "Δ(원본대비) TATA",
        "chg_ocf_to_ni": "Δ(원본대비) OCF/NI",
        "chg_cogs_to_sales": "Δ(원본대비) COGS/Sales",
        "chg_sga_to_sales": "Δ(원본대비) SGA/Sales",
        "chg_opm": "Δ(원본대비) OPM",
        "chg_dep_to_assets": "Δ(원본대비) Dep/Assets",
        "chg_liab_to_assets": "Δ(원본대비) Liab/Assets",
    }
    return mapping.get(name, name)


def _compute_features(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
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

    df = df.sort_values(["company", "year"]).reset_index(drop=True)

    df["ar_to_sales"] = _safe_ratio(df["ar"], df["sales"] + EPS)
    df["inv_to_sales"] = _safe_ratio(df["inventory"], df["sales"] + EPS)
    df["ocf_to_ni"] = _safe_ratio(df["ocf"], df["net_income"])
    df["tata"] = _safe_ratio(df["net_income"] - df["ocf"], df["total_assets"] + EPS)

    df["cogs_to_sales"] = _safe_ratio(df["cogs"], df["sales"] + EPS)
    df["sga_to_sales"] = _safe_ratio(df["sga"], df["sales"] + EPS)
    df["opm"] = _safe_ratio(df["op_income"], df["sales"] + EPS)
    df["dep_to_assets"] = _safe_ratio(df["dep"], df["total_assets"] + EPS)
    df["liab_to_assets"] = _safe_ratio(df["total_liab"], df["total_assets"] + EPS)

    df["ar_to_sales"] = _clip_series(df["ar_to_sales"], -5, 5)
    df["inv_to_sales"] = _clip_series(df["inv_to_sales"], -5, 5)
    df["ocf_to_ni"] = _clip_series(df["ocf_to_ni"], -10, 10)
    df["tata"] = _clip_series(df["tata"], -5, 5)

    df["cogs_to_sales"] = _clip_series(df["cogs_to_sales"], -5, 5)
    df["sga_to_sales"] = _clip_series(df["sga_to_sales"], -5, 5)
    df["opm"] = _clip_series(df["opm"], -5, 5)
    df["dep_to_assets"] = _clip_series(df["dep_to_assets"], -5, 5)
    df["liab_to_assets"] = _clip_series(df["liab_to_assets"], -5, 5)

    level_metrics = [
        "ar_to_sales",
        "inv_to_sales",
        "tata",
        "ocf_to_ni",
    ]

    opt = ["cogs_to_sales", "sga_to_sales", "opm", "dep_to_assets", "liab_to_assets"]
    for c in opt:
        if df[c].notna().sum() > 0:
            level_metrics.append(c)

    for m in level_metrics:
        df[m + "_delta"] = df.groupby("company")[m].diff().fillna(0.0)
        df[m + "_delta"] = _clip_series(df[m + "_delta"], -5, 5)

    metrics = level_metrics + [m + "_delta" for m in level_metrics]

    return df, metrics


@dataclass
class PipelineParams:
    group_mode: str
    contamination: float
    w_linear: float
    w_iso: float
    w_delta: float


@st.cache_data(show_spinner=False)
def run_single_cached(df_raw: pd.DataFrame, params: PipelineParams):
    return run_single(df_raw, params)


def run_single(df_raw: pd.DataFrame, params: PipelineParams) -> pd.DataFrame:
    df, metrics = _compute_features(df_raw)

    if params.group_mode == "year":
        df = df.groupby("year", group_keys=False).apply(_robust_z_group, cols=metrics)
    elif params.group_mode == "year_industry":
        df = df.groupby(["year", "industry"], group_keys=False).apply(_robust_z_group, cols=metrics)
    else:
        df = _robust_z_group(df, cols=metrics)

    rz_cols = [m + "_rz" for m in metrics]

    base_m = [m for m in metrics if not m.endswith("_delta")]
    delta_m = [m for m in metrics if m.endswith("_delta")]

    base_score = np.zeros(len(df), dtype=float)
    for m in base_m:
        base_score += np.abs(df[m + "_rz"].values)

    delta_score = np.zeros(len(df), dtype=float)
    for m in delta_m:
        delta_score += np.abs(df[m + "_rz"].values)

    df["linear_raw"] = (params.w_linear * base_score) + (params.w_delta * delta_score)
    df["linear_norm"] = _pct_rank(df["linear_raw"])

    X = df[metrics].fillna(0.0).values
    try:
        scaler = RobustScaler()
        Xs = scaler.fit_transform(X)
        iso = IsolationForest(contamination=params.contamination, random_state=42)
        iso.fit(Xs)
        iso_raw = -iso.decision_function(Xs)
        df["iso_raw"] = iso_raw
        df["iso_score"] = _pct_rank(pd.Series(iso_raw)).values
    except Exception:
        df["iso_raw"] = 0.0
        df["iso_score"] = 0.0

    df["score_linear_part"] = df["linear_norm"]
    df["score_iso_part"] = df["iso_score"]
    df["flag_score"] = (params.w_linear * df["linear_norm"]) + (params.w_iso * df["iso_score"])

    rz_mat = np.column_stack([df[c].values for c in rz_cols])
    abs_mat = np.abs(rz_mat)
    idx = np.argsort(-abs_mat, axis=1)

    names = np.array(metrics, dtype=object)
    r = np.arange(len(df))

    df["_top1_metric"] = names[idx[:, 0]]
    df["_top1_z"] = rz_mat[r, idx[:, 0]]
    df["_top2_metric"] = names[idx[:, 1]]
    df["_top2_z"] = rz_mat[r, idx[:, 1]]
    df["_top3_metric"] = names[idx[:, 2]]
    df["_top3_z"] = rz_mat[r, idx[:, 2]]

    df_scored = df.sort_values("flag_score", ascending=False).reset_index(drop=True)
    df_scored["rank"] = np.arange(1, len(df_scored) + 1)
    return df_scored


@st.cache_data(show_spinner=False)
def run_compare_cached(df_base_raw: pd.DataFrame, df_test_raw: pd.DataFrame, params: PipelineParams):
    return run_compare(df_base_raw, df_test_raw, params)


def run_compare(df_base_raw: pd.DataFrame, df_test_raw: pd.DataFrame, params: PipelineParams) -> pd.DataFrame:
    base, base_metrics = _compute_features(df_base_raw)
    test, test_metrics = _compute_features(df_test_raw)

    metrics = [m for m in base_metrics if m in test_metrics]
    key = ["company", "year", "industry"]

    b = base[key + metrics].copy()
    t = test[key + metrics + ["row_id"]].copy()

    mdf = pd.merge(
        t,
        b,
        on=key,
        how="left",
        suffixes=("_test", "_base"),
    )

    for m in metrics:
        mdf["chg_" + m] = (mdf[m + "_test"] - mdf[m + "_base"]).abs().fillna(0.0)

    chg_metrics = ["chg_" + m for m in metrics]

    if params.group_mode == "year":
        mdf = mdf.groupby("year", group_keys=False).apply(_robust_z_group, cols=chg_metrics)
    elif params.group_mode == "year_industry":
        mdf = mdf.groupby(["year", "industry"], group_keys=False).apply(_robust_z_group, cols=chg_metrics)
    else:
        mdf = _robust_z_group(mdf, cols=chg_metrics)

    rz_cols = [m + "_rz" for m in chg_metrics]
    score = np.zeros(len(mdf), dtype=float)
    for m in chg_metrics:
        score += np.abs(mdf[m + "_rz"].values)

    mdf["linear_raw"] = score
    mdf["linear_norm"] = _pct_rank(mdf["linear_raw"])
    mdf["iso_score"] = 0.0
    mdf["score_linear_part"] = mdf["linear_norm"]
    mdf["score_iso_part"] = 0.0
    mdf["flag_score"] = mdf["linear_norm"]

    rz_mat = np.column_stack([mdf[c].values for c in rz_cols])
    abs_mat = np.abs(rz_mat)
    idx = np.argsort(-abs_mat, axis=1)

    names = np.array(chg_metrics, dtype=object)
    r = np.arange(len(mdf))

    mdf["_top1_metric"] = names[idx[:, 0]]
    mdf["_top1_z"] = rz_mat[r, idx[:, 0]]
    mdf["_top2_metric"] = names[idx[:, 1]]
    mdf["_top2_z"] = rz_mat[r, idx[:, 1]]
    mdf["_top3_metric"] = names[idx[:, 2]]
    mdf["_top3_z"] = rz_mat[r, idx[:, 2]]

    df_scored = mdf.sort_values("flag_score", ascending=False).reset_index(drop=True)
    df_scored["rank"] = np.arange(1, len(df_scored) + 1)
    return df_scored


def explain_row(row: pd.Series) -> str:
    t1m, t1z = _pretty_metric(str(row.get("_top1_metric", ""))), float(row.get("_top1_z", 0.0))
    t2m, t2z = _pretty_metric(str(row.get("_top2_metric", ""))), float(row.get("_top2_z", 0.0))
    t3m, t3z = _pretty_metric(str(row.get("_top3_metric", ""))), float(row.get("_top3_z", 0.0))
    return (
        f"- 점수: {float(row.get('flag_score', 0.0)):.3f}\n"
        f"- Top3 이유: {t1m}({t1z:+.2f}), {t2m}({t2z:+.2f}), {t3m}({t3z:+.2f})"
    )


st.sidebar.header("설정")
group_mode_ui = st.sidebar.radio("그룹 표준화 기준", ["연도", "연도+산업", "전체"])
group_mode_key = {"연도": "year", "연도+산업": "year_industry", "전체": "all"}[group_mode_ui]

contamination = st.sidebar.slider("ISO 민감도", 0.01, 0.30, 0.10, 0.01)
w_linear = st.sidebar.slider("Linear 비중", 0.0, 3.0, 1.0, 0.1)
w_iso = st.sidebar.slider("ISO 비중", 0.0, 3.0, 1.0, 0.1)
w_delta = st.sidebar.slider("전년 대비 변화(Δ) 반영", 0.0, 3.0, 1.0, 0.1)

st.sidebar.markdown("---")
rule = st.sidebar.radio("출력 규칙", ["OR(추천)", "AND(엄격)"])
k_limit = st.sidebar.slider("후보 상한(K) (0이면 제한 없음)", 0, 200, 0, 1)
top_n = st.sidebar.slider("표시 Top-N", 1, 200, 10, 1)
p_cut_lin = st.sidebar.slider("Linear 컷(퍼센타일)", 0.70, 0.99, 0.95, 0.01)
p_cut_iso = st.sidebar.slider("ISO 컷(퍼센타일)", 0.70, 0.99, 0.95, 0.01)

st.sidebar.markdown("---")
target_year = st.sidebar.selectbox("대상 연도 필터(선택)", ["(전체)"], index=0)

st.title("회계 이상 스크리닝")

base_file = st.file_uploader("원본(정상) 파일(선택)", type=["csv", "xlsx"], key="base")
test_file = st.file_uploader("검사 대상(변경/시연) 파일", type=["csv", "xlsx"], key="test")

if test_file is None:
    st.stop()

def _read_file(u):
    return pd.read_csv(u) if u.name.lower().endswith(".csv") else pd.read_excel(u)

df_test_raw = _read_file(test_file)

params = PipelineParams(
    group_mode=group_mode_key,
    contamination=contamination,
    w_linear=w_linear,
    w_iso=w_iso,
    w_delta=w_delta,
)

compare_mode = base_file is not None
if compare_mode:
    df_base_raw = _read_file(base_file)
    df_scored = run_compare_cached(df_base_raw, df_test_raw, params)
else:
    df_scored = run_single_cached(df_test_raw, params)

df_scored = df_scored.copy()

years_all = sorted(pd.to_numeric(df_scored["year"], errors="coerce").dropna().unique().tolist())
target_year = st.sidebar.selectbox("대상 연도 필터(선택)", ["(전체)"] + [str(int(y)) for y in years_all])

if target_year != "(전체)":
    df_scored = df_scored[df_scored["year"].astype(str) == target_year].copy()
    df_scored = df_scored.reset_index(drop=True)
    df_scored["rank"] = np.arange(1, len(df_scored) + 1)

if df_scored.empty:
    st.info("필터 결과가 없습니다.")
    st.stop()

if not compare_mode:
    thr_lin = float(df_scored["linear_norm"].quantile(p_cut_lin))
    thr_i = float(df_scored["iso_score"].quantile(p_cut_iso))
    cond_l = df_scored["linear_norm"] >= thr_lin
    cond_i = df_scored["iso_score"] >= thr_i
    if rule.startswith("OR"):
        mask = (cond_l | cond_i)
    else:
        mask = (cond_l & cond_i)
    df_candidates = df_scored[mask].sort_values("flag_score", ascending=False).reset_index(drop=True)
else:
    thr_lin = float(df_scored["flag_score"].quantile(p_cut_lin))
    thr_i = 0.0
    df_candidates = df_scored[df_scored["flag_score"] >= thr_lin].sort_values("flag_score", ascending=False).reset_index(drop=True)

if k_limit > 0:
    df_view = df_candidates.head(int(k_limit)).copy()
else:
    df_view = df_candidates.copy()

st.caption(
    f"모드: {'비교(원본대비 변경 감지)' if compare_mode else '단일(비지도 이상치)'} | "
    f"후보: {df_view.shape[0]} / {df_scored.shape[0]} | "
    f"컷: Linear(p{int(p_cut_lin*100)}) | 규칙={rule.split('(')[0]} | K={'제한없음' if k_limit==0 else k_limit}"
)

tab1, tab2 = st.tabs(["🔍 후보 리스트 & Top3 이유", "🌡️ 동종 그룹 열지도"])

with tab1:
    if df_view.empty:
        st.info("현재 기준에서는 후보가 없습니다. (컷을 p90 정도로 낮추면 보통 바로 뜹니다.)")
    else:
        df_top = df_view.head(int(top_n)).copy()
        show_cols = [
            "rank", "company", "year", "industry",
            "flag_score",
            "linear_raw", "linear_norm",
            "iso_score",
            "_top1_metric", "_top1_z", "_top2_metric", "_top2_z", "_top3_metric", "_top3_z",
        ]
        show_cols = [c for c in show_cols if c in df_top.columns]
        st.dataframe(df_top[show_cols], use_container_width=True, height=360)

        top_k = min(3, len(df_top))
        for i in range(top_k):
            r0 = df_top.iloc[i]
            with st.expander(f"#{int(r0['rank'])} {r0['company']} ({int(r0['year'])})", expanded=(i == 0)):
                st.markdown(explain_row(r0))

    st.markdown("---")
    st.subheader("회사/연도 조회(통과 여부 무관)")

    all_companies = sorted(df_scored["company"].astype(str).unique().tolist())
    years2 = sorted(df_scored["year"].dropna().unique().tolist())

    sel_companies = st.multiselect("조회 회사 선택", all_companies, default=[])
    sel_year2 = st.selectbox("연도 선택(선택)", ["(전체)"] + [str(int(y)) for y in years2])

    if len(sel_companies) == 0:
        st.info("회사 선택을 해야 조회가 됩니다.")
    else:
        hit = df_scored[df_scored["company"].astype(str).isin(sel_companies)].copy()
        if sel_year2 != "(전체)":
            hit = hit[hit["year"].astype(str) == sel_year2].copy()

        if hit.empty:
            st.info("검색 결과가 없습니다.")
        else:
            hit = hit.sort_values("rank").copy()
            cols2 = [
                "rank","company","year","industry",
                "flag_score","linear_norm","iso_score",
                "_top1_metric","_top1_z","_top2_metric","_top2_z","_top3_metric","_top3_z"
            ]
            cols2 = [c for c in cols2 if c in hit.columns]
            st.dataframe(hit[cols2], use_container_width=True, height=240)

with tab2:
    years = sorted(df_scored["year"].dropna().unique())
    industries = sorted(df_scored["industry"].dropna().unique())

    if not years or not industries:
        st.info("열지도를 만들 데이터가 부족합니다.")
    else:
        sel_year = st.selectbox("연도 선택", years, key="peer_year")
        sel_ind = st.selectbox("산업 선택", industries, key="peer_ind")

        subset = df_scored[(df_scored["year"] == sel_year) & (df_scored["industry"] == sel_ind)].copy()

        if subset.shape[0] < 3:
            st.info("해당 연도·산업 조합의 데이터가 3개 미만이라 열지도를 만들 수 없습니다.")
        else:
            companies = subset["company"].unique().tolist()
            sel_comp = st.selectbox("기준 회사 선택", companies, key="peer_comp")

            if "total_assets" not in subset.columns or "sales_yoy" not in subset.columns:
                st.info("동종 그룹 계산에 필요한 컬럼이 부족합니다.")
            else:
                subset["sales_yoy"] = pd.to_numeric(subset.get("sales_yoy", 0.0), errors="coerce").fillna(0.0)
                subset["size_metric"] = np.log1p(pd.to_numeric(subset["total_assets"], errors="coerce").fillna(0.0))
                subset["growth_metric"] = subset["sales_yoy"].fillna(0.0)

                sales = pd.to_numeric(subset.get("sales", 0.0), errors="coerce").fillna(0.0)
                ni = pd.to_numeric(subset.get("net_income", 0.0), errors="coerce").fillna(0.0)
                subset["profit_metric"] = (ni / (sales + EPS)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

                for c in ["size_metric", "growth_metric", "profit_metric"]:
                    med = subset[c].median()
                    mad = (subset[c] - med).abs().median()
                    denom = 1.4826 * mad
                    if denom is None or denom == 0 or np.isnan(denom):
                        subset[c + "_z"] = 0.0
                    else:
                        subset[c + "_z"] = ((subset[c] - med) / denom).clip(-6, 6)

                focus = subset[subset["company"] == sel_comp].copy()
                if focus.empty:
                    st.info("선택한 회사 데이터가 없습니다.")
                else:
                    fr = focus.iloc[0]
                    f_vec = np.array([float(fr["size_metric_z"]), float(fr["growth_metric_z"]), float(fr["profit_metric_z"])])

                    subset["peer_dist"] = subset.apply(
                        lambda r1: np.linalg.norm(np.array([r1["size_metric_z"], r1["growth_metric_z"], r1["profit_metric_z"]]) - f_vec),
                        axis=1,
                    )

                    k_max = min(10, subset.shape[0])
                    k_default = min(5, subset.shape[0])
                    if k_max < 3:
                        st.info("동종 그룹 크기를 설정할 수 없습니다.")
                    else:
                        k = st.slider("동종 그룹 크기", 3, k_max, k_default)
                        peer = subset.nsmallest(k, "peer_dist").copy()

                        cols_for_hm = []
                        for c in ["flag_score", "linear_norm", "iso_score"]:
                            if c in peer.columns:
                                cols_for_hm.append(c)

                        if not cols_for_hm:
                            st.info("열지도로 보여줄 지표가 없습니다.")
                        else:
                            peer_z = peer.copy()
                            for m in cols_for_hm:
                                med = peer_z[m].median()
                                mad = (peer_z[m] - med).abs().median()
                                denom = 1.4826 * mad
                                if denom is None or denom == 0 or np.isnan(denom):
                                    peer_z[m + "_z_peer"] = 0.0
                                else:
                                    peer_z[m + "_z_peer"] = ((peer_z[m] - med) / denom).clip(-6, 6)

                            z_cols2 = [m + "_z_peer" for m in cols_for_hm]
                            z_vals = peer_z[z_cols2].values
                            labels = [f"{r2['company']}_{int(r2['year'])}" for _, r2 in peer.iterrows()]

                            fig, ax = plt.subplots(figsize=(1.2 * len(cols_for_hm), 0.55 * len(peer) + 1))
                            im = ax.imshow(z_vals, aspect="auto", cmap="coolwarm")
                            ax.set_xticks(np.arange(len(cols_for_hm)))
                            ax.set_xticklabels(cols_for_hm, rotation=45, ha="right")
                            ax.set_yticks(np.arange(len(labels)))
                            ax.set_yticklabels(labels)
                            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                            ax.set_title("동종 그룹 내 지표 편차 (Robust z)")
                            st.pyplot(fig)
