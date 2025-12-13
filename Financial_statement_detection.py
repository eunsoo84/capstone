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


def _robust_z(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").astype(float)
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    med = x.median()
    mad = (x - med).abs().median()
    denom = 1.4826 * mad
    if denom is None or denom == 0 or np.isnan(denom):
        return pd.Series(0.0, index=x.index)
    return ((x - med) / denom).clip(-6, 6)


def _robust_z_group(df: pd.DataFrame, cols: list[str], suffix: str) -> pd.DataFrame:
    g = df.copy()
    for c in cols:
        g[f"{c}{suffix}"] = _robust_z(g[c])
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
        "d_ar_to_sales": "Δ AR/Sales",
        "d_inv_to_sales": "Δ Inv/Sales",
        "d_tata": "Δ TATA",
        "d_ocf_to_ni": "Δ OCF/NI",
        "d_cogs_to_sales": "Δ COGS/Sales",
        "d_sga_to_sales": "Δ SGA/Sales",
        "d_opm": "Δ OPM",
        "d_dep_to_assets": "Δ Dep/Assets",
        "d_liab_to_assets": "Δ Liab/Assets",
        "d_sales": "Δ 매출",
        "d_ar": "Δ 매출채권",
        "d_inventory": "Δ 재고",
        "d_total_assets": "Δ 자산",
        "d_ocf": "Δ OCF",
        "d_net_income": "Δ NI",
    }
    return mapping.get(name, name)


def explain_row(row: pd.Series) -> str:
    t1m, t1z = _pretty_metric(str(row.get("_top1_metric", ""))), float(row.get("_top1_z", 0.0))
    t2m, t2z = _pretty_metric(str(row.get("_top2_metric", ""))), float(row.get("_top2_z", 0.0))
    t3m, t3z = _pretty_metric(str(row.get("_top3_metric", ""))), float(row.get("_top3_z", 0.0))
    return (
        f"- 점수: Change {float(row.get('score_change_part', 0.0)):.3f} / Peer {float(row.get('score_peer_part', 0.0)):.3f} / ISO {float(row.get('score_iso_part', 0.0)):.3f} / 합 {float(row.get('flag_score', 0.0)):.3f}\n"
        f"- Top3 이유: {t1m}({t1z:+.2f}), {t2m}({t2z:+.2f}), {t3m}({t3z:+.2f})"
    )


@dataclass
class Params:
    group_mode: str
    contamination: float
    w_change: float
    w_peer: float
    w_iso: float


@st.cache_data(show_spinner=False)
def run_pipeline_cached(df_raw: pd.DataFrame, params: Params) -> pd.DataFrame:
    return run_pipeline(df_raw, params)


def run_pipeline(df_raw: pd.DataFrame, params: Params) -> pd.DataFrame:
    df = _ensure_columns(df_raw)

    num_cols = [
        "sales", "cogs", "sga", "op_income", "dep",
        "ar", "inventory", "total_assets", "total_liab",
        "ocf", "net_income",
    ]
    for c in num_cols:
        df[c] = _parse_number_series(df[c])

    df = df.reset_index(drop=True)
    df["row_id"] = df.index + 1
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.sort_values(["company", "year"]).reset_index(drop=True)

    df["ar_to_sales"] = _safe_ratio(df["ar"], df["sales"] + EPS).clip(-5, 5)
    df["inv_to_sales"] = _safe_ratio(df["inventory"], df["sales"] + EPS).clip(-5, 5)
    df["ocf_to_ni"] = _safe_ratio(df["ocf"], df["net_income"]).clip(-10, 10)
    df["tata"] = _safe_ratio(df["net_income"] - df["ocf"], df["total_assets"] + EPS).clip(-5, 5)

    df["cogs_to_sales"] = _safe_ratio(df["cogs"], df["sales"] + EPS).clip(-5, 5)
    df["sga_to_sales"] = _safe_ratio(df["sga"], df["sales"] + EPS).clip(-5, 5)
    df["opm"] = _safe_ratio(df["op_income"], df["sales"] + EPS).clip(-5, 5)
    df["dep_to_assets"] = _safe_ratio(df["dep"], df["total_assets"] + EPS).clip(-5, 5)
    df["liab_to_assets"] = _safe_ratio(df["total_liab"], df["total_assets"] + EPS).clip(-5, 5)

    level_metrics = ["ar_to_sales", "inv_to_sales", "tata", "ocf_to_ni"]
    opt = ["cogs_to_sales", "sga_to_sales", "opm", "dep_to_assets", "liab_to_assets"]
    for c in opt:
        if df[c].notna().sum() > 0:
            level_metrics.append(c)

    change_metrics = []
    raw_change_cols = ["sales", "ar", "inventory", "total_assets", "ocf", "net_income"]
    for c in raw_change_cols:
        df[f"d_{c}"] = df.groupby("company")[c].diff().fillna(0.0)
        change_metrics.append(f"d_{c}")

    for m in level_metrics:
        df[f"d_{m}"] = df.groupby("company")[m].diff().fillna(0.0)
        df[f"d_{m}"] = df[f"d_{m}"].clip(-5, 5)
        change_metrics.append(f"d_{m}")

    peer_base = level_metrics
    change_base = change_metrics

    if params.group_mode == "year":
        df = df.groupby("year", group_keys=False).apply(_robust_z_group, cols=peer_base, suffix="_peer_rz")
    elif params.group_mode == "year_industry":
        df = df.groupby(["year", "industry"], group_keys=False).apply(_robust_z_group, cols=peer_base, suffix="_peer_rz")
    else:
        df = _robust_z_group(df, cols=peer_base, suffix="_peer_rz")

    df = df.groupby("company", group_keys=False).apply(_robust_z_group, cols=change_base, suffix="_chg_rz")

    peer_rz_cols = [f"{c}_peer_rz" for c in peer_base]
    chg_rz_cols = [f"{c}_chg_rz" for c in change_base]

    peer_score_raw = np.zeros(len(df), dtype=float)
    for c in peer_rz_cols:
        peer_score_raw += np.abs(df[c].values)

    chg_score_raw = np.zeros(len(df), dtype=float)
    for c in chg_rz_cols:
        chg_score_raw += np.abs(df[c].values)

    df["peer_raw"] = peer_score_raw
    df["change_raw"] = chg_score_raw

    df["peer_norm"] = _pct_rank(df["peer_raw"])
    df["change_norm"] = _pct_rank(df["change_raw"])

    X = df[change_base].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    try:
        Xs = RobustScaler().fit_transform(X)
        iso = IsolationForest(contamination=params.contamination, random_state=42)
        iso.fit(Xs)
        iso_raw = -iso.decision_function(Xs)
        df["iso_raw"] = iso_raw
        df["iso_score"] = _pct_rank(pd.Series(iso_raw)).values
    except Exception:
        df["iso_raw"] = 0.0
        df["iso_score"] = 0.0

    df["score_change_part"] = df["change_norm"]
    df["score_peer_part"] = df["peer_norm"]
    df["score_iso_part"] = df["iso_score"]

    df["flag_score"] = (params.w_change * df["change_norm"]) + (params.w_peer * df["peer_norm"]) + (params.w_iso * df["iso_score"])

    reason_keys = []
    reason_vals = []

    for m in peer_base:
        reason_keys.append(m)
        reason_vals.append(np.abs(df[f"{m}_peer_rz"].values))

    for m in change_base:
        reason_keys.append(m)
        reason_vals.append(np.abs(df[f"{m}_chg_rz"].values))

    reason_mat = np.column_stack(reason_vals)
    idx = np.argsort(-reason_mat, axis=1)
    names = np.array(reason_keys, dtype=object)
    r = np.arange(len(df))

    df["_top1_metric"] = names[idx[:, 0]]
    df["_top1_z"] = reason_mat[r, idx[:, 0]]
    df["_top2_metric"] = names[idx[:, 1]]
    df["_top2_z"] = reason_mat[r, idx[:, 1]]
    df["_top3_metric"] = names[idx[:, 2]]
    df["_top3_z"] = reason_mat[r, idx[:, 2]]

    out = df.sort_values("flag_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


st.sidebar.header("설정")
group_mode_ui = st.sidebar.radio("동종 비교 기준", ["연도", "연도+산업", "전체"])
group_mode_key = {"연도": "year", "연도+산업": "year_industry", "전체": "all"}[group_mode_ui]

contamination = st.sidebar.slider("ISO 민감도", 0.01, 0.30, 0.10, 0.01)

w_change = st.sidebar.slider("변화(Δ) 비중", 0.0, 3.0, 1.5, 0.1)
w_peer = st.sidebar.slider("동종 편차 비중", 0.0, 3.0, 1.0, 0.1)
w_iso = st.sidebar.slider("ISO 비중", 0.0, 3.0, 0.7, 0.1)

st.sidebar.markdown("---")
rule = st.sidebar.radio("출력 규칙", ["OR(추천)", "AND(엄격)"])
k_limit = st.sidebar.slider("후보 상한(K) (0이면 제한 없음)", 0, 200, 30, 1)
top_n = st.sidebar.slider("표시 Top-N", 1, 200, 10, 1)
p_cut = st.sidebar.slider("퍼센타일 컷", 0.70, 0.99, 0.95, 0.01)

st.title("회계 이상 스크리닝")

uploaded = st.file_uploader("CSV 또는 Excel 업로드", type=["csv", "xlsx"])
if uploaded is None:
    st.stop()

df_raw = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)

params = Params(
    group_mode=group_mode_key,
    contamination=contamination,
    w_change=w_change,
    w_peer=w_peer,
    w_iso=w_iso,
)

try:
    df_scored = run_pipeline_cached(df_raw, params)
except Exception as e:
    st.error(f"처리 중 오류: {e}")
    st.stop()

years_all = sorted(pd.to_numeric(df_scored["year"], errors="coerce").dropna().unique().tolist())
sel_year = st.sidebar.selectbox("연도 필터(선택)", ["(전체)"] + [str(int(y)) for y in years_all])

df_work = df_scored.copy()
if sel_year != "(전체)":
    df_work = df_work[df_work["year"].astype(str) == sel_year].copy()
    df_work = df_work.reset_index(drop=True)
    df_work["rank"] = np.arange(1, len(df_work) + 1)

if df_work.empty:
    st.info("필터 결과가 없습니다.")
    st.stop()

thr_flag = float(df_work["flag_score"].quantile(p_cut))
thr_change = float(df_work["change_norm"].quantile(p_cut))
thr_peer = float(df_work["peer_norm"].quantile(p_cut))
thr_iso = float(df_work["iso_score"].quantile(p_cut))

cond_flag = df_work["flag_score"] >= thr_flag
cond_change = df_work["change_norm"] >= thr_change
cond_peer = df_work["peer_norm"] >= thr_peer
cond_iso = df_work["iso_score"] >= thr_iso

if rule.startswith("OR"):
    mask = cond_flag & (cond_change | cond_peer | cond_iso)
else:
    mask = cond_flag & cond_change & cond_peer

df_candidates = df_work[mask].sort_values("flag_score", ascending=False).reset_index(drop=True)

if k_limit > 0:
    df_view = df_candidates.head(int(k_limit)).copy()
else:
    df_view = df_candidates.copy()

st.caption(
    f"후보: {df_view.shape[0]} / {df_work.shape[0]} | "
    f"컷(p{int(p_cut*100)}): flag≥{thr_flag:.4f}, change≥{thr_change:.2f}, peer≥{thr_peer:.2f}, iso≥{thr_iso:.2f} | "
    f"규칙={rule.split('(')[0]} | K={'제한없음' if k_limit==0 else k_limit}"
)

tab1, tab2 = st.tabs(["🔍 후보 리스트 & Top3 이유", "🌡️ 동종 그룹 열지도"])

with tab1:
    if df_view.empty:
        st.info("현재 기준에서는 후보가 없습니다. (퍼센타일 컷을 p90으로 낮춰보면 바로 뜨는 경우가 많습니다.)")
    else:
        df_top = df_view.head(int(top_n)).copy()
        show_cols = [
            "rank","company","year","industry",
            "flag_score","score_change_part","score_peer_part","score_iso_part",
            "change_norm","peer_norm","iso_score",
            "_top1_metric","_top1_z","_top2_metric","_top2_z","_top3_metric","_top3_z",
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

    all_companies = sorted(df_work["company"].astype(str).unique().tolist())
    years2 = sorted(df_work["year"].dropna().unique().tolist())

    sel_companies = st.multiselect("조회 회사 선택", all_companies, default=[])
    sel_year2 = st.selectbox("연도 선택(선택)", ["(전체)"] + [str(int(y)) for y in years2])

    if len(sel_companies) == 0:
        st.info("회사 선택을 해야 조회가 됩니다.")
    else:
        hit = df_work[df_work["company"].astype(str).isin(sel_companies)].copy()
        if sel_year2 != "(전체)":
            hit = hit[hit["year"].astype(str) == sel_year2].copy()

        if hit.empty:
            st.info("검색 결과가 없습니다.")
        else:
            hit = hit.sort_values("rank").copy()
            cols2 = [
                "rank","company","year","industry",
                "flag_score","change_norm","peer_norm","iso_score",
                "_top1_metric","_top1_z","_top2_metric","_top2_z","_top3_metric","_top3_z"
            ]
            cols2 = [c for c in cols2 if c in hit.columns]
            st.dataframe(hit[cols2], use_container_width=True, height=240)

with tab2:
    years = sorted(df_work["year"].dropna().unique())
    industries = sorted(df_work["industry"].dropna().unique())

    if not years or not industries:
        st.info("열지도를 만들 데이터가 부족합니다.")
    else:
        sel_year_h = st.selectbox("연도 선택", years, key="peer_year")
        sel_ind = st.selectbox("산업 선택", industries, key="peer_ind")

        subset = df_work[(df_work["year"] == sel_year_h) & (df_work["industry"] == sel_ind)].copy()

        if subset.shape[0] < 3:
            st.info("해당 연도·산업 조합의 데이터가 3개 미만이라 열지도를 만들 수 없습니다.")
        else:
            companies = subset["company"].unique().tolist()
            sel_comp = st.selectbox("기준 회사 선택", companies, key="peer_comp")

            subset["size_metric"] = np.log1p(pd.to_numeric(subset["total_assets"], errors="coerce").fillna(0.0))
            subset["growth_metric"] = pd.to_numeric(subset.get("sales_yoy", 0.0), errors="coerce").fillna(0.0)

            sales = pd.to_numeric(subset.get("sales", 0.0), errors="coerce").fillna(0.0)
            ni = pd.to_numeric(subset.get("net_income", 0.0), errors="coerce").fillna(0.0)
            subset["profit_metric"] = (ni / (sales + EPS)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

            for c in ["size_metric", "growth_metric", "profit_metric"]:
                subset[c + "_z"] = _robust_z(subset[c])

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

                    cols_for_hm = [c for c in ["flag_score", "change_norm", "peer_norm", "iso_score"] if c in peer.columns]
                    if not cols_for_hm:
                        st.info("열지도로 보여줄 지표가 없습니다.")
                    else:
                        peer_z = peer.copy()
                        for m in cols_for_hm:
                            peer_z[m + "_z_peer"] = _robust_z(peer_z[m])

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
