import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="부정회계 탐지 스크리닝", layout="wide")
st.title("부정회계 탐지 스크리닝")

ALIASES = {
    "company": ["company", "회사명", "법인명", "기업명"],
    "year": ["year", "결산연도", "연도"],
    "industry": ["industry", "업종", "산업", "섹터"],
    "sales": ["sales", "매출액", "수익", "매출"],
    "cogs": ["cogs", "매출원가"],
    "sga": ["sga", "판매비와관리비", "판관비"],
    "ebit": ["ebit", "영업이익"],
    "depr": ["depr", "감가상각비", "감가상각"],
    "ar": ["ar", "accounts_receivable", "매출채권"],
    "inventory": ["inventory", "재고자산", "재고"],
    "total_assets": ["total_assets", "자산총계", "총자산"],
    "total_liabilities": ["total_liabilities", "부채총계", "총부채"],
    "ocf": ["ocf", "영업활동현금흐름", "영업현금흐름"],
    "net_income": ["net_income", "당기순이익", "순이익"]
}

NUMERIC_CANDIDATES = [
    "year", "sales", "cogs", "sga", "ebit", "depr", "ar", "inventory",
    "total_assets", "total_liabilities", "ocf", "net_income"
]


def _normalize_column_name(col):
    col = str(col)
    col = col.replace("\ufeff", "")
    col = col.replace("\xa0", " ")
    col = col.strip()
    col = " ".join(col.split())
    return col


def _normalize_columns(df):
    df = df.copy()
    df.columns = [_normalize_column_name(c) for c in df.columns]
    return df


def _read_uploaded_file(uploaded):
    name = uploaded.name.lower()

    if name.endswith(".csv"):
        raw = uploaded.getvalue()
        for enc in ["utf-8-sig", "cp949", "utf-8"]:
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=enc)
            except Exception:
                continue
        return pd.read_csv(io.BytesIO(raw))

    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded)

    raise ValueError("csv 또는 xlsx 파일만 업로드할 수 있습니다.")


def _safe_to_numeric(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("\xa0", "", regex=False),
        errors="coerce"
    )


def _apply_aliases(df):
    df = _normalize_columns(df)
    current_cols = list(df.columns)
    rename_map = {}

    normalized_lookup = {}
    for c in current_cols:
        key = c.lower().replace(" ", "").replace("_", "")
        normalized_lookup[key] = c

    for standard, candidates in ALIASES.items():
        found = None
        for cand in candidates:
            cand_key = str(cand).lower().replace(" ", "").replace("_", "")
            if cand_key in normalized_lookup:
                found = normalized_lookup[cand_key]
                break
        if found is not None and found != standard:
            rename_map[found] = standard

    df = df.rename(columns=rename_map)
    return df


def _ensure_columns(df):
    df = _apply_aliases(df)

    if "industry" not in df.columns:
        df["industry"] = "미지정"

    required = ["company", "year", "sales", "ar", "inventory", "total_assets", "ocf", "net_income"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 누락되었습니다: {missing}")

    for col in NUMERIC_CANDIDATES:
        if col in df.columns:
            df[col] = _safe_to_numeric(df[col])

    df["company"] = df["company"].astype(str).str.strip()
    df["industry"] = df["industry"].astype(str).str.strip().replace("", "미지정")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    df = df.dropna(subset=["company", "year", "sales", "ar", "inventory", "total_assets", "ocf", "net_income"]).copy()
    df["year"] = df["year"].astype(int)

    return df


def _winsorize_series(s, lower=0.01, upper=0.99):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 3:
        return s
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lo, hi)


def _robust_zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    mean = s.mean()
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mean) / std


def _group_zscore(df, col, mode):
    vals = pd.to_numeric(df[col], errors="coerce")

    if mode == "all":
        return _robust_zscore(vals)

    if mode == "year":
        return df.groupby(["year"], dropna=False)[col].transform(lambda x: _robust_zscore(pd.to_numeric(x, errors="coerce")))

    if mode == "year_industry":
        return df.groupby(["year", "industry"], dropna=False)[col].transform(lambda x: _robust_zscore(pd.to_numeric(x, errors="coerce")))

    return _robust_zscore(vals)


def _minmax_scale(s):
    s = pd.to_numeric(s, errors="coerce")
    mn = s.min()
    mx = s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def _first_digit(x):
    if pd.isna(x):
        return np.nan
    x = abs(float(x))
    if x <= 0:
        return np.nan
    while x < 1:
        x *= 10
    while x >= 10:
        x /= 10
    return int(x)


def _benford_expected():
    return np.array([math.log10(1 + 1 / d) for d in range(1, 10)])


def _benford_stats(values):
    digits = pd.Series(values).dropna().apply(_first_digit).dropna()
    digits = digits[digits.between(1, 9)]

    n = len(digits)
    nonzero_vals = pd.Series(values).dropna()
    nonzero_vals = nonzero_vals[nonzero_vals != 0]
    if len(nonzero_vals) > 0:
        span = abs(nonzero_vals).max() / max(abs(nonzero_vals).min(), 1e-12)
    else:
        span = np.nan

    exp = _benford_expected()
    obs_counts = digits.value_counts().reindex(range(1, 10), fill_value=0).sort_index()
    obs = obs_counts / obs_counts.sum() if obs_counts.sum() > 0 else pd.Series(np.zeros(9), index=range(1, 10))
    mad = np.mean(np.abs(obs.values - exp))

    applicable = True
    reasons = []
    if n < 100:
        applicable = False
        reasons.append(f"표본 수(n={n})가 충분하지 않습니다(권장 100개 이상).")
    if pd.isna(span) or span < 100:
        applicable = False
        if pd.isna(span):
            reasons.append("값의 범위(span)를 계산하기 어렵습니다.")
        else:
            reasons.append(f"값의 범위(span={span:.1f})가 충분히 넓지 않습니다(권장 100 이상).")

    return {
        "n": n,
        "span": span,
        "mad": mad,
        "obs": obs.values,
        "exp": exp,
        "digits": list(range(1, 10)),
        "applicable": applicable,
        "reason_text": " / ".join(reasons) if reasons else ""
    }


def _continuous_years_count(years):
    years = sorted(pd.Series(years).dropna().astype(int).unique())
    if not years:
        return 0
    longest = 1
    cur = 1
    for i in range(1, len(years)):
        if years[i] == years[i - 1] + 1:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 1
    return longest


def _prepare_features(df):
    df = df.copy()

    df["ar_to_sales"] = df["ar"] / df["sales"].replace(0, np.nan)
    df["inv_to_sales"] = df["inventory"] / df["sales"].replace(0, np.nan)
    df["tata"] = (df["net_income"] - df["ocf"]) / df["total_assets"].replace(0, np.nan)
    df["ocf_to_ni"] = df["ocf"] / df["net_income"].replace(0, np.nan)

    for c in ["ar_to_sales", "inv_to_sales", "tata", "ocf_to_ni"]:
        df[c] = _winsorize_series(df[c])

    return df


def _run_pipeline(df, group_mode="year_industry", contamination=0.10, w_beneish=1.0, w_iforest=1.0, w_benford=1.0):
    df = _prepare_features(df)

    df["ar_to_sales_z"] = _group_zscore(df, "ar_to_sales", group_mode)
    df["inv_to_sales_z"] = _group_zscore(df, "inv_to_sales", group_mode)
    df["tata_z"] = _group_zscore(df, "tata", group_mode)
    df["ocf_to_ni_z"] = _group_zscore(df, "ocf_to_ni", group_mode)

    df["mscore_raw"] = (
        df["ar_to_sales_z"].fillna(0)
        + df["inv_to_sales_z"].fillna(0)
        + df["tata_z"].fillna(0)
        - df["ocf_to_ni_z"].fillna(0)
    )

    feature_cols = ["ar_to_sales", "inv_to_sales", "tata", "ocf_to_ni"]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)

    if len(X) >= 5:
        iso = IsolationForest(
            n_estimators=300,
            contamination=float(contamination),
            random_state=42
        )
        iso.fit(X)
        raw_score = -iso.score_samples(X)
        df["iso_score"] = raw_score
    else:
        df["iso_score"] = 0.0

    benford_map = {}
    benford_detail = {}

    for (year, industry), g in df.groupby(["year", "industry"], dropna=False):
        stats = _benford_stats(g["sales"])
        benford_detail[(year, industry)] = stats
        benford_map[(year, industry)] = stats["mad"] if stats["applicable"] else np.nan

    df["benford_mad"] = df.apply(lambda r: benford_map.get((r["year"], r["industry"]), np.nan), axis=1)

    mscore_norm = _minmax_scale(df["mscore_raw"].fillna(0))
    iso_norm = _minmax_scale(df["iso_score"].fillna(0))
    benford_norm = _minmax_scale(df["benford_mad"].fillna(0))

    total_weight = max(w_beneish + w_iforest + w_benford, 1e-12)
    df["flag_score"] = (
        w_beneish * mscore_norm
        + w_iforest * iso_norm
        + w_benford * benford_norm
    ) / total_weight

    baseline = _run_baseline_for_consistency(df)
    return df, benford_detail, baseline


def _run_baseline_for_consistency(df):
    temp = df.copy()

    mscore_norm = _minmax_scale(temp["mscore_raw"].fillna(0))

    feature_cols = ["ar_to_sales", "inv_to_sales", "tata", "ocf_to_ni"]
    X = temp[feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)

    if len(X) >= 5:
        iso = IsolationForest(
            n_estimators=300,
            contamination=0.10,
            random_state=42
        )
        iso.fit(X)
        temp["iso_score_base"] = -iso.score_samples(X)
    else:
        temp["iso_score_base"] = 0.0

    benford_map = {}
    for (year, industry), g in temp.groupby(["year", "industry"], dropna=False):
        stats = _benford_stats(g["sales"])
        benford_map[(year, industry)] = stats["mad"] if stats["applicable"] else np.nan

    temp["benford_mad_base"] = temp.apply(lambda r: benford_map.get((r["year"], r["industry"]), np.nan), axis=1)

    iso_norm = _minmax_scale(temp["iso_score_base"].fillna(0))
    benford_norm = _minmax_scale(temp["benford_mad_base"].fillna(0))

    temp["flag_score_base"] = (mscore_norm + iso_norm + benford_norm) / 3
    return temp


def _top_n_table(df, top_n):
    show_cols = [
        "company", "year", "industry", "flag_score", "mscore_raw", "iso_score",
        "benford_mad", "ar_to_sales", "inv_to_sales", "ocf_to_ni"
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    out = df.sort_values("flag_score", ascending=False).head(top_n).copy()
    out.insert(0, "rank", range(1, len(out) + 1))
    return out[["rank"] + show_cols]


def _consistent_suspects(current_df, baseline_df, top_n):
    cur = current_df.sort_values("flag_score", ascending=False).head(top_n)[["company", "year"]].copy()
    base = baseline_df.sort_values("flag_score_base", ascending=False).head(top_n)[["company", "year"]].copy()

    cur["key"] = cur["company"].astype(str) + "_" + cur["year"].astype(str)
    base["key"] = base["company"].astype(str) + "_" + base["year"].astype(str)

    inter = set(cur["key"]).intersection(set(base["key"]))
    if not inter:
        return pd.DataFrame(columns=["company", "year"])

    rows = []
    for key in inter:
        company, year = key.rsplit("_", 1)
        rows.append({"company": company, "year": int(year)})

    out = pd.DataFrame(rows).sort_values(["year", "company"]).reset_index(drop=True)
    return out


def _company_year_warning(df):
    counts = df.groupby("company")["year"].apply(_continuous_years_count).reset_index(name="continuous_years")
    short = counts[counts["continuous_years"] < 3]
    return short


def _peer_group_heatmap_source(df):
    out = df.copy()
    out["size_metric"] = np.log1p(out["total_assets"].clip(lower=0))
    out["growth_metric"] = out.groupby("company")["sales"].pct_change()
    out["profit_metric"] = out["net_income"] / out["sales"].replace(0, np.nan)

    for c in ["size_metric", "growth_metric", "profit_metric", "ar_to_sales", "inv_to_sales", "tata", "ocf_to_ni"]:
        out[c] = out[c].replace([np.inf, -np.inf], np.nan)

    return out


def _peer_group_for_focus(df, focus_company, focus_year, focus_industry, k=5):
    sub = df[(df["year"] == focus_year) & (df["industry"] == focus_industry)].copy()
    sub = sub.dropna(subset=["company"])
    if sub.empty:
        return pd.DataFrame()

    metric_cols = ["size_metric", "growth_metric", "profit_metric"]
    for c in metric_cols:
        sub[c + "_z"] = _robust_zscore(sub[c])

    focus = sub[sub["company"] == focus_company].copy()
    if focus.empty:
        return pd.DataFrame()

    focus_row = focus.iloc[0]
    dist = (
        (sub["size_metric_z"] - focus_row["size_metric_z"]) ** 2
        + (sub["growth_metric_z"] - focus_row["growth_metric_z"]) ** 2
        + (sub["profit_metric_z"] - focus_row["profit_metric_z"]) ** 2
    ) ** 0.5
    sub["distance"] = dist

    k = max(3, min(int(k), min(10, len(sub))))
    peers = sub.sort_values("distance").head(k).copy()

    heat_cols = ["ar_to_sales", "inv_to_sales", "tata", "ocf_to_ni"]
    for c in heat_cols:
        peers[c + "_z"] = _robust_zscore(peers[c])

    heat = peers[["company"] + [c + "_z" for c in heat_cols]].set_index("company")
    heat.columns = ["매출채권/매출", "재고자산/매출", "TATA", "OCF/순이익"]
    return heat


uploaded = st.file_uploader("CSV 또는 XLSX 파일 업로드", type=["csv", "xlsx", "xls"])

with st.sidebar:
    st.header("설정")
    group_mode_label = st.radio(
        "그룹 표준화 기준",
        ["연도", "연도+산업", "전체"],
        index=1
    )

    group_mode = {
        "연도": "year",
        "연도+산업": "year_industry",
        "전체": "all"
    }[group_mode_label]

    contamination = st.slider(
        "Isolation Forest 이상치 비율",
        min_value=0.01,
        max_value=0.30,
        value=0.10,
        step=0.01
    )

    top_n = st.slider(
        "Top-N",
        min_value=3,
        max_value=30,
        value=10,
        step=1
    )

    w_beneish = st.slider("Beneish 가중치", 0.0, 3.0, 1.0, 0.1)
    w_iforest = st.slider("Isolation Forest 가중치", 0.0, 3.0, 1.0, 0.1)
    w_benford = st.slider("Benford 가중치", 0.0, 3.0, 1.0, 0.1)

st.markdown(
    """
필수 항목 예시: 회사명, 결산연도, 업종, 매출액, 매출채권, 재고자산, 자산총계, 영업활동현금흐름, 당기순이익

권장:
- 기업별 3개년 이상 데이터
- 업종 컬럼 포함
- 숫자형 데이터는 쉼표 포함 가능
"""
)

if uploaded is not None:
    try:
        df_raw = _read_uploaded_file(uploaded)
        df = _ensure_columns(df_raw)

        short_companies = _company_year_warning(df)
        if not short_companies.empty:
            st.warning("일부 기업은 연속 3년 미만 데이터입니다. 시계열 해석에 주의하세요.")

        result_df, benford_detail, baseline_df = _run_pipeline(
            df=df,
            group_mode=group_mode,
            contamination=contamination,
            w_beneish=w_beneish,
            w_iforest=w_iforest,
            w_benford=w_benford
        )

        tab1, tab2, tab3 = st.tabs([
            "🔍 Top-N & 일관 의심 기업",
            "🌡️ 동종 그룹 열지도",
            "📊 Benford 진단"
        ])

        with tab1:
            st.subheader("Top-N 의심 기업")
            top_table = _top_n_table(result_df, top_n)
            st.dataframe(top_table, use_container_width=True)

            st.subheader("일관 의심 기업")
            consistent = _consistent_suspects(result_df, baseline_df, top_n)
            if consistent.empty:
                st.info("현재 설정과 기준 설정(오염도 0.10, 가중치 1:1:1) 사이에 공통으로 포함된 기업이 없습니다.")
            else:
                st.dataframe(consistent, use_container_width=True)

        with tab2:
            st.subheader("동종 그룹 열지도")
            heat_source = _peer_group_heatmap_source(result_df)

            years = sorted(heat_source["year"].dropna().unique().tolist())
            industries = sorted(heat_source["industry"].dropna().astype(str).unique().tolist())

            if years and industries:
                col1, col2 = st.columns(2)
                with col1:
                    sel_year = st.selectbox("연도 선택", years, index=len(years) - 1)
                with col2:
                    sel_industry = st.selectbox("업종 선택", industries, index=0)

                subset = heat_source[(heat_source["year"] == sel_year) & (heat_source["industry"] == sel_industry)].copy()

                if subset.empty:
                    st.info("선택한 연도/업종에 해당하는 데이터가 없습니다.")
                else:
                    companies = sorted(subset["company"].astype(str).unique().tolist())
                    focus_company = st.selectbox("기준 회사 선택", companies, index=0)
                    k_default = min(5, len(subset))
                    k_value = st.slider("비교 기업 수", 3, min(10, len(subset)), max(3, k_default))

                    heat = _peer_group_for_focus(
                        df=heat_source,
                        focus_company=focus_company,
                        focus_year=sel_year,
                        focus_industry=sel_industry,
                        k=k_value
                    )

                    if heat.empty:
                        st.info("열지도를 만들 수 없습니다.")
                    else:
                        fig, ax = plt.subplots(figsize=(8, max(3, len(heat) * 0.6)))
                        im = ax.imshow(heat.values, aspect="auto", cmap="coolwarm")
                        ax.set_xticks(np.arange(len(heat.columns)))
                        ax.set_xticklabels(heat.columns, rotation=30, ha="right")
                        ax.set_yticks(np.arange(len(heat.index)))
                        ax.set_yticklabels(heat.index)
                        ax.set_title("동종 그룹 내 지표 편차 (z-score)")
                        cbar = plt.colorbar(im, ax=ax)
                        cbar.ax.set_ylabel("z-score", rotation=90)
                        st.pyplot(fig, clear_figure=True)

                        st.dataframe(heat.round(3), use_container_width=True)

        with tab3:
            st.subheader("Benford 진단")

            years = sorted(result_df["year"].dropna().unique().tolist())
            industries = sorted(result_df["industry"].dropna().astype(str).unique().tolist())

            if years and industries:
                col1, col2 = st.columns(2)
                with col1:
                    ben_year = st.selectbox("진단 연도", years, index=len(years) - 1, key="ben_year")
                with col2:
                    ben_industry = st.selectbox("진단 업종", industries, index=0, key="ben_industry")

                stats = benford_detail.get((ben_year, ben_industry), None)

                if stats is None:
                    st.info("선택한 그룹의 Benford 진단 결과가 없습니다.")
                else:
                    if stats["applicable"]:
                        st.success(f"적용 가능: n={stats['n']}, span={stats['span']:.1f}, MAD={stats['mad']:.4f}")
                    else:
                        span_text = f"{stats['span']:.1f}" if not pd.isna(stats["span"]) else "계산 불가"
                        st.warning(f"적용 주의: n={stats['n']}, span={span_text}, MAD={stats['mad']:.4f}")
                        st.caption(f"사유: {stats['reason_text']}")

                    fig, ax = plt.subplots(figsize=(8, 4))
                    x = np.arange(1, 10)
                    width = 0.38
                    ax.bar(x - width / 2, stats["obs"], width=width, label="관측 비율")
                    ax.bar(x + width / 2, stats["exp"], width=width, label="Benford 기대 비율")
                    ax.set_xticks(x)
                    ax.set_xlabel("첫 자리 숫자")
                    ax.set_ylabel("비율")
                    ax.set_title("Benford 분포 비교")
                    ax.legend()
                    st.pyplot(fig, clear_figure=True)

                    benford_df = pd.DataFrame({
                        "digit": stats["digits"],
                        "observed": stats["obs"],
                        "expected": stats["exp"]
                    })
                    st.dataframe(benford_df.round(4), use_container_width=True)

    except Exception as e:
        st.error(f"처리 중 오류가 발생했습니다: {e}")
else:
    st.info("분석할 파일을 업로드해 주세요.")
