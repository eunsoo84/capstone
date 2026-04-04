import io
import os

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest

font_path = "NotoSansKR-Regular.ttf"
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    rcParams["font.family"] = "Noto Sans KR"
    rcParams["axes.unicode_minus"] = False

st.set_page_config(
    page_title="부정회계 탐지 스크리닝",
    layout="wide",
)


def reset_session_for_new_file(filename: str):
    st.session_state["uploaded_name"] = filename
    st.session_state["base_top_ids"] = None
    st.session_state["base_params"] = None


def _normalize_column_name(col):
    col = str(col)
    col = col.replace("\ufeff", "")
    col = col.replace("\u00a0", " ")
    col = col.strip()
    col = " ".join(col.split())
    return col


def _normalized_key(text):
    text = _normalize_column_name(text)
    text = text.lower().replace(" ", "").replace("_", "")
    return text


def _read_uploaded_file(uploaded):
    name = uploaded.name.lower()

    if name.endswith(".csv"):
        raw = uploaded.getvalue()
        for enc in ["utf-8-sig", "cp949", "utf-8"]:
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=enc)
            except Exception:
                pass
        return pd.read_csv(io.BytesIO(raw))

    return pd.read_excel(uploaded)


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_normalize_column_name(c) for c in df.columns]

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

    current_lookup = {_normalized_key(c): c for c in df.columns}

    col_map = {}
    for canonical, cands in aliases.items():
        for c in cands:
            key = _normalized_key(c)
            if key in current_lookup:
                col_map[current_lookup[key]] = canonical
                break

    df = df.rename(columns=col_map)

    required = [
        "company",
        "year",
        "sales",
        "ar",
        "inventory",
        "total_assets",
        "ocf",
        "net_income",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"필수 컬럼이 누락되었습니다: {missing}. "
            f"현재 컬럼: {list(df.columns)}"
        )

    if "industry" not in df.columns:
        df["industry"] = "미지정"

    return df


def _clean_numeric_cols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        s = df[col].astype(str)
        s = s.str.replace(",", "", regex=False)
        s = s.str.replace(" ", "", regex=False)
        s = s.str.replace("\u00a0", "", regex=False)
        df[col] = pd.to_numeric(s, errors="coerce")
    return df


def _norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    return (x - mn) / (mx - mn + 1e-9)


def _safe_zscore(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    m = series.mean()
    s = series.std(ddof=0)
    if s is None or s == 0 or np.isnan(s):
        return pd.Series(0.0, index=series.index)
    return (series - m) / s


def _metric_label_map():
    return {
        "ar_to_sales": "AR/Sales(매출채권/매출액)",
        "inv_to_sales": "Inv/Sales(재고자산/매출액)",
        "tata": "TATA((NI-OCF)/자산)",
        "ocf_to_ni": "OCF/NI(영업CF/순이익)",
        "mscore_raw": "mscore_raw(간이 Beneish)",
        "iso_score": "iso_score(ISO 이상치)",
        "change_score": "change_score(전년대비 변화)",
    }


def run_pipeline(
    df_raw: pd.DataFrame,
    group_mode: str = "year_industry",
    contamination: float = 0.10,
    w_beneish: float = 1.0,
    w_iso: float = 1.0,
    w_change: float = 1.0,
):
    df = _ensure_columns(df_raw)

    df = _clean_numeric_cols(
        df,
        ["sales", "ar", "inventory", "total_assets", "ocf", "net_income"],
    )

    df = df.reset_index(drop=True)
    df["row_id"] = df.index + 1
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    df = df.dropna(
        subset=[
            "company",
            "year",
            "sales",
            "ar",
            "inventory",
            "total_assets",
            "ocf",
            "net_income",
        ]
    ).copy()

    df["company"] = df["company"].astype(str).str.strip()
    df["industry"] = df["industry"].astype(str).str.strip().replace("", "미지정")
    df["year"] = df["year"].astype(int)

    eps = 1e-9

    df["ar_to_sales"] = df["ar"] / (df["sales"] + eps)
    df["inv_to_sales"] = df["inventory"] / (df["sales"] + eps)
    df["ocf_to_ni"] = df["ocf"] / (df["net_income"] + eps)
    df["tata"] = (df["net_income"] - df["ocf"]) / (df["total_assets"] + eps)

    df = df.sort_values(["company", "year"]).reset_index(drop=True)
    df["sales_yoy"] = (
        df.groupby("company")["sales"].pct_change().fillna(0.0) * 100.0
    )

    metrics = ["ar_to_sales", "inv_to_sales", "tata", "ocf_to_ni"]

    for c in metrics:
        if group_mode == "year_industry":
            df[c + "_z"] = df.groupby(["year", "industry"])[c].transform(_safe_zscore)
        else:
            df[c + "_z"] = _safe_zscore(df[c])

    z_ar = df.get("ar_to_sales_z", pd.Series(0, index=df.index))
    z_inv = df.get("inv_to_sales_z", pd.Series(0, index=df.index))
    z_tata = df.get("tata_z", pd.Series(0, index=df.index))
    z_ocf = df.get("ocf_to_ni_z", pd.Series(0, index=df.index))

    df["mscore_raw"] = z_ar + z_inv + z_tata - z_ocf

    iso_features = ["ar_to_sales", "inv_to_sales", "tata", "ocf_to_ni"]
    X = df[iso_features].fillna(0.0).values

    try:
        iso = IsolationForest(
            contamination=contamination,
            random_state=42,
        )
        iso.fit(X)
        iso_raw = -iso.decision_function(X)
        iso_raw = np.array(iso_raw)
        iso_norm = _norm01(iso_raw)
    except Exception:
        iso_norm = np.zeros(df.shape[0])

    df["iso_score"] = iso_norm

    df = df.sort_values(["company", "year"]).reset_index(drop=True)
    for m in metrics:
        df[m + "_d1"] = df.groupby("company")[m].diff().fillna(0.0)

    delta_cols = [m + "_d1" for m in metrics]

    for c in delta_cols:
        if group_mode == "year_industry":
            df[c + "_z"] = df.groupby(["year", "industry"])[c].transform(_safe_zscore)
        else:
            df[c + "_z"] = _safe_zscore(df[c])

    change_raw = np.zeros(df.shape[0], dtype=float)
    for m in metrics:
        change_raw += np.abs(df[m + "_d1_z"].fillna(0.0).values)

    df["change_score"] = _norm01(change_raw)

    m = df["mscore_raw"].fillna(0.0).values
    m_norm = _norm01(m)
    df["mscore_norm"] = m_norm

    df["score_beneish_part"] = w_beneish * df["mscore_norm"].values
    df["score_iso_part"] = w_iso * df["iso_score"].values
    df["score_change_part"] = w_change * df["change_score"].values

    df["flag_score"] = (
        df["score_beneish_part"].values
        + df["score_iso_part"].values
        + df["score_change_part"].values
    )

    label_map = _metric_label_map()
    reason_pool = [
        ("ar_to_sales_z", label_map["ar_to_sales"]),
        ("inv_to_sales_z", label_map["inv_to_sales"]),
        ("tata_z", label_map["tata"]),
        ("ocf_to_ni_z", label_map["ocf_to_ni"]),
        ("ar_to_sales_d1_z", "Δ " + label_map["ar_to_sales"]),
        ("inv_to_sales_d1_z", "Δ " + label_map["inv_to_sales"]),
        ("tata_d1_z", "Δ " + label_map["tata"]),
        ("ocf_to_ni_d1_z", "Δ " + label_map["ocf_to_ni"]),
    ]

    def pick_reasons(row: pd.Series):
        scores = []
        for col, label in reason_pool:
            v = row.get(col, 0.0)
            if pd.isna(v):
                v = 0.0
            scores.append((abs(float(v)), float(v), label))
        scores.sort(key=lambda x: x[0], reverse=True)
        top3 = scores[:3]
        r1 = f"{top3[0][2]}: {top3[0][1]:+.2f}"
        r2 = f"{top3[1][2]}: {top3[1][1]:+.2f}"
        r3 = f"{top3[2][2]}: {top3[2][1]:+.2f}"
        return pd.Series([r1, r2, r3])

    df[["reason_1", "reason_2", "reason_3"]] = df.apply(pick_reasons, axis=1)

    df_scored = df.sort_values("flag_score", ascending=False).reset_index(drop=True)
    df_scored["rank"] = np.arange(1, len(df_scored) + 1)

    meta = {
        "label_map": label_map,
    }

    return df_scored, meta


st.sidebar.header("옵션")

group_mode = st.sidebar.radio(
    "그룹 표준화 기준",
    ["연도+산업", "전체"],
)

if group_mode == "연도+산업":
    group_mode_key = "year_industry"
else:
    group_mode_key = "all"

contamination = st.sidebar.slider(
    "탐지 민감도(의심 비율, ISO contamination)",
    min_value=0.01,
    max_value=0.30,
    value=0.10,
    step=0.01,
)

top_n = st.sidebar.slider(
    "Top-N(표시 개수)",
    min_value=3,
    max_value=50,
    value=10,
    step=1,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**가중치 설정**")

w_beneish = st.sidebar.slider(
    "Beneish(간이) 비중",
    min_value=0.0,
    max_value=3.0,
    value=1.0,
    step=0.1,
)
w_iso = st.sidebar.slider(
    "Isolation Forest 비중",
    min_value=0.0,
    max_value=3.0,
    value=1.0,
    step=0.1,
)
w_change = st.sidebar.slider(
    "전년대비 변화(Δ) 비중",
    min_value=0.0,
    max_value=3.0,
    value=1.0,
    step=0.1,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**AND 고정 + 상위 % 컷(기본 0.80) 고정**")

p_flag = 0.80
p_beneish = 0.80
p_iso = 0.80
p_change = 0.80

st.sidebar.write(f"최종 점수(flag_score) 상위 {(1 - p_flag) * 100:.0f}% 컷")
st.sidebar.write(f"Beneish 상위 {(1 - p_beneish) * 100:.0f}% 컷")
st.sidebar.write(f"ISO 상위 {(1 - p_iso) * 100:.0f}% 컷")
st.sidebar.write(f"변화(Δ) 상위 {(1 - p_change) * 100:.0f}% 컷")

st.sidebar.markdown("---")
st.sidebar.info(
    "📎 필수 항목: 회사명, 결산연도, 매출액, 매출채권, 재고자산, 자산총계, 영업활동현금흐름, 당기순이익, 업종(권장)"
)

st.title("부정회계 탐지 스크리닝")

st.markdown(
    """
1. 아래에 CSV/엑셀 파일을 업로드하세요.  
2. 필수 항목이 들어있어야 합니다. (회사명, 결산연도, 매출액, 매출채권, 재고자산, 자산총계, 영업활동현금흐름, 당기순이익, 업종(권장))  
3. 왼쪽에서 **탐지 민감도(ISO contamination)** 와 **가중치(Beneish/ISO/Δ)** 를 조정하며 결과 변화를 확인합니다.  

4. 하단 탭에서   
   - 🔍 **Top-N 의심 리스트 & Top 1~3 적발 사유**,   
   - 🌡️ **동종 그룹 열지도(동종 업계끼리 비교)**  
   를 확인할 수 있습니다.
"""
)

uploaded = st.file_uploader("CSV 또는 Excel 업로드", type=["csv", "xlsx"])

if uploaded is None:
    st.stop()

if "uploaded_name" not in st.session_state or st.session_state["uploaded_name"] != uploaded.name:
    reset_session_for_new_file(uploaded.name)

df_raw = _read_uploaded_file(uploaded)

st.caption(f"업로드된 데이터 크기: {df_raw.shape[0]}행 × {df_raw.shape[1]}열")
with st.expander("원본 일부 미리보기", expanded=False):
    st.dataframe(df_raw.head())

try:
    df_scored, meta = run_pipeline(
        df_raw,
        group_mode=group_mode_key,
        contamination=contamination,
        w_beneish=w_beneish,
        w_iso=w_iso,
        w_change=w_change,
    )
except Exception as e:
    st.exception(e)
    st.stop()

thr_flag = float(df_scored["flag_score"].quantile(p_flag))
thr_b = float(df_scored["mscore_norm"].quantile(p_beneish))
thr_i = float(df_scored["iso_score"].quantile(p_iso))
thr_c = float(df_scored["change_score"].quantile(p_change))

mask = (
    (df_scored["flag_score"] >= thr_flag)
    & (df_scored["mscore_norm"] >= thr_b)
    & (df_scored["iso_score"] >= thr_i)
    & (df_scored["change_score"] >= thr_c)
)

df_candidates = (
    df_scored[mask]
    .copy()
    .sort_values("flag_score", ascending=False)
    .reset_index(drop=True)
)
df_candidates["rank"] = np.arange(1, len(df_candidates) + 1)

tab1, tab2, tab3 = st.tabs(
    ["🔍 Top-N & 사유", "🌡️ 동종 그룹 열지도", "🧾 지표 뜻(발표 대비)"]
)

with tab1:
    st.subheader("의심 후보 Top-N")

    st.caption(
        f"기준값: flag≥{thr_flag:.4f} / Beneish≥{thr_b:.4f} / ISO≥{thr_i:.4f} / Δ≥{thr_c:.4f} "
        f"(모두 AND) | 상위 컷: flag {int((1-p_flag)*100)}%, Beneish {int((1-p_beneish)*100)}%, ISO {int((1-p_iso)*100)}%, Δ {int((1-p_change)*100)}%"
    )

    if df_candidates.empty:
        st.info("현재 기준에서는 추가 점검 후보가 없습니다.")
    else:
        df_view = df_candidates.head(top_n).copy()
        show_cols = [
            "rank",
            "company",
            "year",
            "industry",
            "flag_score",
            "score_beneish_part",
            "score_iso_part",
            "score_change_part",
            "mscore_raw",
            "mscore_norm",
            "iso_score",
            "change_score",
            "ar_to_sales",
            "inv_to_sales",
            "tata",
            "ocf_to_ni",
            "reason_1",
            "reason_2",
            "reason_3",
        ]
        show_cols = [c for c in show_cols if c in df_view.columns]

        st.dataframe(
            df_view[show_cols],
            use_container_width=True,
            height=380,
        )

        top3 = df_view.head(min(3, len(df_view))).copy()
        st.markdown("---")
        st.subheader("Top 1~3 상세 사유(자동)")

        for _, r in top3.iterrows():
            st.markdown(
                f"**#{int(r['rank'])} {r['company']} ({int(r['year'])})**  \n"
                f"- 점수 분해: Beneish {float(r['score_beneish_part']):.3f} / ISO {float(r['score_iso_part']):.3f} / Δ {float(r['score_change_part']):.3f} / 합 {float(r['flag_score']):.3f}  \n"
                f"- Top3 요인: {r['reason_1']} · {r['reason_2']} · {r['reason_3']}"
            )

with tab2:
    st.subheader("동종 그룹 열지도 (비슷한 회사끼리 지표 비교)")

    if df_scored.empty:
        st.info("데이터가 없습니다.")
    else:
        years = sorted(df_scored["year"].dropna().unique())
        sel_year = st.selectbox("연도 선택", years, key="peer_year")

        industries = sorted(df_scored["industry"].dropna().unique())
        sel_ind = st.selectbox("산업 선택", industries, key="peer_ind")

        subset = df_scored[
            (df_scored["year"] == sel_year) & (df_scored["industry"] == sel_ind)
        ].copy()

        if subset.shape[0] < 3:
            st.warning("해당 연도·산업 조합에 데이터가 3개 미만이라 열지도를 만들 수 없습니다.")
        else:
            companies = subset["company"].unique().tolist()
            sel_comp = st.selectbox("기준 회사 선택", companies, key="peer_comp")

            focus = subset[subset["company"] == sel_comp].copy()
            if focus.empty:
                st.warning("선택한 회사 데이터가 없습니다.")
            else:
                eps = 1e-9
                subset["size_metric"] = np.log1p(subset["total_assets"])
                subset["growth_metric"] = subset["sales_yoy"].fillna(0.0)
                subset["profit_metric"] = (
                    subset["net_income"] / (subset["sales"] + eps)
                ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

                for c in ["size_metric", "growth_metric", "profit_metric"]:
                    m = subset[c].mean()
                    s = subset[c].std(ddof=0) or eps
                    subset[c + "_z"] = (subset[c] - m) / s

                focus = subset[subset["company"] == sel_comp].copy()
                focus_row = focus.iloc[0]

                f_vec = np.array(
                    [
                        float(focus_row["size_metric_z"]),
                        float(focus_row["growth_metric_z"]),
                        float(focus_row["profit_metric_z"]),
                    ]
                )

                subset["peer_dist"] = subset.apply(
                    lambda r: np.linalg.norm(
                        np.array(
                            [
                                r["size_metric_z"],
                                r["growth_metric_z"],
                                r["profit_metric_z"],
                            ]
                        )
                        - f_vec
                    ),
                    axis=1,
                )

                kmax = min(10, subset.shape[0])
                if kmax < 3:
                    st.info("열지도를 만들 수 없습니다.")
                else:
                    k = st.slider(
                        "동종 그룹 크기 (기준 회사 포함)",
                        min_value=3,
                        max_value=kmax,
                        value=min(5, subset.shape[0]),
                    )

                    peer = subset.nsmallest(k, "peer_dist").copy()

                    metrics = [
                        "ar_to_sales",
                        "inv_to_sales",
                        "tata",
                        "ocf_to_ni",
                        "mscore_raw",
                        "iso_score",
                        "change_score",
                    ]
                    metrics = [m for m in metrics if m in peer.columns]

                    if len(metrics) == 0:
                        st.info("열지도로 보여줄 지표가 없습니다.")
                    else:
                        peer_z = peer.copy()
                        for m in metrics:
                            mm = peer[m].mean()
                            ss = peer[m].std(ddof=0) or 1e-9
                            peer_z[m + "_z_peer"] = (peer[m] - mm) / ss

                        z_cols = [m + "_z_peer" for m in metrics]
                        z_vals = peer_z[z_cols].values
                        labels = [
                            f"{r['company']}_{int(r['year'])}"
                            for _, r in peer.iterrows()
                        ]

                        label_map = meta.get("label_map", {})
                        xlabels = [label_map.get(m, m) for m in metrics]

                        fig, ax = plt.subplots(
                            figsize=(1.25 * len(metrics), 0.55 * len(peer) + 1)
                        )
                        im = ax.imshow(z_vals, aspect="auto", cmap="coolwarm")

                        ax.set_xticks(np.arange(len(metrics)))
                        ax.set_xticklabels(xlabels, rotation=45, ha="right")
                        ax.set_yticks(np.arange(len(labels)))
                        ax.set_yticklabels(labels)

                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        ax.set_title("동종 그룹 내 지표 편차 (z-score)")
                        st.pyplot(fig)

                        st.caption("색이 붉을수록 동종 평균보다 높고, 푸를수록 낮습니다.")

with tab3:
    st.subheader("지표 뜻(발표 대비)")

    st.markdown(
        """
- AR/Sales(매출채권/매출액): 매출 대비 채권이 과도하면 매출 인식/회수 지연 가능성 신호
- Inv/Sales(재고자산/매출액): 매출 대비 재고가 과도하면 재고 과대계상/판매부진 가능성 신호
- TATA((NI-OCF)/자산): 이익은 큰데 현금이 안 따라오면 발생주의(비현금) 조정이 커진 신호
- OCF/NI(영업CF/순이익): 순이익 대비 영업현금흐름이 낮거나 음수면 이익의 질 저하 신호

- mscore_raw(간이 Beneish): 동종(연도/산업) 내에서 위 지표들을 표준화(z)한 뒤
  mscore_raw = z(AR/Sales) + z(Inv/Sales) + z(TATA) − z(OCF/NI)

- iso_score(ISO 이상치): 4개 지표 조합이 다변량 관점에서 “특이한 조합”인지 Isolation Forest로 점수화

- change_score(전년대비 변화): 같은 회사의 전년 대비 지표 변화(Δ)가 동종 대비 얼마나 급격한지 점수화
"""
    )
