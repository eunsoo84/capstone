import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest

st.set_page_config(
    page_title="회계 이상 탐지 대시보드 · 강화판",
    layout="wide",
)


def reset_session_for_new_file(filename: str):
    st.session_state["uploaded_name"] = filename
    st.session_state["base_top_ids"] = None
    st.session_state["base_params"] = None


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    aliases = {
        "company": ["company", "회사명", "법인명"],
        "year": ["year", "결산연도", "연도"],
        "industry": ["industry", "업종","산업"],
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


def _compute_benford_for_dataset(df: pd.DataFrame) -> dict:
    vals = df["sales"].astype(float).abs()
    vals = vals[vals > 0]

    if len(vals) == 0:
        return {
            "obs": None,
            "exp": None,
            "mad": None,
            "n": 0,
            "span": 0.0,
            "applicable": False,
            "reason": "매출 값이 없습니다.",
        }

    first_digits = []
    for v in vals:
        s = str(int(round(v)))
        s = s.lstrip("0")
        if not s:
            continue
        d = s[0]
        if d in "123456789":
            first_digits.append(int(d))

    n = len(first_digits)
    if n == 0:
        return {
            "obs": None,
            "exp": None,
            "mad": None,
            "n": 0,
            "span": float(vals.max() / max(vals.min(), 1e-9)),
            "applicable": False,
            "reason": "선두 자릿수를 계산할 수 있는 데이터가 부족합니다.",
        }

    counts = pd.value_counts(first_digits).reindex(range(1, 10), fill_value=0)
    obs = (counts / counts.sum()).values

    digits = np.arange(1, 10)
    exp = np.log10(1 + 1 / digits)

    mad = float(np.mean(np.abs(obs - exp)))
    span = float(vals.max() / max(vals.min(), 1e-9))

    applicable = True
    reason = "기본 표본/범위 기준을 충족합니다."
    if n < 100:
        applicable = False
        reason = f"표본 수(n={n})가 충분하지 않습니다(권장 100개 이상)."
    elif span < 100:
        applicable = False
        reason = f"금액 범위가 좁습니다(최대/최소 비율≈{span:.1f}, 권장 ≥ 100)."

    return {
        "obs": obs.tolist(),
        "exp": exp.tolist(),
        "mad": mad,
        "n": n,
        "span": span,
        "applicable": applicable,
        "reason": reason,
    }


def run_pipeline(
    df_raw: pd.DataFrame,
    group_mode: str = "year_industry",
    contamination: float = 0.10,
    w_beneish: float = 1.0,
    w_iso: float = 1.0,
    w_benford: float = 1.0,
):
    df = _ensure_columns(df_raw)

    for col in ["sales", "ar", "inventory", "total_assets", "ocf", "net_income"]:
        s = df[col].astype(str)
        s = s.str.replace(",", "", regex=False)
        s = s.str.replace(" ", "", regex=False)
        s = s.str.replace("\u00a0", "", regex=False)
        df[col] = pd.to_numeric(s, errors="coerce")

    df = df.reset_index(drop=True)
    df["row_id"] = df.index + 1

    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    eps = 1e-9

    df["ar_to_sales"] = df["ar"] / (df["sales"] + eps)
    df["inv_to_sales"] = df["inventory"] / (df["sales"] + eps)
    df["ocf_to_ni"] = df["ocf"] / (df["net_income"] + eps)
    df["tata"] = (df["net_income"] - df["ocf"]) / (df["total_assets"] + eps)


    df = df.sort_values(["company", "year"])
    df["sales_yoy"] = (
        df.groupby("company")["sales"].pct_change().fillna(0.0) * 100.0
    )

    metrics = ["ar_to_sales", "inv_to_sales", "tata", "ocf_to_ni"]

    def zscore_group(g: pd.DataFrame, cols: list):
        g = g.copy()
        for c in cols:
            col_name = str(c)
            m = g[col_name].mean()
            s = g[col_name].std(ddof=0)
            if s is None or s == 0 or np.isnan(s):
                g[col_name + "_z"] = 0.0
            else:
                g[col_name + "_z"] = (g[col_name] - m) / s
        return g

    if group_mode == "year":
        df = df.groupby("year", group_keys=False).apply(zscore_group, cols=metrics)
    elif group_mode == "year_industry":
        df = (
            df.groupby(["year", "industry"], group_keys=False)
            .apply(zscore_group, cols=metrics)
        )
    else:
        df = zscore_group(df, metrics)

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
        iso_norm = (iso_raw - iso_raw.min()) / (iso_raw.max() - iso_raw.min() + eps)
    except Exception:
        iso_norm = np.zeros(df.shape[0])

    df["iso_score"] = iso_norm

    benford_info = _compute_benford_for_dataset(df)
    benford_applicable = benford_info["applicable"]
    benford_reason = benford_info["reason"]
    benford_overall = {
        "obs": benford_info["obs"],
        "exp": benford_info["exp"],
        "mad": benford_info["mad"],
    }

    if benford_info["mad"] is not None:
        df["benford_mad"] = float(benford_info["mad"])
    else:
        df["benford_mad"] = np.nan

    m = df["mscore_raw"].fillna(0.0).values
    m_norm = (m - m.min()) / (m.max() - m.min() + eps)

    ben_used = bool(
        benford_applicable and w_benford > 0 and benford_info["mad"] is not None
    )
    if ben_used:
        b = df["benford_mad"].fillna(0.0).values
        b_norm = (b - b.min()) / (b.max() - b.min() + eps)
    else:
        b_norm = np.zeros(df.shape[0])

    flag_score = (
        w_beneish * m_norm
        + w_iso * df["iso_score"].values
        + w_benford * b_norm
    )

    df["flag_score"] = flag_score

    df_scored = df.sort_values("flag_score", ascending=False).reset_index(drop=True)
    df_scored["rank"] = np.arange(1, len(df_scored) + 1)

    meta = {
        "benford_applicable": benford_applicable,
        "benford_reason": benford_reason,
        "benford_overall": benford_overall,
        "benford_n": benford_info["n"],
        "benford_span": benford_info["span"],
        "benford_used_in_score": ben_used,
    }

    return df_scored, meta


st.sidebar.header("옵션")

group_mode = st.sidebar.radio(
    "그룹 표준화 기준",
    ["연도", "연도+산업", "전체"],
    help="연도/산업별로 지표를 표준화해 업종·규모 차이에서 오는 왜곡을 줄입니다.",
)

if group_mode == "연도":
    group_mode_key = "year"
elif group_mode == "연도+산업":
    group_mode_key = "year_industry"
else:
    group_mode_key = "all"

contamination = st.sidebar.slider(
    "탐지 민감도(의심 비율, ISO contamination)",
    min_value=0.01,
    max_value=0.30,
    value=0.10,
    step=0.01,
    help="Isolation Forest에서 이상치로 볼 비율입니다. 높일수록 더 많은 회사를 의심으로 잡습니다.",
)

top_n = st.sidebar.slider(
    "Top-N(의심 후보 수)",
    min_value=3,
    max_value=30,
    value=10,
    step=1,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**가중치 설정**")

w_beneish = st.sidebar.slider(
    "Beneish 비중",
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
w_benford = st.sidebar.slider(
    "Benford 비중",
    min_value=0.0,
    max_value=3.0,
    value=1.0,
    step=0.1,
)

st.sidebar.markdown("---")
st.sidebar.info(
    "📎 필수 항목: 회사명, 결산연도, 매출액, 매출원가, 판매관리비, 영업이익, 감가상각비, "
    "매출채권, 재고자산, 자산총계, 부채총계, 영업활동현금흐름, 당기순이익, 업종(권장)"
)

st.title("회계 이상 탐지 대시보드 · 강화판")

st.markdown(
    """
1. 아래에 CSV/엑셀 파일을 업로드하세요.  
2. 필수 항목이 들어있어야 합니다.  
3. 왼쪽에서 **탐지 민감도(의심 비율)**와 **가중치**를 조정하며 Top-N 변화를 확인합니다.  
4. 하단 탭에서  
   - 🔍 **Top-N 의심 리스트 & 일관 의심 기업**,  
   - 🌡️ **동종 그룹 열지도(비슷한 회사끼리 비교)**,  
   - 📊 **Benford 사용 가능성 진단**  
   을 볼 수 있습니다.
"""
)

uploaded = st.file_uploader("CSV 또는 Excel 업로드", type=["csv", "xlsx"])

if uploaded is None:
    st.stop()

if "uploaded_name" not in st.session_state or st.session_state["uploaded_name"] != uploaded.name:
    reset_session_for_new_file(uploaded.name)

if uploaded.name.lower().endswith(".csv"):
    df_raw = pd.read_csv(uploaded)
else:
    df_raw = pd.read_excel(uploaded)

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
        w_benford=w_benford,
    )
except Exception as e:
    st.error(f"⚠️ 처리 중 오류가 발생했습니다: {e}")
    st.stop()

df_top = df_scored.head(top_n).copy()

base_params = {
    "group_mode": group_mode_key,
    "contamination": 0.10,
    "w_beneish": 1.0,
    "w_iso": 1.0,
    "w_benford": 1.0,
}

if st.session_state.get("base_top_ids") is None:
    base_df, _ = run_pipeline(
        df_raw,
        group_mode=base_params["group_mode"],
        contamination=base_params["contamination"],
        w_beneish=base_params["w_beneish"],
        w_iso=base_params["w_iso"],
        w_benford=base_params["w_benford"],
    )
    base_top = base_df.head(top_n).copy()
    st.session_state["base_top_ids"] = set(base_top["row_id"].tolist())
    st.session_state["base_params"] = base_params

current_ids = set(df_top["row_id"].tolist())
stable_ids = current_ids.intersection(st.session_state["base_top_ids"])
stable_df = df_top[df_top["row_id"].isin(stable_ids)].copy()

tab1, tab2, tab3 = st.tabs(
    ["🔍 Top-N & 일관 의심 기업", "🌡️ 동종 그룹 열지도", "📊 Benford 진단"]
)

with tab1:
    st.subheader("의심 후보 Top-N")

    show_cols = [
        "rank",
        "company",
        "year",
        "industry",
        "flag_score",
        "mscore_raw",
        "iso_score",
        "benford_mad",
        "ar_to_sales",
        "inv_to_sales",
        "ocf_to_ni",
    ]
    show_cols = [c for c in show_cols if c in df_top.columns]

    st.dataframe(
        df_top[show_cols],
        use_container_width=True,
        height=360,
    )

    st.markdown("---")
    st.markdown("#### 🎯 설정을 바꿔도 계속 남는 ‘일관 의심 기업’")

    if len(stable_df) == 0:
        st.info(
            "현재 파라미터에서는 기준 설정 시점의 Top-N과 겹치는 의심 기업이 없습니다. "
            "오염 비율·가중치를 조금 조정해보세요."
        )
    else:
        st.caption(
            "※ 기준: 업로드 당시 **기본 설정(오염비율 0.10, 가중치 1:1:1)** 로 계산한 Top-N과, "
            "현재 설정 Top-N에 모두 포함된 기업/연도 조합입니다."
        )
        st.dataframe(
            stable_df[show_cols],
            use_container_width=True,
            height=260,
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

        if subset.empty:
            st.warning("해당 연도·산업 조합에 데이터가 없습니다.")
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

                k = st.slider(
                    "동종 그룹 크기 (기준 회사 포함)",
                    min_value=3,
                    max_value=min(10, subset.shape[0]),
                    value=min(5, subset.shape[0]),
                )

                peer = subset.nsmallest(k, "peer_dist").copy()

                st.caption(
                    "※ 같은 연도·산업 내에서 자산 규모, 매출 성장률, 이익률이 비슷한 회사를 동종 그룹으로 구성했습니다."
                )

                metrics = [
                    "ar_to_sales",
                    "inv_to_sales",
                    "tata",
                    "ocf_to_ni",
                    "mscore_raw",
                    "iso_score",
                ]
                metrics = [m for m in metrics if m in peer.columns]

                if len(metrics) == 0:
                    st.info("열지도로 보여줄 지표가 없습니다.")
                else:
                    peer_z = peer.copy()
                    for m in metrics:
                        col_name = str(m)
                        mm = peer[col_name].mean()
                        ss = peer[col_name].std(ddof=0) or 1e-9
                        peer_z[col_name + "_z_peer"] = (peer[col_name] - mm) / ss

                    z_cols = [str(m) + "_z_peer" for m in metrics]
                    z_vals = peer_z[z_cols].values
                    labels = [
                        f"{r['company']}_{int(r['year'])}"
                        for _, r in peer.iterrows()
                    ]

                    fig, ax = plt.subplots(
                        figsize=(1.2 * len(metrics), 0.5 * len(peer) + 1)
                    )
                    im = ax.imshow(z_vals, aspect="auto", cmap="coolwarm")

                    ax.set_xticks(np.arange(len(metrics)))
                    ax.set_xticklabels(metrics, rotation=45, ha="right")
                    ax.set_yticks(np.arange(len(labels)))
                    ax.set_yticklabels(labels)

                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    ax.set_title("동종 그룹 내 지표 편차 (z-score)")
                    st.pyplot(fig)

                    st.caption(
                        "색이 붉을수록 동종 평균보다 높고, 푸를수록 낮습니다."
                    )

with tab3:
    st.subheader("Benford 법칙 사용 가능성 진단")

    ben_ok = meta.get("benford_applicable", False)
    reason = meta.get("benford_reason", "")
    dist = meta.get("benford_overall", {})

    if ben_ok:
        st.success(
            f"이 데이터 집합은 Benford 법칙을 적용하기에 대체로 적절합니다. "
            f"(표본 수 n={meta.get('benford_n')}, "
            f"최대/최소 비율≈{meta.get('benford_span'):.1f})"
        )
    else:
        st.warning(
            "⚠️ 해당 데이터는 Benford 법칙을 적용하기에 적절하지 않을 수 있습니다.\n\n"
            f"사유: {reason}"
        )

    if dist:
        obs = dist.get("obs")
        exp = dist.get("exp")
        if obs is not None and exp is not None and len(obs) == 9:
            digits = np.arange(1, 10)
            width = 0.35

            fig, ax = plt.subplots()
            ax.bar(digits - width / 2, exp, width, label="이론(베니포드)")
            ax.bar(digits + width / 2, obs, width, label="실제(매출)")
            ax.set_xticks(digits)
            ax.set_xlabel("선두 자릿수")
            ax.set_ylabel("비율")
            ax.set_title(
                f"선두 자릿수 분포 비교 (MAD={dist.get('mad', np.nan):.4f})"
            )
            ax.legend()
            st.pyplot(fig)

            st.caption(
                "※ 그래프는 전체 매출 데이터를 한 번에 모아, 선두 숫자 분포가 "
                "이론적 Benford 분포와 얼마나 다른지 보여줍니다. "
                "표본 수가 적거나 금액 범위가 좁으면 신뢰도가 떨어질 수 있습니다."
            )
        else:
            st.info("Benford 분포를 그릴 수 있는 데이터가 부족합니다.")

    st.markdown("---")
    st.markdown(
        f"Benford 결과가 최종 점수에 실제 반영되었는지 여부: "
        f"**{'예' if meta.get('benford_used_in_score') else '아니오(가중치 0으로 처리)' }**"
    )
