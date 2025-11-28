
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="회계 이상 탐지 대시보드 · 강화판",
    layout="wide",
)

def reset_session_for_new_file(filename: str):
    st.session_state["uploaded_name"] = filename
    st.session_state["base_top_ids"] = None
    st.session_state["base_params"] = None


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
st.sidebar.info("📎 필수 항목: 회사명, 결산연도, 매출액, 매출원가, 판매관리비, 영업이익, 감가상각비, 매출채권, "
                "재고자산, 자산총계, 부채총계, 영업활동현금흐름, 당기순이익, 산업 종류(권장)")


st.title("회계 이상 탐지 대시보드 · 강화판")

st.markdown(
    """

1. 아래에 CSV/엑셀 파일을 업로드하세요.  
2. 필수 항목(회사명, 매출원가, 판매관리비, 영업이익, 감가상각비, 매출채권, 재고자산, 자산총계, 부채총계, 영업활동현금흐름, 당기순이익)이 들어있어야 합니다.  
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
     
        years = sorted(df_scored["year"].unique())
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
                focus_row = focus.iloc[0]

               
                eps = 1e-9
                subset["size_metric"] = np.log1p(subset["total_assets"])
                subset["growth_metric"] = subset["sales_yoy"].fillna(0.0)
                subset["profit_metric"] = (
                    subset["net_income"] / (subset["sales"] + eps)
                ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

                # z-score
                for c in ["size_metric", "growth_metric", "profit_metric"]:
                    m = subset[c].mean()
                    s = subset[c].std(ddof=0) or eps
                    subset[c + "_z"] = (subset[c] - m) / s

              
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
                    "※ 같은 연도·산업 내에서 **자산 규모, 매출 성장률, 이익률**이 비슷한 회사를 "
                    "동종 그룹으로 구성했습니다."
                )

                metrics = ["ar_to_sales", "inv_to_sales", "tata", "ocf_to_ni", "mscore_raw", "iso_score"]
                metrics = [m for m in metrics if m in peer.columns]

                if len(metrics) == 0:
                    st.info("열지도로 보여줄 지표가 없습니다.")
                else:
                  
                    mat = []
                    labels = []
                    for _, r in peer.iterrows():
                        labels.append(f"{r['company']}_{int(r['year'])}")
                    peer_z = peer.copy()
                    for m in metrics:
                        col = []
                        mm = peer[m].mean()
                        ss = peer[m].std(ddof=0) or 1e-9
                        peer_z[m + "_z_peer"] = (peer[m] - mm) / ss
                        col.append(m + "_z_peer")
                    z_cols = [m + "_z_peer" for m in metrics]
                    z_vals = peer_z[z_cols].values

                    fig, ax = plt.subplots(figsize=(1.2 * len(metrics), 0.5 * len(peer) + 1))
                    im = ax.imshow(z_vals, aspect="auto", cmap="coolwarm")

                    ax.set_xticks(np.arange(len(metrics)))
                    ax.set_xticklabels(metrics, rotation=45, ha="right")
                    ax.set_yticks(np.arange(len(labels)))
                    ax.set_yticklabels(labels)

                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    ax.set_title("동종 그룹 내 지표 편차 (z-score)")
                    st.pyplot(fig)

                    st.caption(
                        "색이 **붉을수록 동종 평균보다 높고**, **푸를수록 낮습니다.** "
                        "예: 매출채권/재고/TATA가 붉게 튀는 기업은 해당 지표가 또래 대비 과도할 수 있습니다."
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
            "⚠️ 본 데이터 집합은 Benford 법칙을 적용하기에 적절하지 않을 수 있습니다.\n\n"
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
            ax.bar(digits + width / 2, obs, width, label="실제(매출+원가)")
            ax.set_xticks(digits)
            ax.set_xlabel("선두 자릿수")
            ax.set_ylabel("비율")
            ax.set_title(f"선두 자릿수 분포 비교 (MAD={dist.get('mad', np.nan):.4f})")
            ax.legend()
            st.pyplot(fig)

            st.caption(
                "※ 그래프는 전체 매출·원가 데이터를 한 번에 모아, 선두 숫자 분포가 "
                "이론적 Benford 분포와 얼마나 다른지 보여줍니다. "
                "표본 수가 적거나 금액 범위가 좁으면 신뢰도가 떨어질 수 있습니다."
            )
        else:
            st.info("Benford 분포를 그릴 수 있는 정보가 부족합니다.")

    st.markdown("---")
    st.markdown(
        f"Benford 결과가 최종 점수에 실제 반영되었는지 여부: "
        f"**{'예' if meta.get('benford_used_in_score') else '아니오(가중치 0으로 처리)' }**"
    )
