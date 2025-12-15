import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# --- 설정 ---
# 데이터 파일 이름은 'titanic_data.csv'로 통일합니다.
FILE_NAME = "titanic_data.csv" 
st.set_page_config(layout="wide", page_title="타이타닉 데이터 상관관계 분석")

@st.cache_data
def load_data(file_path):
    """
    CSV 파일을 로드하고, 데이터 타입을 확인하며, 필요한 전처리를 수행합니다.
    - 숫자형 열만 선택하고 결측치(NaN)를 중앙값으로 대체합니다.
    - 'Sex' (성별)을 숫자형 ('Sex_numeric': female=1, male=0)으로 변환합니다.
    """
    try:
        # CSV 파일 로드
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"오류: 데이터 파일 '{file_path}'을(를) 찾을 수 없습니다. 파일 이름을 확인해 주세요.")
        return None
    
    # 'Sex'를 숫자형으로 변환 (female=1, male=0)
    df['Sex_numeric'] = df['Sex'].apply(lambda x: 1 if x == 'female' else 0)
    
    # 분석에 사용할 주요 숫자형 속성 리스트
    numeric_cols = ['Pclass', 'Sex_numeric', 'Age', 'SibSp', 'Parch', 'Fare']
    
    for col in numeric_cols:
        if col in df.columns:
            # 문자열을 숫자로 강제 변환 (오류 발생 시 NaN 처리)
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 결측치(NaN)를 해당 열의 중앙값(median)으로 대체
            df[col].fillna(df[col].median(), inplace=True)
            
    # 실제로 존재하는 숫자형 열만 최종 필터링
    final_numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    return df[final_numeric_cols]

def calculate_correlation(df):
    """데이터프레임의 상관관계 행렬을 계산합니다."""
    return df.corr()

def get_extreme_correlations(corr_matrix, is_positive=True):
    """가장 높은 양의/음의 상관관계를 가진 쌍을 찾습니다."""
    
    # 상관관계 행렬을 Series로 펼친 후 정렬
    corr_unstacked = corr_matrix.unstack().sort_values(ascending=False)
    
    # 자기 자신과의 상관관계 (1.0) 및 중복 쌍 제거
    pairs = corr_unstacked[corr_unstacked.index.get_level_values(0) != corr_unstacked.index.get_level_values(1)]
    
    if is_positive:
        # 가장 높은 양의 상관관계 (최대 양수)
        for (var1, var2), corr_value in pairs.items():
            if corr_value > 0:
                return var1, var2, corr_value
    else:
        # 가장 높은 음의 상관관계 (최소 음수)
        negative_pairs = corr_unstacked.sort_values(ascending=True)
        for (var1, var2), corr_value in negative_pairs.items():
            if corr_value < 0:
                return var1, var2, corr_value
                
    return None, None, None

def create_scatterplot(df, var1, var2, corr_value):
    """두 변수 간의 산점도를 생성합니다."""
    chart = alt.Chart(df).mark_point().encode(
        x=alt.X(var1, title=var1),
        y=alt.Y(var2, title=var2),
        tooltip=[var1, var2]
    ).properties(
        title=f'{var1} vs {var2} 산점도 (R={corr_value:.3f})'
    ).interactive()
    return chart

def create_heatmap(corr_df):
    """상관관계 행렬 히트맵을 생성합니다."""
    # 상관관계 행렬을 Altair 차트용 데이터프레임으로 변환
    corr_data = corr_df.stack().reset_index()
    corr_data.columns = ['Variable 1', 'Variable 2', 'Correlation']
    
    # 히트맵 차트 생성
    base = alt.Chart(corr_data).encode(
        x=alt.X('Variable 1', title=None),
        y=alt.Y('Variable 2', title=None)
    ).properties(
        title='속성 간 상관관계 히트맵'
    )

    # 히트맵 레이어: 색상으로 상관계수 강도 표현
    heatmap = base.mark_rect().encode(
        color=alt.Color('Correlation', 
                        # -1부터 1까지의 발산형 색상 스케일 사용
                        scale=alt.Scale(range='diverging', domain=[-1, 1], scheme='redyellowblue'),
                        legend=alt.Legend(title="상관계수")
                       ),
        tooltip=['Variable 1', 'Variable 2', alt.Tooltip('Correlation', format=".3f")]
    )

    # 텍스트 레이어: 히트맵 위에 상관계수 값 표시
    text = base.mark_text().encode(
        text=alt.Text('Correlation', format=".2f"),
        color=alt.value('black') 
    )

    return (heatmap + text).interactive()


# --- Streamlit 앱 본문 ---
st.title("🚢 타이타닉호 데이터 속성 간 상관관계 분석")
st.markdown(f"**{FILE_NAME}** 파일을 사용하여 숫자형 속성 간의 관계를 분석하고 시각화합니다.")

# 1. 데이터 로드 및 전처리
df_numeric = load_data(FILE_NAME)

if df_numeric is None or df_numeric.empty:
    st.stop()

st.subheader("📊 분석에 사용된 숫자형 속성 데이터 샘플")
st.markdown("**(Sex_numeric: 여성=1, 남성=0)**")
st.dataframe(df_numeric.head())

# 2. 상관관계 계산 및 히트맵 표시
corr_matrix = calculate_correlation(df_numeric)

st.subheader("🔥 상관관계 행렬 히트맵 (Correlation Heatmap)")
st.altair_chart(create_heatmap(corr_matrix), use_container_width=True)


# 3. 극단적인 상관관계 탐색
st.subheader("🔎 가장 강력한 상관관계 쌍")

col1, col2 = st.columns(2)

# --- 양의 상관관계 버튼 ---
with col1:
    st.markdown("### 🥇 가장 높은 양의 상관관계 (Positive Correlation)")
    if st.button("양의 상관관계 결과 보기", key="positive_corr"):
        var1, var2, corr_value = get_extreme_correlations(corr_matrix, is_positive=True)
        
        if corr_value:
            st.success(f"**{var1}**와 **{var2}**")
            st.code(f"상관계수 (R): {corr_value:.4f}")
            
            # 산점도 표시
            chart = create_scatterplot(df_numeric, var1, var2, corr_value)
            st.altair_chart(chart, use_container_width=True)

        else:
            st.info("양의 상관관계를 가진 쌍을 찾을 수 없습니다.")

# --- 음의 상관관계 버튼 ---
with col2:
    st.markdown("### 📉 가장 높은 음의 상관관계 (Negative Correlation)")
    if st.button("음의 상관관계 결과 보기", key="negative_corr"):
        var1, var2, corr_value = get_extreme_correlations(corr_matrix, is_positive=False)
        
        if corr_value:
            st.error(f"**{var1}**와 **{var2}**")
            st.code(f"상관계수 (R): {corr_value:.4f}")
            
            # 산점도 표시
            chart = create_scatterplot(df_numeric, var1, var2, corr_value)
            st.altair_chart(chart, use_container_width=True)

        else:
            st.info("음의 상관관계를 가진 쌍을 찾을 수 없습니다.")

st.markdown("---")
st.markdown("Streamlit, Pandas, Altair를 사용하여 분석되었습니다.")
