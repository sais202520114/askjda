import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# --- 설정 ---
FILE_NAME = "titanic survivors.xlsx - Sheet1.csv"
st.set_page_config(layout="wide", page_title="타이타닉 데이터 상관관계 분석")

@st.cache_data
def load_data(file_path):
    """CSV 파일을 로드하고, 데이터 타입을 확인하며, 필요한 전처리를 수행합니다."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"오류: 파일 '{file_path}'을(를) 찾을 수 없습니다. 파일 이름을 확인해 주세요.")
        return None
    
    # 분석에 사용할 숫자형 열만 선택
    # 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare' 등 숫자로 변환 가능한 열만 사용
    
    # 'Sex'를 숫자형으로 변환 (male=0, female=1)
    df['Sex_numeric'] = df['Sex'].apply(lambda x: 1 if x == 'female' else 0)
    
    # 'Embarked'를 더미 변수로 변환 (간단한 예시를 위해 생략하고 기본 숫자형만 사용)
    
    numeric_cols = ['Pclass', 'Sex_numeric', 'Age', 'SibSp', 'Parch', 'Fare']
    
    # Age와 Fare의 결측치는 중앙값으로 대체 (상관관계 계산을 위해)
    for col in numeric_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                # 문자열이 섞여있다면 숫자로 강제 변환하며 오류는 NaN 처리
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 결측치 처리
            df[col].fillna(df[col].median(), inplace=True)
            
    # 실제로 존재하는 숫자형 열만 필터링
    final_numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    return df[final_numeric_cols]

def calculate_correlation(df):
    """데이터프레임의 상관관계 행렬을 계산합니다."""
    return df.corr()

def get_extreme_correlations(corr_matrix, is_positive=True):
    """가장 높은 양의/음의 상관관계를 가진 쌍을 찾습니다."""
    
    # 상관관계 행렬을 길게 펼친 형태로 변환
    corr_unstacked = corr_matrix.unstack().sort_values(ascending=False)
    
    # 자기 자신과의 상관관계 (1.0) 및 중복 쌍 제거
    pairs = corr_unstacked[corr_unstacked.index.get_level_values(0) != corr_unstacked.index.get_level_values(1)]
    
    if is_positive:
        # 가장 높은 양의 상관관계 (1보다 작고, 0보다 큰 값 중 최대)
        # 이미 내림차순 정렬되어 있으므로, 첫 번째 유효한 쌍이 최대 양의 상관관계
        for (var1, var2), corr_value in pairs.items():
            if corr_value > 0:
                return var1, var2, corr_value
    else:
        # 가장 낮은 음의 상관관계 (음수 중 절대값이 가장 큰 값)
        # 전체를 오름차순으로 다시 정렬하여 가장 작은 값(가장 큰 음수)을 찾음
        negative_pairs = corr_unstacked.sort_values(ascending=True)
        for (var1, var2), corr_value in negative_pairs.items():
            if corr_value < 0:
                return var1, var2, corr_value
                
    return None, None, None

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

    # 히트맵 레이어
    heatmap = base.mark_rect().encode(
        color=alt.Color('Correlation', 
                        scale=alt.Scale(range='diverging', domain=[-1, 1], scheme='redyellowblue'),
                        legend=alt.Legend(title="상관계수")
                       ),
        tooltip=['Variable 1', 'Variable 2', alt.Tooltip('Correlation', format=".3f")]
    )

    # 텍스트 레이어 (상관계수 표시)
    text = base.mark_text().encode(
        text=alt.Text('Correlation', format=".2f"),
        color=alt.value('black') # 텍스트 색상을 검정으로 고정
    )

    return (heatmap + text).interactive()


# --- Streamlit 앱 본문 ---
st.title("🚢 타이타닉호 데이터 속성 간 상관관계 분석")
st.markdown("업로드된 데이터(`titanic survivors.xlsx - Sheet1.csv`)를 기반으로 주요 숫자형 속성 간의 상관관계를 분석합니다.")

# 1. 데이터 로드 및 전처리
df_numeric = load_data(FILE_NAME)

if df_numeric is None or df_numeric.empty:
    st.warning("데이터 로드에 문제가 발생했거나, 분석에 사용할 숫자형 데이터가 부족합니다.")
    st.stop()

st.subheader("📊 분석에 사용된 숫자형 속성 데이터 샘플")
st.dataframe(df_numeric.head())

# 2. 상관관계 계산
corr_matrix = calculate_correlation(df_numeric)

st.subheader("🔥 상관관계 행렬 히트맵")
st.altair_chart(create_heatmap(corr_matrix), use_container_width=True)


# 3. 극단적인 상관관계 찾기 및 표시
st.subheader("🔎 극단적인 상관관계 탐색")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 가장 높은 양의 상관관계")
    if st.button("양의 상관관계 결과 보기", key="positive_corr"):
        var1, var2, corr_value = get_extreme_correlations(corr_matrix, is_positive=True)
        
        if corr_value:
            st.success(f"**{var1}**와 **{var2}**")
            st.code(f"상관계수: {corr_value:.4f}")
            
            # 산점도 차트
            chart = alt.Chart(df_numeric).mark_point().encode(
                x=alt.X(var1),
                y=alt.Y(var2),
                tooltip=[var1, var2]
            ).properties(
                title=f'{var1} vs {var2} 산점도 (R={corr_value:.3f})'
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

        else:
            st.info("양의 상관관계를 가진 쌍을 찾을 수 없습니다.")

with col2:
    st.markdown("### 가장 높은 음의 상관관계")
    if st.button("음의 상관관계 결과 보기", key="negative_corr"):
        var1, var2, corr_value = get_extreme_correlations(corr_matrix, is_positive=False)
        
        if corr_value:
            st.error(f"**{var1}**와 **{var2}**")
            st.code(f"상관계수: {corr_value:.4f}")
            
            # 산점도 차트
            chart = alt.Chart(df_numeric).mark_point().encode(
                x=alt.X(var1),
                y=alt.Y(var2),
                tooltip=[var1, var2]
            ).properties(
                title=f'{var1} vs {var2} 산점도 (R={corr_value:.3f})'
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

        else:
            st.info("음의 상관관계를 가진 쌍을 찾을 수 없습니다.")

st.markdown("---")
st.info("참고: 'Sex_numeric'은 female=1, male=0으로 변환된 값입니다.")
