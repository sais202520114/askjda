import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os

st.set_page_config(layout="wide", page_title="타이타닉 데이터 상관관계 분석 (XLSX)")

# 분석에 사용할 표준화된 열 이름 정의 (모두 소문자로 통일)
STANDARD_COLS = ['pclass', 'age', 'sibsp', 'parch', 'fare']

def find_data_file():
    """현재 디렉토리에서 첫 번째 XLSX 파일을 찾습니다."""
    for filename in os.listdir('.'):
        if filename.endswith('.xlsx'):
            st.info(f"✅ 데이터 파일 '{filename}'을(를) 자동으로 찾았습니다.")
            return filename
    return None

@st.cache_data
def load_and_preprocess_data(file_path):
    """
    XLSX 파일을 로드하고, 열 이름 정규화, 숫자 변환, 결측치 처리를 수행합니다.
    (pd.read_csv 대신 pd.read_excel 사용)
    """
    try:
        # 엑셀 파일 로드 (첫 번째 시트(sheet_name=0) 사용)
        df = pd.read_excel(file_path, sheet_name=0)
    except FileNotFoundError:
        st.error(f"❌ 오류: 데이터 파일 '{file_path}'을(를) 찾을 수 없습니다.")
        return None
    except Exception as e:
        st.error(f"❌ 오류: Excel 파일 로드 중 문제가 발생했습니다. (오픈파이엑셀 라이브러리가 설치되었는지 확인하세요: {e})")
        return None
    
    # 1. 열 이름 정규화: 소문자로 변환하고 공백 제거 (유연성 확보)
    df.columns = df.columns.str.lower().str.replace(' ', '', regex=False)
    
    # 2. 'sex' (성별) 열 찾기 및 숫자 변환
    sex_col_name = None
    for col in df.columns:
        if 'sex' in col or 'gender' in col:
            sex_col_name = col
            break
            
    if sex_col_name:
        df['sex_numeric'] = df[sex_col_name].astype(str).str.lower().map({'female': 1, 'male': 0})
        df['sex_numeric'].fillna(df['sex_numeric'].median(), inplace=True)
    else:
        st.warning("⚠️ 경고: 'sex' 또는 'gender' 열을 찾을 수 없어 성별 분석은 제외됩니다.")

    # 3. 분석 대상 숫자형 열 정의 및 처리
    numeric_analysis_cols = [col for col in STANDARD_COLS if col in df.columns]
    if 'sex_numeric' in df.columns:
        numeric_analysis_cols.append('sex_numeric')
        
    if not numeric_analysis_cols:
        st.error("❌ 오류: 분석에 사용할 유효한 숫자형 데이터 열 (pclass, age, fare 등)을 찾을 수 없습니다.")
        return None
            
    processed_df = df[numeric_analysis_cols].copy()
    
    for col in processed_df.columns:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        median_val = processed_df[col].median()
        if not pd.isna(median_val):
            processed_df[col].fillna(median_val, inplace=True)
        else:
            processed_df.drop(columns=[col], inplace=True)
            st.warning(f"⚠️ 경고: '{col}' 열의 모든 값이 비어 있거나 숫자가 아니어서 분석에서 제외됩니다.")

    return processed_df

# --- 나머지 함수들 (calculate_correlation, get_extreme_correlations, create_scatterplot, create_heatmap) ---
# [NOTE: 코드가 너무 길어지므로 함수 정의부는 생략하고 본문만 제공합니다. 이전 답변의 함수 정의를 그대로 사용하세요. 문법 오류는 수정되어 있습니다.]

def calculate_correlation(df):
    """데이터프레임의 상관관계 행렬을 계산합니다."""
    return df.corr()

def get_extreme_correlations(corr_matrix, is_positive=True):
    """가장 높은 양의/음의 상관관계를 가진 쌍을 찾습니다."""
    
    corr_unstacked = corr_matrix.unstack()
    
    pairs = corr_unstacked[corr_unstacked.index.get_level_values(0) != corr_unstacked.index.get_level_values(1)]
    
    if is_positive:
        result = pairs[pairs > 0].nlargest(1)
    else:
        result = pairs[pairs < 0].nsmallest(1)
    
    if result.empty:
        return None, None, None
        
    (var1, var2), corr_value = result.index[0], result.iloc[0]
    return var1, var2, corr_value

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
    corr_data = corr_df.stack().reset_index()
    corr_data.columns = ['Variable 1', 'Variable 2', 'Correlation']
    
    base = alt.Chart(corr_data).encode(
        x=alt.X('Variable 1', title=None),
        y=alt.Y('Variable 2', title=None)
    ).properties(
        title='속성 간 상관관계 히트맵'
    )

    heatmap = base.mark_rect().encode(
        color=alt.Color('Correlation', 
                        scale=alt.Scale(range='diverging', domain=[-1, 1], scheme='redyellowblue'),
                        legend=alt.Legend(title="상관계수")
                       ),
        tooltip=['Variable 1', 'Variable 2', alt.Tooltip('Correlation', format=".3f")]
    )

    text = base.mark_text().encode(
        text=alt.Text('Correlation', format=".2f"),
        color=alt.value('black') 
    )

    return (heatmap + text).interactive()


# --- Streamlit 앱 본문 ---
st.title("🚢 타이타닉호 데이터 속성 간 상관관계 분석")

# 0. 데이터 파일 찾기
data_file_name = find_data_file()

if data_file_name is None:
    st.error("🚨 치명적 오류: 현재 폴더에서 `.xlsx` 파일을 찾을 수 없습니다. 데이터 파일을 `app.py`와 같은 폴더에 넣어주세요.")
    st.stop()

st.markdown(f"분석 파일: **{data_file_name}**")

# 1. 데이터 로드 및 전처리
df_numeric = load_and_preprocess_data(data_file_name)

if df_numeric is None or df_numeric.empty:
    st.stop()

st.subheader("📊 분석에 사용된 데이터 샘플 (전처리 및 정규화 완료)")
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
            
            chart = create_scatterplot(df_numeric, var1, var2, corr_value)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("음의 상관관계를 가진 쌍을 찾을 수 없습니다.")

st.markdown("---")
st.markdown("모든 열 이름은 분석의 일관성을 위해 소문자로 변환되었습니다.")
