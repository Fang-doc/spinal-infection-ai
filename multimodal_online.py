import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# 设置页面配置：宽屏模式
st.set_page_config(page_title="Spinal Infection Diagnostic Tool", layout="wide")

# 使用相对路径，确保在云端也能找到模型
MODEL_PATH = "Multi-modal RF.pkl"

@st.cache_resource
def load_multimodal_rf():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        st.error("Error: Multi-modal RF model file not found in the repository.")
        st.stop()

# 加载单模型
model = load_multimodal_rf()

# 2. 界面顶部标题与说明
st.title("Differential Diagnosis: Spinal Tuberculosis vs. Brucellosis Spondylitis")
st.markdown("""
This interactive calculator is based on machine learning models to assist clinicians in differentiating 
between **Spinal Tuberculosis (ST)** and **Brucellosis Spondylitis (BS)** using MRI and clinical features.
""")
st.markdown("---")

# 3. 侧边栏：输入特征参数
st.sidebar.header("Input Patient Features")

def get_user_input():
    # 严格按照模型要求的9个参数构建侧边栏输入组件
    abscess_spread = st.sidebar.slider("Abscess Spread (Number of vertebrae)", 0, 10, 3)
    thoracic_spine = st.sidebar.selectbox("Thoracic Spine Involved? (0=No, 1=Yes)", [0, 1], index=1)
    vert_collapse = st.sidebar.selectbox("Vertebra Collapse? (0=No, 1=Yes)", [0, 1], index=1)
    involved_vert = st.sidebar.number_input("Number of Involved Vertebrae", min_value=1, max_value=25, value=2)
    stir_signal = st.sidebar.selectbox("High SI of discs on STIR? (0=No, 1=Yes)", [0, 1], index=0)
    lumbar_spine = st.sidebar.selectbox("Lumbar Spine Involved? (0=No, 1=Yes)", [0, 1], index=1)
    
    # 临床特征
    age = st.sidebar.number_input("Age (years)", min_value=0, max_value=120, value=50, step=1)
    fever = st.sidebar.selectbox("Fever? (0=No, 1=Yes)", [0, 1], index=0)
    neuro_deficiency = st.sidebar.selectbox("Neurological Deficiency? (0=No, 1=Yes)", [0, 1], index=0)

    # 🚨 核心步骤：构建 DataFrame，列名和顺序必须与您的 Multi-modal RF 模型训练时完全一致
    data = {
        'abscess spread': abscess_spread,
        'thoracic spine': thoracic_spine,
        'vertebra collapse': vert_collapse,
        'number of involved vertebrae': involved_vert,
        'high SI of discs on STIR': stir_signal,
        'lumbar spine': lumbar_spine,
        'age': float(age),
        'fever': fever,
        'neurological deficiency': neuro_deficiency
    }
    return pd.DataFrame(data, index=[0])

input_df = get_user_input()

# 4. 主界面：左右 1:1 两列平铺布局
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Patient Profile Summary")
    # 转置表格让显示更直观
    st.table(input_df.T.rename(columns={0: 'Value'}))

with col2:
    st.subheader("ML Prediction Result")
    
    try:
        # 直接使用加载好的 Multimodal RF 模型进行概率预测
        probabilities = model.predict_proba(input_df)[0]
        
        # 1 代表布氏杆菌 (Brucellosis BS), 0 代表脊柱结核 (Tuberculosis ST)
        prob_bs = probabilities[1]
        prob_st = probabilities[0]
        
        # 渲染结果（完美对齐 2.png 的视觉大字号效果）
        st.write("**Model Used:** Random Forest (Multimodal Framework)")
        st.write("Probability of Brucellosis Spondylitis")
        st.markdown(f"<h1 style='font-size: 64px; margin-top: -15px; margin-bottom: 5px;'>{prob_bs:.2%}</h1>", unsafe_allow_html=True)
        
        # 概率进度条
        st.progress(float(prob_bs))
        st.markdown("---")
        
        # 最终临床建议彩色卡片
        if prob_bs > 0.5:
            st.error(f"**Recommendation:** The system predicts a higher risk of **Brucellosis Spondylitis** (Confidence: {prob_bs:.1%}).")
        else:
            st.success(f"**Recommendation:** The system predicts a higher risk of **Spinal Tuberculosis** (Confidence: {prob_st:.1%}).")
            
    except Exception as e:
        st.warning(f"Probability estimation error: {e}")

# 5. 页面底部页脚
st.markdown("---")
st.info("**Disclaimer:** This tool is intended for academic research and clinical auxiliary reference only. It should not replace professional medical judgment or biopsy results.")