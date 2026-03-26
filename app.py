import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Lung Cancer Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .normal-result {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .cancer-result {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .high-risk {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .sidebar-info {
        background-color: transparent;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    .footer {
        text-align: center;
        padding: 1rem;
        background-color: transparent;
        border-radius: 5px;
        margin-top: 2rem;
        border-top: 2px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🫁 Lung Cancer Detection System</h1>
    <p>AI-powered CT scan and symptom analysis</p>
</div>
""", unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def load_models():
    """Load all trained models"""
    try:
        # Load CNN model
        cnn_model = tf.keras.models.load_model('models/best_cnn_model.keras')
        
        # Load ML models
        with open('models/ensemble.pkl', 'rb') as f:
            ensemble_model = pickle.load(f)
        
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        return cnn_model, ensemble_model, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# Load models
cnn_model, ensemble_model, scaler = load_models()

# Class labels
class_labels = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Normal', 'Squamous Cell Carcinoma']

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-info">
        <h3>📋 About</h3>
        <p>This system uses advanced AI to detect lung cancer from:</p>
        <ul>
            <li>📸 CT Scan Images</li>
            <li>🏥 Clinical Symptoms</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-info">
        <h3>🎯 Instructions</h3>
        <ol>
            <li><strong>CT Scan:</strong> Upload a medical image</li>
            <li><strong>Symptoms:</strong> Fill in all clinical data</li>
            <li><strong>Predict:</strong> Click analyze buttons</li>
            <li><strong>Results:</strong> Review predictions</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-info">
        <h3>🤖 Model Performance</h3>
        <p><strong>CNN Model:</strong> 90% Accuracy</p>
        <p><strong>ML Model:</strong> 94.44% Accuracy</p>
        <p><strong>AUC Score:</strong> 98.07%</p>
    </div>
    """, unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["📸 CT Scan Analysis", "🏥 Symptom Analysis"])

with tab1:
    st.header("CT Scan Image Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload CT Scan")
        uploaded_file = st.file_uploader("Choose a CT scan image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded CT Scan", use_column_width=True)
            
            # Center the predict button
            col_center = st.columns([1, 2, 1])[1]
            with col_center:
                predict_button = st.button("🔍 Predict CT Scan", key="ct_predict", use_container_width=True)
        
        if uploaded_file is not None and predict_button:
            if cnn_model is not None:
                with st.spinner("🔄 Analyzing CT scan..."):
                    try:
                        # Preprocess image
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        
                        img = image.resize((224, 224))
                        img_array = np.array(img, dtype=np.float32) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        # Make prediction
                        prediction = cnn_model.predict(img_array)
                        predicted_class = np.argmax(prediction[0])
                        confidence = np.max(prediction[0]) * 100
                        
                        # Store results for display in right column
                        st.session_state.ct_prediction = {
                            'class': class_labels[predicted_class],
                            'confidence': confidence,
                            'probabilities': {class_labels[i]: prediction[0][i] * 100 for i in range(4)}
                        }
                        
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {e}")
            else:
                st.error("❌ CNN model not loaded. Please check model files.")
    
    with col2:
        st.subheader("Prediction Results")
        
        if 'ct_prediction' in st.session_state:
            result = st.session_state.ct_prediction
            
            # Determine result styling
            if result['class'] == 'Normal':
                result_class = "normal-result"
                icon = "✅"
            else:
                result_class = "cancer-result"
                icon = "⚠️"
            
            st.markdown(f"""
            <div class="prediction-box {result_class}">
                <h2>{icon} {result['class']}</h2>
                <p><strong>Confidence:</strong> {result['confidence']:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show all probabilities
            st.subheader("📊 Detailed Probabilities")
            for label, prob in result['probabilities'].items():
                st.progress(prob/100)
                st.write(f"**{label}**: {prob:.2f}%")
        else:
            st.markdown("""
            <div class="sidebar-info">
                <p>👈 Upload a CT scan image and click "Predict CT Scan" to see results here.</p>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.header("Symptom Analysis")
    
    # Patient Information Section
    st.subheader("📋 Patient Information")
    
    # Create organized input layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Basic Info**")
        gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female", index=0)
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        
        st.markdown("**Lifestyle**")
        smoking = st.selectbox("Smoking", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes", index=0)
        alcohol_consuming = st.selectbox("Alcohol Consuming", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes", index=0)
        
    with col2:
        st.markdown("**Physical Symptoms**")
        yellow_fingers = st.selectbox("Yellow Fingers", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes", index=0)
        anxiety = st.selectbox("Anxiety", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes", index=0)
        peer_pressure = st.selectbox("Peer Pressure", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes", index=0)
        chronic_disease = st.selectbox("Chronic Disease", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes", index=0)
        fatigue = st.selectbox("Fatigue", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes", index=0)
        
    with col3:
        st.markdown("**Respiratory Symptoms**")
        allergy = st.selectbox("Allergy", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes", index=0)
        wheezing = st.selectbox("Wheezing", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes", index=0)
        coughing = st.selectbox("Coughing", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes", index=0)
        shortness_of_breath = st.selectbox("Shortness of Breath", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes", index=0)
        swallowing_difficulty = st.selectbox("Swallowing Difficulty", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes", index=0)
        chest_pain = st.selectbox("Chest Pain", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes", index=0)
    
    # Center the predict button
    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        symptom_predict_button = st.button("🔬 Analyze Symptoms", key="symptom_predict", use_container_width=True)
    
    # Prediction Results
    if symptom_predict_button:
        if ensemble_model is not None and scaler is not None:
            with st.spinner("🔄 Analyzing symptoms..."):
                try:
                    # Collect features
                    features = [
                        gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
                        chronic_disease, fatigue, allergy, wheezing, alcohol_consuming,
                        coughing, shortness_of_breath, swallowing_difficulty, chest_pain
                    ]
                    
                    # Add engineered features
                    if age <= 50:
                        age_group = 0
                    elif age <= 60:
                        age_group = 1
                    elif age <= 70:
                        age_group = 2
                    else:
                        age_group = 3
                    
                    risk_score = smoking + alcohol_consuming + chronic_disease
                    symptom_severity = coughing + shortness_of_breath + chest_pain + wheezing
                    
                    features.extend([age_group, risk_score, symptom_severity])
                    
                    # Scale features
                    features_scaled = scaler.transform([features])
                    
                    # Make prediction
                    prediction = ensemble_model.predict(features_scaled)[0]
                    
                    # Display result
                    st.subheader("🎯 Symptom Analysis Results")
                    
                    if prediction == 1:
                        st.markdown("""
                        <div class="prediction-box high-risk">
                            <h2>⚠️ HIGH RISK</h2>
                            <p>The symptoms indicate a high risk of lung cancer.</p>
                            <p><strong>⚡ Immediate Action Required:</strong> Please consult a medical professional immediately.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="prediction-box normal-result">
                            <h2>✅ LOW RISK</h2>
                            <p>The symptoms indicate a low risk of lung cancer.</p>
                            <p><strong>💡 Recommendation:</strong> Regular check-ups are still recommended for monitoring.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")
        else:
            st.error("❌ ML models not loaded. Please check model files.")

# Footer
st.markdown("""
<div class="footer">
    <p>⚠️ <strong>For research purposes only. Not for medical diagnosis.</strong></p>
    <p>Always consult qualified healthcare professionals for medical concerns.</p>
</div>
""", unsafe_allow_html=True)
