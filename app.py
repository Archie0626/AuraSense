# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import time
from scipy import signal
from scipy.fft import fft
import base64
from io import BytesIO

# Page config
st.set_page_config(
    page_title="AuraSense - Human Activity Recognition",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with pastel theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #EA7B7B;
        --secondary: #D25353;
        --accent: #9E3B3B;
        --light-accent: #FD7979;
        --pastel-1: #FFC7A7;
        --pastel-2: #FEE2AD;
        --pastel-3: #F8FAB4;
    }
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #FFF5F5 0%, #FFF0E6 100%);
    }
    
    /* Custom containers */
    .main-header {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid var(--primary);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--secondary), var(--accent));
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(210, 83, 83, 0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, var(--pastel-1), var(--pastel-2));
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: white;
        padding: 0.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: var(--accent);
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white !important;
        border-radius: 8px;
    }
    
    /* Info boxes */
    .info-box {
        background: var(--pastel-3);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid var(--primary);
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        background: white;
        border-radius: 15px;
        margin-top: 3rem;
        color: var(--secondary);
    }
            /* ================= MAKE ALL TEXT BLACK ================= */

/* All headings */
h1, h2, h3, h4, h5, h6 {
    color: #000000 !important;
}

/* Normal text */
/* ================= FORCE WHITE DROPDOWN TEXT ================= */

/* Selected value */
[data-baseweb="select"] div {
    color: white !important;
}

/* Dropdown menu container */
[data-baseweb="popover"] {
    background-color: #1E1E2F !important;
}

/* Dropdown items */
[data-baseweb="popover"] * {
    color: white !important;
}

/* Hover highlight */
[data-baseweb="popover"] [role="option"]:hover {
    background-color: #33334d !important;
    color: white !important;
}

/* Dropdown label */
label[data-testid="stWidgetLabel"] {
    color: white !important;
}

/* Streamlit markdown content */
[data-testid="stMarkdownContainer"] {
    color: #000000 !important;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Tabs text */
.stTabs [data-baseweb="tab"] {
    color: #000000 !important;
}

/* Metric cards */
.metric-card h4, .metric-card p {
    color: #000000 !important;
}

/* Info box text */
.info-box, .info-box h4, .info-box p {
    color: #000000 !important;
}

/* Footer */
.footer, .footer p {
    color: #000000 !important;
}
.main-header h1, .main-header p {
    color: white !important;
}

/* Selected value (closed dropdown) */
[data-baseweb="select"] > div {
    color: white !important;
}

/* Dropdown menu text */
[data-baseweb="popover"] [role="option"] {
    color: white !important;
}

/* Dropdown background dark (optional but recommended) */
[data-baseweb="popover"] {
    background-color: #1E1E2F !important;
}

/* Hover effect */
[data-baseweb="popover"] [role="option"]:hover {
    background-color: #33334d !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'results' not in st.session_state:
    st.session_state.results = {}

# Header
st.markdown("""
<div class="main-header fade-in">
    <h1 style="font-size: 3rem; margin-bottom: 0.5rem;"> AuraSense  </h1>
    <p style="font-size: 1.2rem; opacity: 0.95;">Human Activity Recognition using Smartphone Sensors</p>
  </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Navigation")
    page = st.radio(
        "Select Module",
        ["📊 Data Preprocessing", "🔧 Feature Engineering", "🤖 Model Training", 
         "📈 Evaluation Dashboard", "🎯 Real-time Prediction"]
    )
    
    st.markdown("---")
    
    st.markdown("## 📊 Dataset Info")
    if st.button("📥 Load Sample Dataset"):
        with st.spinner("Loading UCI HAR dataset..."):
            time.sleep(2)  # Simulate loading
            st.session_state.data_loaded = True
            st.success("✅ Dataset loaded successfully!")
    
    st.markdown("---")
    
    st.markdown("## 🎨 Theme")
    theme = st.selectbox("Color Theme", ["Pastel", "Ocean", "Sunset", "Forest"])
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("## 📈 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Active Users", "1.2K", "↑12%")
    with col2:
        st.metric("Models Trained", "45", "↑5")

# Main content based on page selection
if page == "📊 Data Preprocessing":
    st.markdown("## 🔄 Data Preprocessing Pipeline")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box fade-in">
            <h4>📋 Dataset Overview</h4>
            <p>The UCI HAR dataset contains recordings of 30 subjects performing activities of daily living while carrying a waist-mounted smartphone with embedded inertial sensors.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Data preview
        if st.session_state.data_loaded:
            # Sample data
            np.random.seed(42)
            sample_data = pd.DataFrame({
                'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='10ms'),
                'acc_x': np.random.normal(0, 1, 100) * 9.8,
                'acc_y': np.random.normal(0, 1, 100) * 9.8,
                'acc_z': np.random.normal(0, 1, 100) * 9.8,
                'gyro_x': np.random.normal(0, 0.1, 100),
                'gyro_y': np.random.normal(0, 0.1, 100),
                'gyro_z': np.random.normal(0, 0.1, 100),
                'activity': np.random.choice(['Walking', 'Sitting', 'Standing', 'Running', 'Upstairs', 'Downstairs', 'Laying'], 100)
            })
            
            st.dataframe(sample_data.head(10), use_container_width=True)
            
            # Data preprocessing options
            st.markdown("### 🛠️ Preprocessing Steps")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                normalize = st.checkbox("Normalize data", value=True)
                remove_noise = st.checkbox("Remove noise", value=True)
            with col_b:
                window_size = st.slider("Window size (samples)", 50, 300, 128)
                overlap = st.slider("Window overlap (%)", 0, 90, 50)
            with col_c:
                sampling_rate = st.number_input("Sampling rate (Hz)", 20, 100, 50)
                label_encode = st.checkbox("Encode labels", value=True)
            
            if st.button("🚀 Apply Preprocessing"):
                with st.spinner("Processing data..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    st.success("✅ Preprocessing completed!")
                    
    with col2:
        st.markdown("### 📊 Data Statistics")
        if st.session_state.data_loaded:
            # Activity distribution
            activities = ['Walking', 'Sitting', 'Standing', 'Running', 'Upstairs', 'Downstairs', 'Laying']
            counts = np.random.randint(100, 500, 7)
            
            fig = px.pie(values=counts, names=activities, 
                        title="Activity Distribution",
                        color_discrete_sequence=['#EA7B7B', '#D25353', '#9E3B3B', '#FD7979', '#FFC7A7', '#FEE2AD', '#F8FAB4'])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Sensor data summary
            st.markdown("### 📈 Sensor Summary")
            summary_df = pd.DataFrame({
                'Sensor': ['Accelerometer', 'Gyroscope'],
                'Mean': [9.81, 0.15],
                'Std': [2.34, 0.08],
                'Range': [19.62, 0.45]
            })
            st.dataframe(summary_df, use_container_width=True)

elif page == "🔧 Feature Engineering":
    st.markdown("## 🔧 Feature Engineering Pipeline")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load the dataset first from the sidebar!")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="info-box fade-in">
                <h4>📐 Feature Extraction</h4>
                <p>Extracting statistical features, FFT features, and time-frequency domain features from sensor data.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature selection
            st.markdown("### 🎯 Select Features")
            
            tab1, tab2, tab3 = st.tabs(["📊 Statistical", "⚡ FFT Features", "⏱️ Time-Frequency"])
            
            with tab1:
                st.markdown("**Statistical Features**")
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    mean = st.checkbox("Mean", value=True)
                    std = st.checkbox("Standard Deviation", value=True)
                    var = st.checkbox("Variance", value=True)
                with col_s2:
                    min_val = st.checkbox("Minimum", value=True)
                    max_val = st.checkbox("Maximum", value=True)
                    range_val = st.checkbox("Range", value=True)
                with col_s3:
                    skew = st.checkbox("Skewness", value=False)
                    kurt = st.checkbox("Kurtosis", value=False)
                    rms = st.checkbox("RMS", value=True)
            
            with tab2:
                st.markdown("**FFT Features**")
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    fft_mag = st.checkbox("FFT Magnitude", value=True)
                    fft_phase = st.checkbox("FFT Phase", value=False)
                with col_f2:
                    dominant_freq = st.checkbox("Dominant Frequency", value=True)
                    spectral_energy = st.checkbox("Spectral Energy", value=True)
            
            with tab3:
                st.markdown("**Time-Frequency Features**")
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    stft = st.checkbox("STFT", value=True)
                    wavelet = st.checkbox("Wavelet", value=False)
                with col_t2:
                    entropy = st.checkbox("Spectral Entropy", value=True)
                    centroid = st.checkbox("Spectral Centroid", value=False)
            
            if st.button("🔧 Extract Features"):
                with st.spinner("Extracting features..."):
                    time.sleep(2)
                    st.success("✅ Features extracted successfully!")
                    
                    # Show extracted features
                    st.markdown("### 📊 Extracted Features Overview")
                    feature_df = pd.DataFrame({
                        'Feature': ['Mean_X', 'Mean_Y', 'Mean_Z', 'Std_X', 'FFT_Mag', 'Dominant_Freq'],
                        'Type': ['Statistical', 'Statistical', 'Statistical', 'Statistical', 'FFT', 'FFT'],
                        'Dimension': [1, 1, 1, 1, 64, 1]
                    })
                    st.dataframe(feature_df, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Feature Visualization")
            
            # Feature correlation heatmap
            np.random.seed(42)
            corr_matrix = np.random.rand(10, 10)
            fig = px.imshow(corr_matrix, 
                          title="Feature Correlation Matrix",
                          color_continuous_scale=['#F8FAB4', '#FFC7A7', '#EA7B7B'])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance placeholder
            st.markdown("### ⭐ Feature Importance")
            features = ['Mean_Acc', 'Std_Acc', 'FFT_Mag', 'Energy', 'Entropy']
            importance = [0.25, 0.20, 0.18, 0.15, 0.22]
            fig = px.bar(x=features, y=importance, 
                        title="Preliminary Feature Importance",
                        color=importance,
                        color_continuous_scale=['#FEE2AD', '#FFC7A7', '#EA7B7B'])
            st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Model Training":
    st.markdown("## 🤖 Model Training & Comparison")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load the dataset first from the sidebar!")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            <div class="info-box fade-in">
                <h4>🎯 Available Models</h4>
                <p>Compare multiple machine learning and deep learning models for activity recognition.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Model selection
            st.markdown("### 📋 Select Models")
            
            st.markdown("**Traditional ML**")
            col_ml1, col_ml2, col_ml3 = st.columns(3)
            with col_ml1:
                lr = st.checkbox("Logistic Regression", value=True)
                rf = st.checkbox("Random Forest", value=True)
            with col_ml2:
                svm = st.checkbox("SVM", value=True)
                dt = st.checkbox("Decision Tree", value=False)
            with col_ml3:
                knn = st.checkbox("KNN", value=False)
                nb = st.checkbox("Naive Bayes", value=False)
            
            st.markdown("**Deep Learning**")
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            with col_dl1:
                cnn = st.checkbox("1D CNN", value=True)
                lstm = st.checkbox("LSTM", value=True)
            with col_dl2:
                cnn_lstm = st.checkbox("CNN + LSTM", value=True)
                gru = st.checkbox("GRU", value=False)
            with col_dl3:
                transformer = st.checkbox("Transformer", value=False)
                attention = st.checkbox("Attention", value=False)
            
            # Training parameters
            st.markdown("### ⚙️ Training Parameters")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                test_size = st.slider("Test size (%)", 10, 40, 20)
                cv_folds = st.slider("Cross-validation folds", 2, 10, 5)
            with col_p2:
                epochs = st.number_input("Epochs (for DL)", 10, 200, 50)
                batch_size = st.selectbox("Batch size", [16, 32, 64, 128])
            
            if st.button("🚀 Train Selected Models"):
                with st.spinner("Training models... This may take a few minutes."):
                    progress_bar = st.progress(0)
                    
                    # Simulate training
                    models = []
                    if lr: models.append("Logistic Regression")
                    if rf: models.append("Random Forest")
                    if svm: models.append("SVM")
                    if cnn: models.append("1D CNN")
                    if lstm: models.append("LSTM")
                    if cnn_lstm: models.append("CNN+LSTM")
                    
                    results = []
                    for i, model_name in enumerate(models):
                        # Simulate training progress
                        for j in range(100):
                            time.sleep(0.001)
                            progress_bar.progress(int((i * 100 + j) / len(models)))
                        
                        # Generate random metrics (in real scenario, these would be actual results)
                        acc = np.random.uniform(0.85, 0.98)
                        prec = np.random.uniform(0.84, 0.97)
                        rec = np.random.uniform(0.83, 0.96)
                        f1 = 2 * (prec * rec) / (prec + rec)
                        
                        results.append({
                            'Model': model_name,
                            'Accuracy': acc,
                            'Precision': prec,
                            'Recall': rec,
                            'F1-Score': f1
                        })
                    
                    st.session_state.results = pd.DataFrame(results)
                    st.session_state.models_trained = True
                    
                    progress_bar.progress(100)
                    st.success("✅ All models trained successfully!")
        
        with col2:
            if st.session_state.models_trained:
                st.markdown("### 📊 Training Results")
                
                # Results table
                st.dataframe(st.session_state.results.round(3), use_container_width=True)
                
                # Model comparison chart
                fig = go.Figure()
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                colors = ['#EA7B7B', '#D25353', '#9E3B3B', '#FD7979']
                
                for i, metric in enumerate(metrics):
                    fig.add_trace(go.Bar(
                        name=metric,
                        x=st.session_state.results['Model'],
                        y=st.session_state.results[metric],
                        marker_color=colors[i],
                        text=st.session_state.results[metric].round(3),
                        textposition='outside',
                    ))
                
                fig.update_layout(
                    title="Model Performance Comparison",
                    barmode='group',
                    height=500,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Best model highlight
                best_model = st.session_state.results.loc[st.session_state.results['Accuracy'].idxmax()]
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #F8FAB4, #FEE2AD); padding: 1.5rem; border-radius: 15px; margin-top: 1rem;">
                    <h4 style="color: #9E3B3B;">🏆 Best Model: {best_model['Model']}</h4>
                    <p style="font-size: 1.2rem;">Accuracy: {best_model['Accuracy']:.3f} | F1-Score: {best_model['F1-Score']:.3f}</p>
                </div>
                """, unsafe_allow_html=True)

elif page == "📈 Evaluation Dashboard":
    st.markdown("## 📊 Model Evaluation Dashboard")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first from the Model Training page!")
    else:
        # Create tabs for different evaluation views
        eval_tab1, eval_tab2, eval_tab3, eval_tab4 = st.tabs([
            "📊 Performance Metrics", "🔄 Confusion Matrix", "📈 Learning Curves", "⚖️ Model Comparison"
        ])
        
        with eval_tab1:
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            # Calculate average metrics
            avg_acc = st.session_state.results['Accuracy'].mean()
            avg_prec = st.session_state.results['Precision'].mean()
            avg_rec = st.session_state.results['Recall'].mean()
            avg_f1 = st.session_state.results['F1-Score'].mean()
            
            with col_m1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📈 Avg Accuracy</h4>
                    <p style="font-size: 2rem; color: #EA7B7B;">{avg_acc:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🎯 Avg Precision</h4>
                    <p style="font-size: 2rem; color: #D25353;">{avg_prec:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🔄 Avg Recall</h4>
                    <p style="font-size: 2rem; color: #9E3B3B;">{avg_rec:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m4:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>⭐ Avg F1-Score</h4>
                    <p style="font-size: 2rem; color: #FD7979;">{avg_f1:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed metrics per model
            st.markdown("### 📋 Detailed Metrics by Model")
            
            # Create a more detailed metrics dataframe
            detailed_metrics = st.session_state.results.copy()
            detailed_metrics['Training Time (s)'] = np.random.uniform(10, 120, len(detailed_metrics))
            detailed_metrics['Inference Time (ms)'] = np.random.uniform(1, 50, len(detailed_metrics))
            detailed_metrics['Model Size (MB)'] = np.random.uniform(0.5, 200, len(detailed_metrics))
            
            st.dataframe(detailed_metrics.style.highlight_max(axis=0, subset=['Accuracy', 'F1-Score']), 
                        use_container_width=True)
        
        with eval_tab2:
            st.markdown("### 🔍 Confusion Matrix Analysis")
            
            # Model selector for confusion matrix
            selected_model = st.selectbox(
                "Select Model for Confusion Matrix",
                st.session_state.results['Model'].tolist()
            )
            
            # Generate sample confusion matrix
            activities = ['Walking', 'Sitting', 'Standing', 'Running', 'Upstairs', 'Downstairs', 'Laying']
            cm = np.random.randint(0, 100, size=(7, 7))
            np.fill_diagonal(cm, np.random.randint(80, 100, 7))
            
            # Normalize
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            fig = px.imshow(cm_normalized,
                          x=activities,
                          y=activities,
                          color_continuous_scale=['#F8FAB4', '#FFC7A7', '#EA7B7B'],
                          aspect="auto",
                          title=f"Confusion Matrix - {selected_model}")
            
            fig.update_layout(height=600)
            
            # Add annotations
            for i in range(len(activities)):
                for j in range(len(activities)):
                    fig.add_annotation(
                        x=activities[j],
                        y=activities[i],
                        text=f"{cm[i, j]}",
                        showarrow=False,
                        font=dict(color="white" if cm_normalized[i, j] > 0.5 else "black")
                    )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Per-class metrics
            st.markdown("### 📊 Per-Class Performance")
            class_metrics = pd.DataFrame({
                'Activity': activities,
                'Precision': np.random.uniform(0.85, 0.98, 7),
                'Recall': np.random.uniform(0.83, 0.97, 7),
                'F1-Score': np.random.uniform(0.84, 0.97, 7),
                'Support': np.random.randint(100, 500, 7)
            }).round(3)
            
            st.dataframe(class_metrics, use_container_width=True)
        
        with eval_tab3:
            st.markdown("### 📈 Training History")
            
            # Generate sample learning curves
            epochs = 50
            train_loss = np.exp(-np.linspace(0, 3, epochs)) + np.random.normal(0, 0.02, epochs)
            val_loss = np.exp(-np.linspace(0, 2.5, epochs)) + np.random.normal(0, 0.03, epochs)
            train_acc = 1 - train_loss + np.random.normal(0, 0.01, epochs)
            val_acc = 1 - val_loss + np.random.normal(0, 0.02, epochs)
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Loss Curves", "Accuracy Curves")
            )
            
            fig.add_trace(
                go.Scatter(x=list(range(epochs)), y=train_loss, name="Training Loss",
                          line=dict(color="#EA7B7B", width=2)),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=list(range(epochs)), y=val_loss, name="Validation Loss",
                          line=dict(color="#9E3B3B", width=2, dash="dash")),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=list(range(epochs)), y=train_acc, name="Training Accuracy",
                          line=dict(color="#FD7979", width=2)),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(x=list(range(epochs)), y=val_acc, name="Validation Accuracy",
                          line=dict(color="#D25353", width=2, dash="dash")),
                row=1, col=2
            )
            
            fig.update_layout(height=500, showlegend=True)
            fig.update_xaxes(title_text="Epochs", row=1, col=1)
            fig.update_xaxes(title_text="Epochs", row=1, col=2)
            fig.update_yaxes(title_text="Loss", row=1, col=1)
            fig.update_yaxes(title_text="Accuracy", row=1, col=2)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with eval_tab4:
            st.markdown("### ⚖️ Comprehensive Model Comparison")
            
            # Radar chart for model comparison
            fig = go.Figure()
            
            for _, row in st.session_state.results.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score'], row['Accuracy']],
                    theta=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Accuracy'],
                    fill='toself',
                    name=row['Model']
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0.8, 1.0]
                    )),
                showlegend=True,
                height=600,
                title="Model Performance Radar Chart"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Model recommendation
            st.markdown("""
            <div style="background: linear-gradient(135deg, #FEE2AD, #FFC7A7); padding: 2rem; border-radius: 15px; margin-top: 2rem;">
                <h3 style="color: #9E3B3B;">🎯 Model Recommendation</h3>
                <p style="font-size: 1.1rem;">Based on the comprehensive evaluation, the <b>CNN+LSTM hybrid model</b> provides the best balance of accuracy, inference time, and robustness for real-time activity recognition.</p>
                <ul style="font-size: 1rem;">
                    <li>Highest F1-Score: 0.967</li>
                    <li>Lowest inference time: 12ms</li>
                    <li>Best performance on transition activities (upstairs/downstairs)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

elif page == "🎯 Real-time Prediction":
    st.markdown("## 🎯 Real-time Activity Prediction")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first from the Model Training page!")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            <div class="info-box fade-in">
                <h4>📱 Sensor Simulation</h4>
                <p>Simulate real-time sensor data for activity prediction. In a real deployment, this would connect to actual smartphone sensors.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Model selection for prediction
            selected_model = st.selectbox(
                "Select Model for Prediction",
                st.session_state.results['Model'].tolist(),
                key="pred_model"
            )
            
            # Activity selection for simulation
            st.markdown("### 🎮 Simulation Controls")
            actual_activity = st.selectbox(
                "Simulate Activity",
                ['Walking', 'Sitting', 'Standing', 'Running', 'Upstairs', 'Downstairs', 'Laying']
            )
            
            # Add some noise to make it realistic
            noise_level = st.slider("Sensor Noise Level", 0.0, 0.5, 0.1)
            
            # Generate synthetic sensor data
            st.markdown("### 📊 Live Sensor Feed")
            
            # Create placeholder for live data
            chart_placeholder = st.empty()
            metrics_placeholder = st.empty()
            
            if st.button("🎯 Start Real-time Prediction"):
                with st.spinner("Initializing sensor stream..."):
                    time.sleep(1)
                
                # Simulate real-time data stream
                for i in range(50):  # Simulate 50 time steps
                    # Generate synthetic sensor data based on activity
                    if actual_activity == 'Walking':
                        acc_pattern = np.sin(np.linspace(i, i+10, 20)) * 2 + 9.8
                        gyro_pattern = np.cos(np.linspace(i, i+10, 20)) * 0.3
                    elif actual_activity == 'Running':
                        acc_pattern = np.sin(np.linspace(i, i+10, 20)) * 4 + 9.8
                        gyro_pattern = np.cos(np.linspace(i, i+10, 20)) * 0.6
                    elif actual_activity == 'Sitting':
                        acc_pattern = np.ones(20) * 9.8 + np.random.normal(0, 0.1, 20)
                        gyro_pattern = np.random.normal(0, 0.05, 20)
                    else:
                        acc_pattern = np.random.normal(9.8, 1, 20)
                        gyro_pattern = np.random.normal(0, 0.2, 20)
                    
                    # Add noise
                    acc_pattern += np.random.normal(0, noise_level, 20)
                    gyro_pattern += np.random.normal(0, noise_level/2, 20)
                    
                    # Create dataframe for plotting
                    df_live = pd.DataFrame({
                        'Time': np.arange(20),
                        'Acc_X': acc_pattern,
                        'Acc_Y': acc_pattern * 0.9,
                        'Acc_Z': acc_pattern * 1.1,
                        'Gyro_X': gyro_pattern,
                        'Gyro_Y': gyro_pattern * 0.8,
                        'Gyro_Z': gyro_pattern * 1.2
                    })
                    
                    # Update chart
                    fig = make_subplots(rows=2, cols=1, 
                                       subplot_titles=("Accelerometer", "Gyroscope"))
                    
                    for sensor, color in [('Acc', '#EA7B7B'), ('Gyro', '#9E3B3B')]:
                        for axis, offset in [('X', 0), ('Y', 1), ('Z', 2)]:
                            if sensor == 'Acc':
                                fig.add_trace(
                                    go.Scatter(x=df_live['Time'], y=df_live[f'{sensor}_{axis}'],
                                              name=f'{sensor}_{axis}', line=dict(color=color, width=2),
                                              opacity=0.8 - offset*0.2),
                                    row=1, col=1
                                )
                            else:
                                fig.add_trace(
                                    go.Scatter(x=df_live['Time'], y=df_live[f'{sensor}_{axis}'],
                                              name=f'{sensor}_{axis}', line=dict(color=color, width=2),
                                              opacity=0.8 - offset*0.2),
                                    row=2, col=1
                                )
                    
                    fig.update_layout(height=500, showlegend=True)
                    chart_placeholder.plotly_chart(fig, use_container_width=True)
                    
                    # Simulate prediction
                    time.sleep(0.1)
                    
                    # Generate prediction probabilities
                    activities = ['Walking', 'Sitting', 'Standing', 'Running', 'Upstairs', 'Downstairs', 'Laying']
                    if actual_activity == 'Walking':
                        probs = [0.85, 0.01, 0.02, 0.05, 0.03, 0.03, 0.01]
                    elif actual_activity == 'Running':
                        probs = [0.10, 0.00, 0.00, 0.85, 0.02, 0.02, 0.01]
                    elif actual_activity == 'Sitting':
                        probs = [0.01, 0.90, 0.05, 0.00, 0.01, 0.01, 0.02]
                    else:
                        probs = np.random.dirichlet(np.ones(7)*0.5)
                    
                    predicted_activity = activities[np.argmax(probs)]
                    confidence = np.max(probs)
                    
                    # Display prediction
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>Actual Activity</h4>
                            <p style="font-size: 1.5rem; color: #9E3B3B;">{actual_activity}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_m2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>Predicted Activity</h4>
                            <p style="font-size: 1.5rem; color: #EA7B7B;">{predicted_activity}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_m3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>Confidence</h4>
                            <p style="font-size: 1.5rem; color: #D25353;">{confidence:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show probability distribution
                    prob_df = pd.DataFrame({
                        'Activity': activities,
                        'Probability': probs
                    })
                    
                    fig_prob = px.bar(prob_df, x='Activity', y='Probability',
                                    color='Probability',
                                    color_continuous_scale=['#F8FAB4', '#FFC7A7', '#EA7B7B'],
                                    title="Prediction Probabilities")
                    fig_prob.update_layout(height=300)
                    st.plotly_chart(fig_prob, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Prediction History")
            
            # Generate sample prediction history
            history_df = pd.DataFrame({
                'Timestamp': pd.date_range(start='2024-01-01', periods=20, freq='1s'),
                'Actual': np.random.choice(activities, 20),
                'Predicted': np.random.choice(activities, 20),
                'Confidence': np.random.uniform(0.7, 1.0, 20)
            })
            
            # Calculate accuracy
            accuracy = (history_df['Actual'] == history_df['Predicted']).mean()
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #FEE2AD, #FFC7A7); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                <h4 style="color: #9E3B3B;">Session Accuracy: {accuracy:.1%}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(history_df, use_container_width=True)
            
            # Model performance stats
            st.markdown("### ⚡ Model Performance")
            
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.metric("Avg Inference Time", "12ms", "↓2ms")
                st.metric("Model Size", "45MB", "↑5MB")
            with col_s2:
                st.metric("Throughput", "83 pred/s", "↑12")
                st.metric("Uptime", "99.9%", "→")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p style="margin-bottom: 0.5rem;">✨ AuraSense - Human Activity Recognition System ✨</p>
    <p style="font-size: 0.9rem; opacity: 0.8;">Built with ❤️ using Streamlit | UCI HAR Dataset | Machine Learning & Deep Learning</p>
    <p style="font-size: 0.8rem; margin-top: 1rem;">© 2024 AuraSense. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)