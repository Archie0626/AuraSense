🧠 AuraSense – Human Activity Recognition System
🚀 Overview

AuraSense is an intelligent Human Activity Recognition (HAR) system that uses machine learning models to classify human activities based on sensor data.

It combines signal processing, feature engineering, and multiple ML algorithms into an interactive Streamlit web application for real-time analysis and visualization 📊⚡

✨ Key Features
🧠 Machine Learning Models
🌳 Random Forest Classifier
🔵 Support Vector Machine (SVM)
📈 Logistic Regression
🧠 Deep Learning Models:
        CNN (Convolutional Neural Network)
        LSTM (Long Short-Term Memory)
📊 Data Processing & Analysis
📉 Signal processing using FFT & filtering
⚙️ Feature engineering pipeline
📏 Data scaling and normalization
🏷️ Label encoding
📈 Visualization Dashboard

Interactive charts using:
    Plotly
    Matplotlib
    Seaborn
    Confusion matrix visualization
    Performance metrics:
    Accuracy
    Precision
    Recall
    F1 Score

🌐 Web Application (Streamlit)
🎯 Clean UI with pastel theme
📂 Upload and analyze datasets
⚡ Real-time predictions
📊 Model comparison interface

🛠️ Tech Stack
💻 Core Technologies
      Python 🐍
      Streamlit 🌐
📊 Data & ML Libraries
      NumPy
      Pandas
      Scikit-learn
      SciPy
🤖 Deep Learning
      TensorFlow 
📉 Visualization
      Plotly
      Matplotlib
      Seaborn

📂 Project Structure
AuraSense-main/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Dependencies
├── run.sh                  # Run script
├── logo.png                # App logo
│
├── models/                 # Trained ML & DL models
│   ├── cnn_model.h5
│   ├── lstm_model.h5
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   ├── label_encoder.pkl
│   ├── scaler.pkl
│   └── activity_labels.txt
│
├── utils/                  # Data processing modules
│   ├── preprocessing.py
│   └── feature_engineering.py

⚙️ Installation & Setup
# Clone the repository
git clone https://github.com/archie0626/aurasense.git

# Navigate into project
cd AuraSense-main

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

👉 Open in browser:
http://localhost:8501

🧪 How It Works
📂 Upload sensor dataset
⚙️ Preprocess data (scaling, encoding)
🔍 Extract features (signal + statistical)
🤖 Run predictions using ML/DL models
📊 Visualize results and performance
📊 Model Performance (Example)
        Model	          Accuracy	      Precision        Recall         F1 Score
        Random Forest	    High	          High	          High	          High
        SVM	            Moderate	        Good	          Good	          Good
        Logistic Regression	Moderate	   Moderate	      Moderate	      Moderate
        CNN / LSTM	    Very High	       Very High	    Very High	      Very High

🎯 Use Cases
🏃 Fitness & activity tracking
🏥 Healthcare monitoring
🧓 Elderly fall detection
📱 Smart wearable applications
🧠 Behavioral analytics

🔮 Future Enhancements
📡 Real-time sensor integration (IoT devices)
🤖 AI-based activity prediction improvements
☁️ Cloud deployment (AWS / GCP)
📱 Mobile app integration
📊 Advanced analytics dashboard

🤝 Contributing

Contributions are welcome!
Feel free to fork, improve, and submit a PR 🚀

📜 License

This project is open-source and available under the MIT License.

💡 Author

Your Name
🎓 Data Science Student
🔗 GitHub: https://github.com/Archie0626
