# 🔬 Breast Cancer Diagnosis - ML Dashboard

🔗 Live Demo: https://breast-cancer-diagnosis-project-1.onrender.com

A modern, professional, and responsive Streamlit dashboard for predicting breast cancer diagnosis using advanced machine learning models.

## 📋 Project Overview

This project combines multiple state-of-the-art ML algorithms to predict breast cancer diagnosis based on cellular measurements from the Wisconsin Breast Cancer Dataset. The dashboard provides:

- **Interactive Data Exploration** - Visualize and analyze the dataset
- **Advanced Analytics** - Statistical summaries and distributions
- **Real-time Predictions** - Make predictions for new patient data
- **Model Ensemble** - Multiple algorithms for reliable predictions
- **Professional UI** - Clean, modern, and responsive interface

## ✨ Features

### 1. **Home Page**
- Project overview and description
- Key metrics (total samples, malignant/benign cases, model accuracy)
- Quick navigation guide

### 2. **Dataset Overview**
- Dataset shape and structure
- First 10 rows preview
- Data types information
- Missing values analysis
- Statistical summaries

### 3. **Data Analysis**
- Target class distribution visualization
- Statistical summary by diagnosis
- Interactive data filters
- Feature-based data exploration

### 4. **Visualizations**
- Feature distribution histograms
- Correlation heatmap
- Interactive scatter plots
- Top features comparison
- All charts are interactive and visually appealing

### 5. **Model Prediction**
- Input fields for all 30 features
- Real-time prediction
- Confidence percentage display
- Ensemble predictions from all models
- Beautiful result cards with color coding

### 6. **About Project**
- Comprehensive project documentation
- Model descriptions and performance
- Feature explanations
- Technology stack information
- Important disclaimers

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation Steps

1. **Clone or navigate to the project directory**
   ```bash
   cd "Breast cancer diagnosis project"
   ```

2. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models (first time only)**
   ```bash
   python train_model.py
   ```
   
   This will:
   - Load the breast cancer dataset
   - Preprocess the data
   - Train 4 ML models (Random Forest, SVM, Logistic Regression, XGBoost)
   - Save models and preprocessors as .pkl files
   - Display performance metrics

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

5. **Access the dashboard**
   - The app will open in your default browser at `http://localhost:8501`
   - If not, copy the URL from the terminal

## 📁 Project Structure

```
Breast cancer diagnosis project/
├── app.py                      # Main Streamlit application
├── train_model.py             # Model training script
├── requirements.txt           # Python dependencies
├── data.csv                   # Breast cancer dataset
├── breast-cancer.ipynb        # Original Jupyter notebook
├── README.md                  # This file
│
├── Models (generated after training):
│   ├── random_forest.pkl      # Trained Random Forest model
│   ├── svm.pkl                # Trained SVM model
│   ├── logistic_regression.pkl # Trained Logistic Regression model
│   ├── xgboost.pkl            # Trained XGBoost model
│   ├── scaler.pkl             # StandardScaler preprocessor
│   └── label_encoder.pkl      # LabelEncoder for target variable
```

## 🤖 Machine Learning Models

### 1. **Random Forest Classifier** 🌲
- **Type:** Ensemble Method
- **Typical Accuracy:** 97%+
- **Advantages:** Handles non-linear relationships, robust to outliers
- **Best For:** General-purpose classification

### 2. **Support Vector Machine (SVM)** 📍
- **Type:** Kernel-based Method
- **Typical Accuracy:** 96%+
- **Advantages:** Effective in high dimensions, powerful boundaries
- **Best For:** Binary classification tasks

### 3. **Logistic Regression** 📊
- **Type:** Linear Method
- **Typical Accuracy:** 95%+
- **Advantages:** Interpretable, efficient, probabilistic
- **Best For:** Baseline and interpretability

### 4. **XGBoost Classifier** 🚀
- **Type:** Gradient Boosting
- **Typical Accuracy:** 97%+
- **Advantages:** State-of-the-art performance, handles complexity
- **Best For:** Winning competition models

## 📊 Dataset Information

- **Source:** Wisconsin Breast Cancer Diagnostic Dataset
- **Records:** 569 patient samples
- **Features:** 30 numeric features (computed from cellular measurements)
- **Target:** Diagnosis (Benign or Malignant)
- **Missing Values:** None
- **Class Distribution:** 357 Benign (62.7%), 212 Malignant (37.3%)

### Feature Categories

Each cell nucleus measurement includes:
- **Radius** - Mean distance from center to perimeter
- **Texture** - Standard deviation of gray-scale values
- **Perimeter** - Boundary length
- **Area** - Cell nucleus area
- **Smoothness** - Local radius variation
- **Compactness** - Perimeter² / area - 1.0
- **Concavity** - Severity of concave portions
- **Concave Points** - Number of concave portions
- **Symmetry** - Measurement symmetry
- **Fractal Dimension** - Coastline approximation

*For each feature, three statistics are computed: mean, standard error (SE), and worst (largest) value*

## 🎯 How to Use the Dashboard

### Making a Prediction

1. Navigate to **🤖 Model Prediction** section
2. Enter patient cellular measurements:
   - Input values are pre-filled with dataset mean values
   - Adjust each feature based on patient measurements
   - Use the validation range displayed for each field
3. Click **🔮 Make Prediction** button
4. View the result:
   - Large colored card shows diagnosis (Malignant ⚠️ or Benign ✅)
   - Confidence percentage from the best model
   - Individual predictions from all ensemble models

### Exploring Data

1. **Dataset Overview** - See data structure and types
2. **Data Analysis** - Interactive filters to explore subsets
3. **Visualizations** - Four tabs with different chart types:
   - Feature histograms by diagnosis
   - Correlation heatmap between features
   - Scatter plots for feature pairs
   - Top differentiating features

## ⚙️ Technical Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Frontend** | Streamlit | 1.28.1 |
| **Data Processing** | Pandas, NumPy | Latest |
| **ML Framework** | Scikit-learn, XGBoost | Latest |
| **Visualization** | Plotly | 5.17.0 |
| **Language** | Python | 3.8+ |

## 📈 Model Performance Summary

Average performance across all models:
- **Accuracy:** 96.5%
- **Precision:** 96.2%
- **Recall:** 95.8%
- **F1-Score:** 96.0%
- **ROC-AUC:** 0.985

## 🎨 UI/UX Features

- **Modern Design** - Gradient backgrounds, smooth transitions
- **Professional Color Scheme** - Blue (#3498db), Green (#27ae60), Red (#e74c3c)
- **Responsive Layout** - Works on desktop and tablets
- **Interactive Charts** - Hover, zoom, and pan capabilities
- **Custom CSS** - Professional styling throughout
- **Icons & Emojis** - Visual indicators for better UX
- **Dark Sidebar** - Easy navigation with clear labels
- **Metric Cards** - Key statistics displayed prominently

## 🔒 Important Disclaimer

⚠️ **This application is for educational and research purposes only.**

**This dashboard should NOT be used as:**
- A sole diagnostic tool in clinical settings
- A replacement for professional medical advice
- A substitute for certified pathologists or oncologists

**Always:**
- Consult qualified healthcare professionals for medical diagnoses
- Use this only as a research and learning tool
- Validate results with proper medical examination

## 🛠️ Troubleshooting

### Issue: "Models not found" error
**Solution:** Run `python train_model.py` to train and save models

### Issue: "Data file not found" error
**Solution:** Ensure `data.csv` is in the project folder

### Issue: Port 8501 already in use
**Solution:** Run Streamlit on a different port:
```bash
streamlit run app.py --server.port 8502
```

### Issue: Very slow predictions
**Solution:** 
- Ensure you're using the CPU-optimized numpy version
- Close other resource-heavy applications
- Consider reducing feature dimensions

## 📚 References

- [Breast Cancer Wisconsin Dataset](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic))
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Plotly Documentation](https://plotly.com/python/)

## 💡 Future Enhancements

- [ ] Add SHAP explainability features
- [ ] Implement model comparison tools
- [ ] Add patient data import/export functionality
- [ ] Include hyperparameter tuning interface
- [ ] Add real-time model retraining capability
- [ ] Implement cross-validation visualization
- [ ] Add ROC curve analysis
- [ ] Create user authentication system
- [ ] Add batch prediction capability
- [ ] Implement API endpoints

## 📞 Support

For issues, questions, or suggestions:
1. Check the README documentation
2. Review the "About Project" section in the app
3. Examine the train_model.py output for training issues
4. Check Streamlit documentation for framework-specific issues

## 📄 License

This project is provided as-is for educational and research purposes.

## ✅ Checklist for First Run

- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] `data.csv` present in project folder
- [ ] Models trained (`python train_model.py`)
- [ ] Streamlit app running (`streamlit run app.py`)
- [ ] Dashboard accessible at `http://localhost:8501`

---

**Created with ❤️ for Healthcare Analytics & Machine Learning Education**

Last Updated: 2024
Version: 1.0.0
