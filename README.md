# 🌺 Iris Flower Classification Project

A comprehensive machine learning project that classifies Iris flowers into three species (setosa, versicolor, and virginica) based on their sepal and petal measurements.

## 📋 Project Overview

This project implements multiple machine learning algorithms to classify Iris flowers and provides:
- **Data exploration and visualization**
- **Multiple ML model training and comparison**
- **Hyperparameter tuning**
- **Model evaluation and performance metrics**
- **Interactive visualizations**
- **Prediction functionality for new samples**

## 🎯 Dataset Information

The Iris dataset contains 150 samples with 4 features:
- **Sepal Length** (cm)
- **Sepal Width** (cm) 
- **Petal Length** (cm)
- **Petal Width** (cm)

**Target Classes:**
- **Setosa** (50 samples)
- **Versicolor** (50 samples)
- **Virginica** (50 samples)

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone or download the project files**

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the classification script:**
   ```bash
   python iris_classification.py
   ```

## 📊 What the Script Does

### 1. Data Loading and Exploration
- Loads the Iris dataset from scikit-learn
- Displays dataset information, statistics, and class distribution
- Creates a comprehensive DataFrame for analysis

### 2. Data Visualization
- **Static plots:** Scatter plots, box plots, correlation heatmaps, histograms, violin plots
- **Interactive 3D plot:** Plotly-based 3D visualization saved as HTML
- **Feature distribution analysis** by species

### 3. Model Training
Trains multiple machine learning algorithms:
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**

### 4. Hyperparameter Tuning
- Performs GridSearchCV on the best performing model
- Optimizes model parameters for better performance

### 5. Model Evaluation
- Compares all models using multiple metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Cross-validation scores
- Generates confusion matrix and classification reports

### 6. Prediction Examples
- Demonstrates prediction on sample data
- Shows confidence scores for each species
- Includes borderline cases for testing

## 📈 Generated Outputs

The script generates several files:

1. **`iris_analysis.png`** - Comprehensive data visualization
2. **`iris_3d_interactive.html`** - Interactive 3D plot (open in browser)
3. **`model_comparison.png`** - Model performance comparison charts
4. **`confusion_matrix.png`** - Confusion matrix for the best model

## 🔧 Usage Examples

### Basic Usage
```python
from iris_classification import IrisClassifier

# Create classifier instance
classifier = IrisClassifier()

# Run complete analysis
results_df, best_model = classifier.run_complete_analysis()
```

### Making Predictions
```python
# Predict species for new sample
predicted_species, probabilities = classifier.predict_new_sample(
    sepal_length=5.1,
    sepal_width=3.5, 
    petal_length=1.4,
    petal_width=0.2
)
print(f"Predicted species: {predicted_species}")
```

### Using Specific Model
```python
# Use a specific model for prediction
predicted_species, probabilities = classifier.predict_new_sample(
    sepal_length=6.0,
    sepal_width=3.0,
    petal_length=4.5, 
    petal_width=1.5,
    model_name="Random Forest"
)
```

## 📊 Expected Results

Typical performance metrics:
- **Accuracy:** 95-100% (depending on model and tuning)
- **Best performing models:** Usually Random Forest or SVM
- **Cross-validation:** Stable performance across folds

## 🎨 Visualization Features

### Static Visualizations
- **Scatter plots** showing feature relationships
- **Box plots** displaying feature distributions
- **Correlation heatmap** showing feature correlations
- **Violin plots** for detailed distribution analysis
- **Model comparison charts** with multiple metrics

### Interactive Features
- **3D scatter plot** with Plotly (rotatable, zoomable)
- **Hover information** showing exact values
- **Color-coded species** for easy identification

## 🔍 Model Details

### Logistic Regression
- Linear classification algorithm
- Good baseline performance
- Fast training and prediction

### Support Vector Machine (SVM)
- Non-linear classification with RBF kernel
- Excellent performance on this dataset
- Good generalization capability

### Random Forest
- Ensemble method with multiple decision trees
- Robust to overfitting
- Provides feature importance

### K-Nearest Neighbors (KNN)
- Instance-based learning
- Simple and interpretable
- Good for small datasets

## 📝 Key Features

- **Comprehensive analysis** from data loading to prediction
- **Multiple visualization types** (static and interactive)
- **Automated hyperparameter tuning**
- **Detailed performance metrics**
- **Easy-to-use prediction interface**
- **Professional code structure** with clear documentation

## 🛠️ Customization

You can easily modify the script to:
- Add more machine learning algorithms
- Change visualization styles
- Adjust hyperparameter ranges
- Modify the train-test split ratio
- Add more evaluation metrics

## 📚 Dependencies

- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **matplotlib** - Static plotting
- **seaborn** - Statistical data visualization
- **plotly** - Interactive plotting

## 🤝 Contributing

Feel free to:
- Add new machine learning algorithms
- Improve visualizations
- Enhance documentation
- Optimize performance
- Add new features

## 📄 License

This project is open source and available under the MIT License.

## 🎓 Educational Value

This project is excellent for:
- Learning machine learning fundamentals
- Understanding classification algorithms
- Practicing data visualization
- Exploring hyperparameter tuning
- Comparing model performance

---

**Happy Classifying! 🌺**

For questions or suggestions, please feel free to reach out!

