"""
Iris Flower Classification Project
=================================

This script implements a machine learning model to classify Iris flowers
into three species (setosa, versicolor, virginica) based on their
sepal and petal measurements.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class IrisClassifier:
    """
    A comprehensive class for Iris flower classification using multiple ML algorithms.
    """
    
    def __init__(self):
        """Initialize the classifier with the Iris dataset."""
        self.iris_data = load_iris()
        self.X = self.iris_data.data
        self.y = self.iris_data.target
        self.feature_names = self.iris_data.feature_names
        self.target_names = self.iris_data.target_names
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_and_explore_data(self):
        """Load the Iris dataset and create a DataFrame for exploration."""
        print("=" * 60)
        print("IRIS FLOWER DATASET EXPLORATION")
        print("=" * 60)
        
        # Create DataFrame
        self.df = pd.DataFrame(self.X, columns=self.feature_names)
        self.df['species'] = [self.target_names[i] for i in self.y]
        
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Features: {self.feature_names}")
        print(f"Target Classes: {self.target_names}")
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        print("\nDataset Info:")
        print(self.df.info())
        
        print("\nBasic Statistics:")
        print(self.df.describe())
        
        print("\nClass Distribution:")
        print(self.df['species'].value_counts())
        
        return self.df
    
    def visualize_data(self):
        """Create comprehensive visualizations of the Iris dataset."""
        print("\n" + "=" * 60)
        print("DATA VISUALIZATION")
        print("=" * 60)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Iris Dataset Analysis', fontsize=16, fontweight='bold')
        
        # 1. Pairplot
        plt.subplot(2, 3, 1)
        sns.scatterplot(data=self.df, x='sepal length (cm)', y='sepal width (cm)', 
                       hue='species', s=100, alpha=0.7)
        plt.title('Sepal Length vs Width')
        
        # 2. Petal measurements
        plt.subplot(2, 3, 2)
        sns.scatterplot(data=self.df, x='petal length (cm)', y='petal width (cm)', 
                       hue='species', s=100, alpha=0.7)
        plt.title('Petal Length vs Width')
        
        # 3. Box plots for all features
        plt.subplot(2, 3, 3)
        df_melted = self.df.melt(id_vars=['species'], 
                                value_vars=self.feature_names,
                                var_name='feature', value_name='value')
        sns.boxplot(data=df_melted, x='feature', y='value', hue='species')
        plt.xticks(rotation=45)
        plt.title('Feature Distribution by Species')
        
        # 4. Correlation heatmap
        plt.subplot(2, 3, 4)
        correlation_matrix = self.df[self.feature_names].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        
        # 5. Histogram of features
        plt.subplot(2, 3, 5)
        for feature in self.feature_names:
            plt.hist(self.df[self.df['species'] == 'setosa'][feature], 
                    alpha=0.5, label=f'setosa - {feature}', bins=10)
        plt.xlabel('Measurement (cm)')
        plt.ylabel('Frequency')
        plt.title('Setosa Feature Distribution')
        plt.legend()
        
        # 6. Violin plots
        plt.subplot(2, 3, 6)
        df_melted = self.df.melt(id_vars=['species'], 
                                value_vars=self.feature_names,
                                var_name='feature', value_name='value')
        sns.violinplot(data=df_melted, x='feature', y='value', hue='species')
        plt.xticks(rotation=45)
        plt.title('Feature Distribution (Violin Plot)')
        
        plt.tight_layout()
        plt.savefig('iris_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Interactive 3D plot using plotly
        self.create_interactive_plot()
    
    def create_interactive_plot(self):
        """Create an interactive 3D plot using Plotly."""
        fig = px.scatter_3d(self.df, 
                           x='sepal length (cm)', 
                           y='sepal width (cm)', 
                           z='petal length (cm)',
                           color='species',
                           title='Interactive 3D Iris Dataset Visualization',
                           labels={'sepal length (cm)': 'Sepal Length (cm)',
                                  'sepal width (cm)': 'Sepal Width (cm)',
                                  'petal length (cm)': 'Petal Length (cm)'})
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Sepal Length (cm)',
                yaxis_title='Sepal Width (cm)',
                zaxis_title='Petal Length (cm)'
            ),
            width=800,
            height=600
        )
        
        fig.write_html('iris_3d_interactive.html')
        print("Interactive 3D plot saved as 'iris_3d_interactive.html'")
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Split and scale the data for training."""
        print("\n" + "=" * 60)
        print("DATA PREPARATION")
        print("=" * 60)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        print(f"Number of features: {self.X_train.shape[1]}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple machine learning models."""
        print("\n" + "=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)
        
        # Define models
        models_config = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Support Vector Machine': SVC(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3)
        }
        
        # Train each model
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            
            # Store results
            self.models[name] = model
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for the best model."""
        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING")
        print("=" * 60)
        
        # Find the best model based on accuracy
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        print(f"Best performing model: {best_model_name}")
        
        # Define parameter grids for tuning
        param_grids = {
            'Logistic Regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'Support Vector Machine': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear', 'poly']
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'K-Nearest Neighbors': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        }
        
        if best_model_name in param_grids:
            print(f"Tuning hyperparameters for {best_model_name}...")
            
            # Get the base model
            base_model = self.models[best_model_name]
            
            # Perform grid search
            grid_search = GridSearchCV(
                base_model, 
                param_grids[best_model_name], 
                cv=5, 
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(self.X_train_scaled, self.y_train)
            
            # Update the model with best parameters
            self.models[f'{best_model_name} (Tuned)'] = grid_search.best_estimator_
            
            # Evaluate tuned model
            y_pred_tuned = grid_search.predict(self.X_test_scaled)
            accuracy_tuned = accuracy_score(self.y_test, y_pred_tuned)
            
            self.results[f'{best_model_name} (Tuned)'] = {
                'accuracy': accuracy_tuned,
                'best_params': grid_search.best_params_,
                'predictions': y_pred_tuned
            }
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Tuned accuracy: {accuracy_tuned:.4f}")
            print(f"Improvement: {accuracy_tuned - self.results[best_model_name]['accuracy']:.4f}")
    
    def evaluate_models(self):
        """Evaluate and compare all trained models."""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[model]['accuracy'] for model in self.results.keys()],
            'Precision': [self.results[model].get('precision', 0) for model in self.results.keys()],
            'Recall': [self.results[model].get('recall', 0) for model in self.results.keys()],
            'F1-Score': [self.results[model].get('f1_score', 0) for model in self.results.keys()],
            'CV Mean': [self.results[model].get('cv_mean', 0) for model in self.results.keys()],
            'CV Std': [self.results[model].get('cv_std', 0) for model in self.results.keys()]
        })
        
        print("Model Performance Comparison:")
        print(results_df.round(4))
        
        # Find best model
        best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
        print(f"\nBest Model: {best_model_name}")
        print(f"Best Accuracy: {results_df['Accuracy'].max():.4f}")
        
        # Visualize model comparison
        self.plot_model_comparison(results_df)
        
        # Detailed classification report for best model
        best_predictions = self.results[best_model_name]['predictions']
        print(f"\nDetailed Classification Report for {best_model_name}:")
        print(classification_report(self.y_test, best_predictions, 
                                  target_names=self.target_names))
        
        # Confusion matrix
        self.plot_confusion_matrix(best_model_name, best_predictions)
        
        return results_df, best_model_name
    
    def plot_model_comparison(self, results_df):
        """Create visualization comparing model performance."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Accuracy comparison
        axes[0, 0].bar(results_df['Model'], results_df['Accuracy'], color='skyblue')
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Precision, Recall, F1 comparison
        x = np.arange(len(results_df))
        width = 0.25
        
        axes[0, 1].bar(x - width, results_df['Precision'], width, label='Precision', color='lightcoral')
        axes[0, 1].bar(x, results_df['Recall'], width, label='Recall', color='lightgreen')
        axes[0, 1].bar(x + width, results_df['F1-Score'], width, label='F1-Score', color='lightblue')
        axes[0, 1].set_title('Precision, Recall, F1-Score Comparison')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(results_df['Model'], rotation=45)
        axes[0, 1].legend()
        
        # Cross-validation scores
        axes[1, 0].bar(results_df['Model'], results_df['CV Mean'], 
                      yerr=results_df['CV Std'], capsize=5, color='orange')
        axes[1, 0].set_title('Cross-Validation Scores')
        axes[1, 0].set_ylabel('CV Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # All metrics radar chart
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, model in enumerate(results_df['Model']):
            values = [results_df.iloc[i][metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            axes[1, 1].plot(angles, values, 'o-', linewidth=2, label=model)
            axes[1, 1].fill(angles, values, alpha=0.25)
        
        axes[1, 1].set_xticks(angles[:-1])
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].set_title('All Metrics Comparison (Radar Chart)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, model_name, predictions):
        """Plot confusion matrix for the best model."""
        cm = confusion_matrix(self.y_test, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.target_names, 
                   yticklabels=self.target_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_new_sample(self, sepal_length, sepal_width, petal_length, petal_width, model_name=None):
        """Predict the species of a new Iris flower sample."""
        if model_name is None:
            # Use the best model
            best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
            model = self.models[best_model_name]
            print(f"Using best model: {best_model_name}")
        else:
            model = self.models[model_name]
            print(f"Using model: {model_name}")
        
        # Prepare the input
        new_sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        new_sample_scaled = self.scaler.transform(new_sample)
        
        # Make prediction
        prediction = model.predict(new_sample_scaled)[0]
        prediction_proba = model.predict_proba(new_sample_scaled)[0]
        
        predicted_species = self.target_names[prediction]
        
        print(f"\nPrediction Results:")
        print(f"Input: Sepal Length={sepal_length}, Sepal Width={sepal_width}, "
              f"Petal Length={petal_length}, Petal Width={petal_width}")
        print(f"Predicted Species: {predicted_species}")
        print(f"Confidence: {prediction_proba[prediction]:.4f}")
        
        print(f"\nProbability for each species:")
        for i, species in enumerate(self.target_names):
            print(f"{species}: {prediction_proba[i]:.4f}")
        
        return predicted_species, prediction_proba
    
    def run_complete_analysis(self):
        """Run the complete Iris classification analysis."""
        print("🌺 IRIS FLOWER CLASSIFICATION PROJECT 🌺")
        print("=" * 60)
        
        # Step 1: Load and explore data
        self.load_and_explore_data()
        
        # Step 2: Visualize data
        self.visualize_data()
        
        # Step 3: Prepare data
        self.prepare_data()
        
        # Step 4: Train models
        self.train_models()
        
        # Step 5: Hyperparameter tuning
        self.hyperparameter_tuning()
        
        # Step 6: Evaluate models
        results_df, best_model = self.evaluate_models()
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"Best performing model: {best_model}")
        print(f"Best accuracy achieved: {results_df['Accuracy'].max():.4f}")
        print("\nGenerated files:")
        print("- iris_analysis.png (Data visualization)")
        print("- iris_3d_interactive.html (Interactive 3D plot)")
        print("- model_comparison.png (Model performance comparison)")
        print("- confusion_matrix.png (Confusion matrix)")
        
        return results_df, best_model

def main():
    """Main function to run the Iris classification project."""
    # Create classifier instance
    classifier = IrisClassifier()
    
    # Run complete analysis
    results_df, best_model = classifier.run_complete_analysis()
    
    # Example predictions
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTIONS")
    print("=" * 60)
    
    # Example 1: Typical setosa
    print("\nExample 1: Typical Setosa")
    classifier.predict_new_sample(5.1, 3.5, 1.4, 0.2)
    
    # Example 2: Typical versicolor
    print("\nExample 2: Typical Versicolor")
    classifier.predict_new_sample(7.0, 3.2, 4.7, 1.4)
    
    # Example 3: Typical virginica
    print("\nExample 3: Typical Virginica")
    classifier.predict_new_sample(6.3, 3.3, 6.0, 2.5)
    
    # Example 4: Borderline case
    print("\nExample 4: Borderline Case")
    classifier.predict_new_sample(6.0, 3.0, 4.5, 1.5)
    
    return classifier, results_df, best_model

if __name__ == "__main__":
    classifier, results, best_model = main()

