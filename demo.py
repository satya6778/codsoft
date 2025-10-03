from iris_classification import IrisClassifier
import numpy as np

def quick_demo():
    """Run a quick demonstration of the Iris classifier."""
    print("🌺 IRIS CLASSIFICATION DEMO 🌺")
    print("=" * 50)
    
    # Create classifier
    classifier = IrisClassifier()
    
    # Load and prepare data
    classifier.load_and_explore_data()
    classifier.prepare_data()
    
    # Train models (just a few for demo)
    print("\nTraining models...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    
    # Train Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(classifier.X_train_scaled, classifier.y_train)
    classifier.models['Logistic Regression'] = lr_model
    
    # Train Random Forest
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(classifier.X_train_scaled, classifier.y_train)
    classifier.models['Random Forest'] = rf_model
    
    # Evaluate models
    from sklearn.metrics import accuracy_score
    
    for name, model in classifier.models.items():
        y_pred = model.predict(classifier.X_test_scaled)
        accuracy = accuracy_score(classifier.y_test, y_pred)
        classifier.results[name] = {'accuracy': accuracy, 'predictions': y_pred}
        print(f"{name}: {accuracy:.4f}")
    
    # Demo predictions
    print("\n" + "=" * 50)
    print("DEMO PREDICTIONS")
    print("=" * 50)
    
    # Sample data for each species
    samples = [
        (5.1, 3.5, 1.4, 0.2, "Typical Setosa"),
        (7.0, 3.2, 4.7, 1.4, "Typical Versicolor"),
        (6.3, 3.3, 6.0, 2.5, "Typical Virginica"),
        (6.0, 3.0, 4.5, 1.5, "Borderline Case")
    ]
    
    for sepal_l, sepal_w, petal_l, petal_w, description in samples:
        print(f"\n{description}:")
        print(f"Measurements: SL={sepal_l}, SW={sepal_w}, PL={petal_l}, PW={petal_w}")
        
        # Predict with both models
        for model_name in classifier.models.keys():
            predicted_species, probabilities = classifier.predict_new_sample(
                sepal_l, sepal_w, petal_l, petal_w, model_name
            )
            print(f"{model_name}: {predicted_species} (confidence: {probabilities.max():.3f})")
        print("-" * 30)

if __name__ == "__main__":
    quick_demo()

