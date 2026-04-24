import pandas as pd
from data.data_loader import TitanicDataLoader
from preprocessing.preprocessing import TitanicPreprocessor
from models.models import LogisticRegressionModel, RandomForestModel 
from evaluation.evaluation import ModelEvaluator
from utils.visualization import TitanicVisualizer

def main():
    # 1. Load Data
    print("Initializing pipeline...")
    loader = TitanicDataLoader(train_path="data/train.csv", test_path="data/test.csv")
    train_data, test_data = loader.load_data()
    
    # 2. Preprocess Data
    preprocessor = TitanicPreprocessor()
    clean_train = preprocessor.clean_data(train_data)
    clean_test = preprocessor.clean_data(test_data)
    final_train = preprocessor.fill_missing_values(clean_train)
    final_test = preprocessor.fill_missing_values(clean_test)

    # 3. Visualization
    print("\nGenerating visualizations...")
    visualizer = TitanicVisualizer()
    visualizer.save_survival_chart(final_train)

    # 4. Prepare for Modeling
    print("\nPreparing data for modeling...")
    X_train = final_train.drop(["Survived", "PassengerId"], axis=1)
    Y_train = final_train["Survived"]
    
    # 5. Train & Evaluate Models
    print("\nStarting Machine Learning...")
    classifiers = [LogisticRegressionModel(), RandomForestModel()]
    evaluator = ModelEvaluator()
    
    for clf in classifiers:
        clf.train(X_train, Y_train)
        accuracy = clf.get_accuracy(X_train, Y_train)
        evaluator.evaluate_model(clf.__class__.__name__, accuracy)

    # 6. Final Submission
    print("Generating final submission file...")
    best_model = classifiers[1] 
    X_test = final_test.drop(["PassengerId"], axis=1)
    predictions = best_model.predict(X_test)
    
    submission = pd.DataFrame({"PassengerId": final_test["PassengerId"], "Survived": predictions})
    submission.to_csv("submission.csv", index=False)
    print("✅ submission.csv successfully created!")

if __name__ == "__main__":
    main()