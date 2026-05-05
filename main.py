import pandas as pd
import yaml
import logging # 1. ADDED: Import the built-in logging module

from data.data_loader import TitanicDataLoader
from preprocessing.preprocessing import TitanicPreprocessor, engineer_features 
from models.models import LogisticRegressionModel, tune_and_train_rf 
from evaluation.evaluation import ModelEvaluator
from utils.visualization import TitanicVisualizer

# 2. ADDED: Configure the logger (This formats the output beautifully)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

def main():
    # 3. CHANGED: Replaced all print() statements with logging.info()
    logging.info("Initializing pipeline...")
    loader = TitanicDataLoader(train_path=config['data']['train_path'], test_path=config['data']['test_path'])
    train_data, test_data = loader.load_data()
    
    logging.info("Adding advanced features (Titles, Family Size)...")
    train_data = engineer_features(train_data)
    test_data = engineer_features(test_data)
    
    preprocessor = TitanicPreprocessor()
    clean_train = preprocessor.clean_data(train_data)
    clean_test = preprocessor.clean_data(test_data)
    final_train = preprocessor.fill_missing_values(clean_train)
    final_test = preprocessor.fill_missing_values(clean_test)

    logging.info("Generating visualizations...")
    visualizer = TitanicVisualizer()
    visualizer.save_survival_chart(final_train)

    logging.info("Preparing data for modeling...")
    X_train = final_train.drop(["Survived", "PassengerId"], axis=1)
    Y_train = final_train["Survived"]
    X_test = final_test.drop(["PassengerId"], axis=1)
    
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    logging.info("Starting Machine Learning & Hyperparameter Tuning...")
    evaluator = ModelEvaluator()

    # 4. FIXED: I removed the duplicate training block so it only runs once!
    logging.info("Finding the absolute best Random Forest settings (this may take a moment)...")
    best_model = tune_and_train_rf(X_train, Y_train)

    accuracy = best_model.score(X_train, Y_train)
    evaluator.evaluate_model("Tuned_RandomForest", accuracy)
    
    logging.info("Extracting Model Brain (Feature Importances)...")
    visualizer.save_feature_importance_chart(best_model, X_train.columns)

    logging.info("Generating final submission file...")
    predictions = best_model.predict(X_test)
    
    submission = pd.DataFrame({"PassengerId": final_test["PassengerId"], "Survived": predictions})
    submission.to_csv("submission.csv", index=False)
    logging.info("✅ submission.csv successfully created!")

if __name__ == "__main__":
    main()