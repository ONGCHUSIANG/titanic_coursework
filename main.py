import pandas as pd
from data_loader import TitanicDataLoader
from preprocessing import TitanicPreprocessor
# Import our new models!
from models import LogisticRegressionModel, RandomForestModel 

def main():
    # 1 & 2. Load and Clean the data
    print("Initializing pipeline...")
    loader = TitanicDataLoader(train_path="data/train.csv", test_path="data/test.csv")
    train_data, test_data = loader.load_data()
    
    preprocessor = TitanicPreprocessor()
    clean_train = preprocessor.clean_data(train_data)
    clean_test = preprocessor.clean_data(test_data)
    
    final_train = preprocessor.fill_missing_values(clean_train)
    final_test = preprocessor.fill_missing_values(clean_test)

    # 3. Separate the inputs (X) from the answers (Y)
    print("\nPreparing data for modeling...")
    X_train = final_train.drop(["Survived", "PassengerId"], axis=1)
    Y_train = final_train["Survived"]
    
    # 4. Loop through the models! (Polymorphism in action)
    print("\nStarting Machine Learning...")
    classifiers = [
        LogisticRegressionModel(),
        RandomForestModel()
    ]
    
    for clf in classifiers:
        # The parent class handles the .train() and .get_accuracy() for both!
        clf.train(X_train, Y_train)
        accuracy = clf.get_accuracy(X_train, Y_train)
        print(f"> {clf.__class__.__name__} Accuracy: {accuracy}%\n")
    # 5. Make Final Predictions for Submission
    print("\nGenerating final submission file...")
    
    # We will use the Random Forest model (the second one in your list, so index 1)
    best_model = classifiers[1] 
    
    # The test data doesn't have a 'Survived' column, so we only need to drop PassengerId
    X_test = final_test.drop(["PassengerId"], axis=1)
    
    # Force the model to guess!
    predictions = best_model.predict(X_test)
    
    # Format the answers exactly how Kaggle expects them
    submission = pd.DataFrame({
        "PassengerId": final_test["PassengerId"],
        "Survived": predictions
    })
    
    # Save it to a brand new CSV file in your main folder
    submission.to_csv("submission.csv", index=False)
    print("✅ submission.csv successfully created in your workspace!")

if __name__ == "__main__":
    main()