from data_loader import TitanicDataLoader
from preprocessing import TitanicPreprocessor

def main():
    # 1. Load the data
    print("Initializing data loader...")
    loader = TitanicDataLoader(train_path="data/train.csv", test_path="data/test.csv")
    train_data, test_data = loader.load_data()
    
    # 2. Clean the data
    print("\nStarting data preprocessing...")
    preprocessor = TitanicPreprocessor()
    
    # Run data through the first cleaning step
    clean_train = preprocessor.clean_data(train_data)
    clean_test = preprocessor.clean_data(test_data)
    
    # 3. Fill missing values
    print("\nHandling missing values...")
    final_train = preprocessor.fill_missing_values(clean_train)
    final_test = preprocessor.fill_missing_values(clean_test)

if __name__ == "__main__":
    main()