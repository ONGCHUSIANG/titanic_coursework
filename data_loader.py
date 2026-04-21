import pandas as pd

class TitanicDataLoader:
    def __init__(self, train_path, test_path):
        """Setup the file paths when the object is created."""
        self.train_path = train_path
        self.test_path = test_path

    def load_data(self):
        """Read the CSV files and return them as pandas DataFrames."""
        # pandas reads the CSV files and turns them into 2D tables 
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        
        print("✅ Data successfully loaded!")
        print(f"Train dataset shape: {train_df.shape}")
        print(f"Test dataset shape: {test_df.shape}")
        
        return train_df, test_df