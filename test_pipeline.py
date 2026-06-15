import unittest
import pandas as pd
import numpy as np
# Import the custom class you wrote in your preprocessing folder
from preprocessing.preprocessing import TitanicPreprocessor

class TestTitanicPreprocessing(unittest.TestCase):

    def setUp(self):
        """
        This runs automatically BEFORE every single test.
        We create a tiny, fake dataset to test our logic safely.
        """
        self.preprocessor = TitanicPreprocessor()
        
        # Fake data: 3 passengers, one has a missing Age (NaN)
        self.fake_data = pd.DataFrame({
            "PassengerId": [1, 2, 3],
            "Sex": ["male", "female", "male"],
            "Age": [22.0, np.nan, 38.0],  # Passenger 2 is missing an age
            "Fare": [7.25, 71.28, 8.05],
            "Pclass": [3, 1, 3]
        })

    def test_missing_values_are_filled(self):
        """
        Test case to ensure missing numeric values (like Age) 
        are successfully imputed by our preprocessor.
        """
        # 1. Run our actual preprocessing code on the fake data
        cleaned_data = self.preprocessor.clean_data(self.fake_data)
        final_data = self.preprocessor.fill_missing_values(cleaned_data)
        
        # 2. Check if any missing values are left in the 'Age' column
        missing_count = final_data["Age"].isnull().sum()
        
        # 3. Assert statement: We expect the missing count to be exactly 0
        self.assertEqual(missing_count, 0, "Error: Missing ages were not filled!")

if __name__ == "__main__":
    unittest.main()