class TitanicPreprocessor:
    def __init__(self):
        """Initialize the preprocessor."""
        pass

    def clean_data(self, df):
        """Applies basic data cleaning steps to a dataframe."""
        # Make a copy so we do not accidentally destroy our original data
        clean_df = df.copy()
        
        # Step 1: Drop columns that don't help us predict survival
        columns_to_drop = ['Ticket', 'Cabin', 'Name']
        clean_df = clean_df.drop(columns_to_drop, axis=1, errors='ignore')
        
        # Step 2: Convert 'Sex' from text to numbers (female=1, male=0)
        if 'Sex' in clean_df.columns:
            clean_df['Sex'] = clean_df['Sex'].map({'female': 1, 'male': 0}).astype(int)
            
        print("✅ Data successfully cleaned!")
        return clean_df
    def fill_missing_values(self, df):
        """Fills blank spaces in the dataset with logical guesses."""
        # Step 1: Fill missing Fare with the median (middle value)
        if 'Fare' in df.columns:
            df['Fare'] = df['Fare'].fillna(df['Fare'].median())
            
        # Step 2: Fill missing Embarked with the most common port ('S')
        if 'Embarked' in df.columns:
            df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
            # Manav also converts the ports to numbers: S=0, C=1, Q=2
            df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
            
        # Step 3: Fill missing Age with the median age
        if 'Age' in df.columns:
            df['Age'] = df['Age'].fillna(df['Age'].median())
            
        print("✅ Missing values successfully filled!")
        return df