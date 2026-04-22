from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# --- THE SUPERCLASS ---
class BaseModel:
    def __init__(self):
        # This will hold the specific scikit-learn model
        self.model = None 

    def train(self, X_train, Y_train):
        """Train the model using the provided data."""
        self.model.fit(X_train, Y_train)
        print(f"✅ {self.__class__.__name__} trained successfully!")

    def get_accuracy(self, X_train, Y_train):
        """Calculate the accuracy score."""
        acc = round(self.model.score(X_train, Y_train) * 100, 2)
        return acc
    def predict(self, X):
        """Use the trained model to make predictions on new data."""
        return self.model.predict(X)

# --- THE SUBCLASSES ---
class LogisticRegressionModel(BaseModel):
    def __init__(self):
        super().__init__() # Call parent setup
        # Assign the specific algorithm
        self.model = LogisticRegression(max_iter=1000) 

class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__() # Call parent setup
        # Assign the specific algorithm
        self.model = RandomForestClassifier(n_estimators=100)