from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

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

def tune_and_train_rf(X_train, y_train):
    """Finds the optimal hyperparameters for a Random Forest model."""
    
    # Define the "grid" of settings you want to test
    param_grid = {
        'n_estimators': [50, 100, 200], # Number of trees
        'max_depth': [None, 5, 10, 15], # Maximum depth of each tree
        'min_samples_split': [2, 5, 10] # Minimum samples required to split a node
    }
    
    # Initialize base model
    rf = RandomForestClassifier(random_state=42)
    
    # Setup Grid Search with 5-fold cross-validation
    print("Starting grid search (this might take a few seconds)...")
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    
    # Train and test all combinations
    grid_search.fit(X_train, y_train)
    
    print(f"Best Parameters Found: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")
    
    # Return the optimized model
    return grid_search.best_estimator_