# evaluation/evaluation.py

class ModelEvaluator:
    def evaluate_model(self, model_name, accuracy):
        """Prints the evaluation metric for the trained model."""
        print(f"📊 {model_name} Evaluation Complete:")
        print(f"   -> Accuracy Score: {accuracy}%\n")