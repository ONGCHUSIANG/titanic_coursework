import json
import os
from datetime import datetime

class ModelEvaluator:
    def __init__(self):
        pass

    # We renamed this from log_experiment to evaluate_model to match main.py
    def evaluate_model(self, model_name, accuracy, best_params=None):
        """Appends the results of a run to a JSON log file."""
        log_file = "experiment_log.json"
        
        new_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_name,
            "accuracy": round(accuracy, 4),
            "parameters": best_params or "Default"
        }
        
        if os.path.exists(log_file):
            with open(log_file, "r") as file:
                logs = json.load(file)
        else:
            logs = []
            
        logs.append(new_entry)
        with open(log_file, "w") as file:
            json.dump(logs, file, indent=4)
            
        print(f"✅ Experiment for {model_name} logged to {log_file}")