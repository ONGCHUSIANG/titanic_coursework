# Titanic Survival Prediction - OOP Pipeline

**Author:** ONG CHU SIANG
**Student ID:** [BS23110188]
**Institution:** Universiti Malaysia Sabah (UMS)

## Project Overview
This repository contains an Object-Oriented Programming (OOP) implementation of a machine learning pipeline designed to predict passenger survival on the Titanic. The project translates a traditional exploratory Kaggle notebook into a structured, modular, and hierarchical Python codebase. 

It specifically demonstrates core software engineering principles, including **Encapsulation**, **Inheritance**, and **Polymorphism**, to manage, evaluate, and visualize multiple machine learning models seamlessly.

## Project Structure
The codebase is decomposed into a strict hierarchical folder system to separate the machine learning workflow stages:

* `data/`: Contains the raw datasets (`train.csv`, `test.csv`) and the data loading module (`data_loader.py`).
* `preprocessing/`: Contains `preprocessing.py`, which handles data cleaning, feature engineering, and missing value imputation.
* `models/`: Contains `models.py`, demonstrating Inheritance and Polymorphism for model selection and training (Logistic Regression, Random Forest).
* `evaluation/`: Contains `evaluation.py` for outputting model accuracy metrics.
* `utils/`: Contains `visualization.py` which uses Matplotlib to generate exploratory charts.
* `main.py`: The central execution script tying all modules together.
* `requirements.txt`: Lists the external Python libraries required (`pandas`, `numpy`, `scikit-learn`, `matplotlib`).

## Outputs Generated
Running the pipeline will automatically generate:
1. `submission.csv`: The final prediction file formatted for Kaggle.
2. `survival_chart.png`: A bar chart visualizing the survival distribution of the training data.

## How to Run

1. Clone this repository to your local machine.
2. Ensure you have Python installed, then install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute the main pipeline:
   ```bash
   python main.py
   ```
4. Upon successful execution, the terminal will output the training accuracy of each model, and a `submission.csv` file will be generated in the root directory.