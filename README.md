# Titanic Survival Prediction - OOP Pipeline

**Author:** ONG CHU SIANG
**Student ID:** [BS23110188]
**Institution:** Universiti Malaysia Sabah (UMS)

## Project Overview
This repository contains an Object-Oriented Programming (OOP) implementation of a machine learning pipeline designed to predict passenger survival on the Titanic. The project translates a traditional exploratory Kaggle notebook into a structured, modular Python codebase. 

It specifically demonstrates core software engineering principles, including **Inheritance** and **Polymorphism**, to manage and evaluate multiple machine learning models seamlessly.

## Project Structure
The code is divided into single-responsibility modules:

* `data/`: Directory containing the raw Kaggle datasets (`train.csv`, `test.csv`, `gender_submission.csv`).
* `data_loader.py`: Contains the `TitanicDataLoader` class responsible for reading the CSV files into Pandas DataFrames.
* `preprocessing.py`: Contains the `TitanicPreprocessor` class which handles data cleaning, dropping unnecessary columns, mapping categorical variables, and imputing missing values.
* `models.py`: Demonstrates **Inheritance**. It features a `BaseModel` parent class that handles training and prediction logic, and specific child classes (`LogisticRegressionModel`, `RandomForestModel`) that inherit these traits while defining their own specific algorithms.
* `main.py`: The control center of the application. It imports all modules, executes the data pipeline, and uses **Polymorphism** to iterate through a list of different classifier objects, training them and generating the final `submission.csv`.
* `requirements.txt`: Lists the external Python libraries required to run the code.

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