# Titanic Survival Prediction - OOP Pipeline

**Author:** ONG CHU SIANG
**Student ID:** [BS23110188]
**Institution:** Universiti Malaysia Sabah (UMS)

## Project Overview
This repository contains an Object-Oriented Programming (OOP) implementation of a machine learning pipeline designed to predict passenger survival on the Titanic. The project translates a traditional exploratory Kaggle notebook into a structured, modular, and hierarchical Python codebase.

It specifically demonstrates core software engineering principles, including **Encapsulation**, **Inheritance**, and **Polymorphism**, to seamlessly manage, evaluate, and visualize multiple machine learning models.

## Project Structure
The codebase is decomposed into a strict hierarchical folder system to separate the machine learning workflow stages:

* `data/`: Contains the raw datasets (`train.csv`, `test.csv`) and the data loading module (`data_loader.py`).
* `preprocessing/`: Contains `preprocessing.py`, which handles data cleaning, feature engineering, and missing value imputation.
* `models/`: Contains `models.py`, demonstrating Inheritance and Polymorphism for model selection and training (Logistic Regression, Random Forest).
* `evaluation/`: Contains `evaluation.py` for outputting model accuracy metrics.
* `utils/`: Contains `visualization.py`, which uses Matplotlib to generate exploratory charts.
* `main.py`: The central execution script tying all modules together.
* `requirements.txt`: Lists the external Python libraries required (`pandas`, `numpy`, `scikit-learn`, `matplotlib`).

## Outputs Generated
Running the pipeline will automatically generate two files in the root directory and two external html links in the browser:
* `submission.csv`: The final prediction file formatted for Kaggle scoring.
* `survival_chart.png`: A bar chart visualizing the survival distribution of the training data.
* `2 html charts`: Titanic Survival Analysis and Model Feature Importance will be generated in the browser.

## How to Run
You can execute this OOP pipeline either locally on your machine using **Visual Studio Code** or completely in the cloud using **GitHub Codespaces**. Choose the option that fits your current setup:

### Option A: Running Locally in VS Code
Use this method if you want to execute the project directly on your physical machine.

1. Clone this repository to your local machine.
2. Ensure you have Python installed, then install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute the main pipeline:
   ```bash
   python main.py
   ```
4. Upon successful execution, the terminal will output the training accuracy of each model, a `submission.csv` file will be generated in the root directory and 2 html charts will be automatically opened in the browser.

### Option B: Running in the Cloud via GitHub Codespaces

1. Launch the Workspace:
   Click the green <> Code button at the top of this GitHub repository page, toggle to the Codespaces tab, and click Create codespace on main.
2. Access the Terminal Workspace:
   Once the cloud container loads, locate the lower dashboard panel. If the terminal window is hidden, open it via the application menu: ☰ -> Terminal -> New Terminal (or use the Ctrl + ~ shortcut).
3. Initialize the Environment:
   Run the following block to install all required analytic and algorithmic dependencies inside your cloud container:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Project:
   Execute the central system framework:
    ```bash
   python main.py
   ```
5. Interact with the Results:
   The metrics will render instantly inside your terminal log. Your submission.csv and static images will appear in the left file explorer sidebar. Because Codespaces runs in a virtual cloud container, if the interactive HTML links do not open automatically, simply right-click the generated .html files in the sidebar and choose Open Preview to view the dynamic graphics!


