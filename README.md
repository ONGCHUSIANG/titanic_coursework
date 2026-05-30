# Titanic Survival Prediction - OOP Pipeline

**Author:** ONG CHU SIANG
**Student ID:** [BS23110188]
**Institution:** Universiti Malaysia Sabah (UMS)

## Project Overview
This repository contains an Object-Oriented Programming (OOP) implementation of a machine learning pipeline designed to predict passenger survival on the Titanic. The project translates a traditional exploratory Kaggle notebook into a structured, modular, and hierarchical Python codebase.

It specifically demonstrates core software engineering principles, including **Encapsulation**, **Inheritance**, and **Polymorphism**, to seamlessly manage, evaluate, and visualize multiple machine learning models.

## System Architecture & Component Mapping

To ensure a clean separation of concerns, the pipeline is decomposed into dedicated, reusable components. Below is a detailed mapping of the structural roles of each file and folder in the repository:

| Component Path | Structural Role & Architectural Function |
| :--- | :--- |
| `main.py` | **Central Pipeline Coordinator:** The root execution engine that orchestrates the entire workflow by importing modules and running processing, training, and evaluation steps in sequence. |
| `config.yaml` | **Configuration Registry:** Centralizes project hyperparameters, relative file directory paths, and random state seeds to keep code logic separate from environment settings. |
| `requirements.txt` | **Ecosystem Dependencies:** Declares the absolute third-party library versions required to run the pipeline safely (`pandas`, `scikit-learn`, etc.). |
| `data/data_loader.py` | **Data Ingestion Module:** Handles robust file-system access, tracking and loading raw CSV inputs into standard data containers. |
| `preprocessing/preprocessing.py` | **Data Engineering Layer:** Encapsulates row cleaning routines, feature transformations, missing value imputations, and predictive variable matrix generation. |
| `models/models.py` | **Algorithmic Engine:** Implements the core **Inheritance** and **Polymorphism** structure, defining unified training and prediction behaviors across multiple model variants. |
| `evaluation/evaluation.py` | **Validation & Analytics:** Calculates comprehensive validation metrics and cross-validation arrays to compare and evaluate model accuracy thresholds. |

## Dependencies & Ecosystem Packages

The pipeline relies on standard Python scientific and machine learning libraries. These dependencies are automatically verified and managed by the ecosystem configuration environment:

| Package Name | Domain Classification | Architectural Application in Pipeline |
| :--- | :--- | :--- |
| `pandas` | Data Manipulation & Structures | Used in `data_loader.py` and `preprocessing.py` for structured DataFrame parsing, tabular alignment, and missing value management. |
| `numpy` | Numerical Computing Vectors | Utilized for low-level matrix transformations, array manipulation, and mathematical indexing operations. |
| `scikit-learn` | Predictive Machine Learning | Powers the entire `models/` and `evaluation/` directories, providing the core algorithmic frameworks for Logistic Regression, Random Forests, and accuracy assessment tracking. |
| `matplotlib` | Static Visual Graphics | Used in `visualization.py` to design and render the static exploratory charts exported directly into the project directory. |
| `plotly` / `bokeh` | High-Fidelity Interactive Plots | Drives the generation of standalone browser interfaces, rendering interactive mouse-hover metrics for both feature priorities and survival distributions. |
| `pyyaml` | Configuration Management | Parses configuration records (`config.yaml`) to feed constant parameters safely to the core runtime pipeline without hardcoding. |

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
   The metrics will render instantly inside your terminal log. The submission.csv and static images will appear in the left file explorer sidebar. Because Codespaces runs in a virtual cloud container, if the interactive HTML links do not open automatically, simply right-click the generated .html files in the sidebar and choose Open Preview to view the dynamic graphics!


