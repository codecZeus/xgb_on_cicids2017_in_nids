# Anomaly Detection in Network Intrusion with XGBoost on CIC-IDS 2017 Dataset

This repository contains Python scripts and a LaTeX report for a project on network intrusion detection. The core of the project involves training an XGBoost model on the CIC-IDS 2017 dataset to identify network anomalies and classify different types of attacks.

## Project Structure

The repository is structured as follows:

- **xgb_on_cicids2017_for_nids.py**: The main Python script for data preprocessing, model training, and evaluation.
- **xgb_on_cicids2017_for_nids.tex**: The LaTeX source file for the project report.
- **xgb_on_cicids2017_for_nids.pdf**: The compiled PDF of the project report.

## Requirements

Before you begin, ensure you have the following prerequisites installed:

- Python 3.x
- A LaTeX distribution (e.g., MiKTeX for Windows, MacTeX for macOS, or TeX Live for Linux)

You can install the required Python libraries using pip:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

## Running the Python Script

To run the anomaly detection script, you must first download the CIC-IDS 2017 dataset.

1. **Download the dataset:**  
   The dataset consists of multiple CSV files. You can download the full dataset from the official CIC-IDS 2017 website or a mirrored source.

2. **Place the dataset in the correct directory:**  
   Extract the downloaded CSV files into a folder. By default, the `xgb_on_cicids2017_for_nids.py` script expects the CSV files to be in a directory named `MachineLearningCVE`.

3. **Update the file path:**  
   Open `xgb_on_cicids2017_for_nids.py` and navigate to line 58. Ensure the file path in the `glob.glob` function correctly points to your dataset directory.

   ```python
   file_paths = glob.glob(f'MachineLearningCVE/*.csv') # Ensure this path is correct
   ```

4. **Run the script:**  
   Execute the Python script from your terminal:

   ```bash
   python3 xgb_on_cicids2017_for_nids.py
   ```

The script will perform data cleaning, preprocessing, model training, and print the evaluation metrics to the console.

## Compiling the LaTeX Report

The project report is provided in `xgb_on_cicids2017_for_nids.tex`. To compile it into a PDF, you will need a LaTeX editor and a compiler.

1. **Open the .tex file:**  
   Open `xgb_on_cicids2017_for_nids.tex` in your LaTeX editor (e.g., TeXstudio, VS Code with LaTeX Workshop, or Overleaf).

2. **Set the compilation options:**  
   It is crucial to set the correct build options for a successful compilation.

   - **LaTeX Build Engine:** Set the build option to `PDFLaTeX`.  
   - **Bibliography Tool:** Set the default bibliography tool to `biber`.

3. **Compile the report:**  
   Run the compilation process from your editor. The compiler will generate `xgb_on_cicids2017_for_nids.pdf`, which contains the full analysis and results of the project.
