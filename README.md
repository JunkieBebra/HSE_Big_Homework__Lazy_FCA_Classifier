# HSE Lazy FCA Classifier Project

🔗 This project is based on the [Lazy FCA classifier](https://gitlab.com/Al_Toretto/lazy_fca_learning), implemented by Mr. A. Tomat
## Overview

This repository hosts a homework project that implements and evaluates a **Lazy Formal Concept Analysis (Lazy FCA) classifier** and several of its novel modifications. The project focuses on classifying data from an **employee dataset**, providing a comprehensive comparison against popular machine learning baselines.

* **Goal:** Implement, evaluate, and compare Lazy_FCA variants against standard classifiers.
* **Key Focus:** F1-score, interpretability, and data preprocessing speed.
* **Dataset:** Employee dataset.

---

## Project Structure

| File/Folder | Type | Description |
| :--- | :--- | :--- |
| `Graphix.ipynb` | **Notebook** | Exploratory data analysis (EDA): Visual checks, feature distributions, and quality verification. |
| `Preprocessing.ipynb` | **Notebook** | Data cleaning and feature engineering pipeline. Generates the final preprocessed dataset. |
| `Processing.ipynb` | **Notebook** | **Main Experiments:** Implements and runs the Lazy FCA classifier and its modifications. Records primary results. |
| `Other_classifiers.ipynb` | **Notebook** | **Benchmarking:** Runs seven common classification algorithms to compare performance against the Lazy FCA variants. |
| `classifiers/` | **Folder** | Implementation of Lazy FCA and its variants. Key files include `executor.py`, `executor_less.py` (optimization), and `executor_soft_match.py` (soft-matching). |
| `data/` | **Folder** | Contains the original (`Employee.csv`) and the preprocessed/encoded dataset (`cleaned_encoded_data.csv`). |

---

## Quick Setup

### Prerequisites
* **Python 3.8+** is recommended.

### Installation
It's highly recommended to install dependencies in a **virtual environment**.

1. Create a venv and activate it:
    ```bash
    python3 -m venv <venv_name>
    source <venv_name>/bin/activate

2.  Run:
    ```bash
    pip3 install -r requirements.txt


---

## How to Reproduce Experiments

To ensure successful reproduction, the notebooks must be executed **in sequence**.

### 1. Inspect Data & Visuals
* Open `Graphix.ipynb` to review feature relationships, distributions, and missing values.

### 2. Preprocess Data
* Run `Preprocessing.ipynb` end-to-end. This generates the necessary input file, **`data/cleaned_encoded_data.csv`**.

### 3. Run Lazy FCA Experiments
* Open and execute `Processing.ipynb`. This notebook uses the implementations in the `classifiers` folder and performs the main experiments with the Lazy FCA variants.

### 4. Compare with Baseline Classifiers
* Run `Other_classifiers.ipynb` to execute benchmarks using common classifiers and compare standard classification metrics.

> **Order Note:** The classification notebooks rely on the output of `Preprocessing.ipynb`. Run them in the sequence listed above.


---

## 👥 Contributors

* Alexander Standrik
* Danila Nikishov
