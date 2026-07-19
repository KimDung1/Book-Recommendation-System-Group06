# Group 06 - Book Recommendation System

## Overview
This project builds a Book Recommendation System using the **Goodbooks-10k** dataset, fulfilling the requirements for DS 423 — Machine Learning with Large Datasets.

## Group Information
- **Group Number:** 6
- **Interest Area:** Recommendation System
- **Members:** 
  - Lê Kim Dũng (ID: 28211452455) - Data & Analysis Lead
  - Võ Minh Chính (ID: 28212306297) - Modeling & Evaluation Lead

## Setup Instructions
1. **Prerequisites:** 
   - Install Python 3.8+
   - Jupyter Notebook or JupyterLab
2. **Kaggle API:**
   - Place your `kaggle.json` (API Token) in this directory or in `~/.kaggle/` to allow automatic dataset downloading.
3. **Execution:**
   - You can install all dependencies via `pip install -r requirements.txt`.
   - Open `Group06_BookRecommendation_LeKimDung.ipynb`. Run it sequentially. It will download the Goodbooks-10k dataset, perform Data Cleaning, EDA, Feature Engineering, and save the processed data to the `Data/` folder.
   - Open `Group06_BookRecommendation_VoMinhChinh.ipynb`. Run it sequentially to perform Collaborative Filtering (SVD), Content-Based Filtering, Evaluation, and generating Top-N recommendations.

## Files
- `Group06_BookRecommendation_LeKimDung.ipynb`: The primary notebook for Data Collection, Cleaning, EDA, and Feature Engineering.
- `Group06_BookRecommendation_VoMinhChinh.ipynb`: The secondary notebook for Modeling (Collaborative and Content-Based), Evaluation, and Hybrid Recommendation.
- `requirements.txt`: List of all Python dependencies required to run the project.
- `Report.pdf` / `report_template.tex`: The compiled PDF and LaTeX files for the Group Project Report.

## References (APA)
Please refer to the final cell in the Jupyter Notebook for the compiled list of references in APA format.
