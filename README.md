# Project1_DataManagement_P172415
# Iris Flower Classification with Spark MLlib

## Overview
This project performs a classification task on the Iris dataset using Apache Spark MLlib. The goal is to predict the flower species from sepal and petal measurements and compare the performance of three machine learning models.

## Dataset
The analysis uses the Iris dataset, which contains 150 samples, 4 numerical features, and 1 target label. The features are `sepallength`, `sepalwidth`, `petallength`, and `petalwidth`, while the target label is `variety` with three classes: Setosa, Versicolor, and Virginica.

Source of data: UCI Machine Learning Repository https://drive.google.com/file/d/1Dffk2vfnsTpSGiP5cL7F8UW0F3vdIcrs/view?usp=drive_link

## Methodology
The dataset is loaded into a Spark DataFrame and checked for missing values and duplicates. The numerical features are combined into a single feature vector using `VectorAssembler`, and the class labels are encoded using `StringIndexer`.

The dataset is split into training and testing sets using a 70/30 split with a fixed seed 42 for reproducibility. Three classification algorithms are trained and tuned using 3-fold cross-validation and grid search: Decision Tree, Random Forest, and Logistic Regression.

## Results and Key Findings
All three models achieved strong performance on the Iris dataset. The reported metrics are:

- Decision Tree: accuracy 0.9348, precision 0.9411, recall 0.9348, F1-score 0.9356.
- Random Forest: accuracy 0.9565, precision 0.9644, recall 0.9565, F1-score 0.9571.
- Logistic Regression: accuracy 0.9783, precision 0.9804, recall 0.9783, F1-score 0.9785.

Logistic Regression was the best performing model overall. It's strong performance suggests that the Iris dataset is relatively clean and well-structured, with class boundaries that are close to linearly separable.

## How to Reproduce
1. Open the notebook in Google Colab or a Jupyter environment with PySpark support.
2. Upload or place the dataset at `contentdrive/MyDrive/P172415/iris.csv`.
3. Mount Google Drive if using Google Colab.
4. Run the notebook cells.
5. Keep the random seed set to `42` to reproduce the split and results as closely as possible.

## Requirements
- Python 3
- PySpark
- pandas
- matplotlib
- Google Colab or a Spark-enabled local environment

## Project Structure
- `Project1_DataManagement.ipynb`: Main notebook for the analysis.
- `iris.csv`: Dataset used for training and testing.
