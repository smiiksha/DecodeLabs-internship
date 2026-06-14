# Project 2 – Data Classification Using AI

## Overview
A supervised learning project that classifies Iris flowers into 3 species (Setosa, Versicolor, Virginica) based on 4 physical measurements, using K-Nearest Neighbors (KNN). Also includes a comparison against a Decision Tree classifier.

## Objective
Implement the full supervised learning pipeline:
1. Load dataset
2. Explore the data
3. Preprocess (feature scaling)
4. Train-test split
5. Train classification model
6. Evaluate performance
7. Predict on new data

## Files

| File | Description |
|------|-------------|
| `iris_classifier.py` | Main implementation – KNN classifier, full pipeline |
| `compare_models.py` | Compares KNN vs Decision Tree accuracy |
| `output_run.txt` | Saved terminal output from a real run |
| `P2_DataClassification_Report.docx` | Full report (dataset description, methodology, evaluation, viva Q&A) |

## Dataset
**Iris Dataset** (built into scikit-learn, no download needed)
- 150 samples, 4 features, 3 balanced classes (50 each)
- Features: sepal length, sepal width, petal length, petal width (all in cm)

## How to Run

```bash
pip install scikit-learn numpy
python iris_classifier.py
python compare_models.py
```

## Results

| Metric | Value |
|--------|-------|
| Algorithm | KNN (k=5) |
| Accuracy | 93.33% |
| Test samples | 30 (10 per class) |
| Misclassified | 2 (Virginica predicted as Versicolor) |

**Comparison:**
| Model | Accuracy |
|-------|----------|
| KNN (k=5) | 93.33% |
| Decision Tree | 93.33% |

## Confusion Matrix (KNN)

```
              Predicted
            Set  Ver  Vir
Actual Set [ 10   0    0 ]
       Ver [  0  10    0 ]
       Vir [  0   2    8 ]
```

## Key Concepts Used
- Feature scaling with `StandardScaler` (fit on train, transform on test)
- Stratified train-test split (`train_test_split(..., stratify=y)`)
- KNN classification (`KNeighborsClassifier`)
- Evaluation: accuracy, confusion matrix, precision, recall, F1-score
- Custom prediction on new input data

## Tech Stack
- Python 3.x
- scikit-learn
- NumPy
