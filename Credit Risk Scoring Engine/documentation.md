# Credit Default Prediction & Threshold Optimization

## Project Overview
This project demonstrates a complete machine learning pipeline for predicting loan defaults. It covers synthetic data generation, feature scaling, model training using ensemble methods, and performance optimization through custom threshold tuning.

## 1. Technical Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn
* **Models:** Decision Tree Classifier, Random Forest Classifier
* **Evaluation:** ROC-AUC, Precision-Recall Curve, F1-Score Optimization

## 2. Key Features
* **Data Simulation:** Generated 600 samples with features like `monthly_income`, `credit_score`, and `loan_amount`.
* **Pipeline Architecture:** Utilized `sklearn.pipeline.make_pipeline` to combine `MinMaxScaler` and classifiers for clean, leak-proof training.
* **Threshold Tuning:** Moved beyond the default 0.5 classification boundary to maximize the F1-Score, balancing the cost of False Positives and False Negatives.

## 3. Results & Model Comparison

### Decision Tree Performance
* **Training Accuracy:** 86.25%
* **Testing Accuracy:** 80.00%
* **ROC-AUC:** 0.7935

### Random Forest Performance
* **Training Accuracy:** 87.50%
* **Testing Accuracy:** 83.33%
* **ROC-AUC:** 0.8969

---

## 4. Threshold Optimization
By analyzing the Precision-Recall curve, the optimal threshold was identified to maximize the F1-Score.

| Metric | Default Threshold (0.5) | Optimized Threshold (0.238) |
| :--- | :--- | :--- |
| **Precision** | 0.8182 | 0.6571 |
| **Recall** | 0.3333 | 0.8519 |
| **F1-Score** | 0.4737 | 0.7419 |

## 5. Conclusion
The Random Forest model combined with a tuned classification threshold provides a robust solution for credit risk assessment. The optimization phase successfully improved the balance between catching defaulters (Recall) and maintaining loan approval accuracy (Precision).