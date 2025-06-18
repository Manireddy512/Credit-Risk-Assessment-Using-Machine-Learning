# 💳 Credit Risk Assessment using Machine Learning

This project builds a robust credit risk assessment pipeline using machine learning techniques. The model predicts whether a borrower will default on a loan, helping financial institutions make informed lending decisions.

---

## 📌 Objective

To develop a binary classification system that assesses credit risk using both logistic regression and XGBoost, improved through ensemble techniques, data balancing (SMOTE), and dimensionality reduction (PCA).

---

## 🧰 Technologies Used

- **Python**  
- **Pandas**, **NumPy**  
- **Scikit-learn**  
- **XGBoost**  
- **Matplotlib**, **Seaborn**  
- **SMOTE (Imbalanced-learn)**  
- **Principal Component Analysis (PCA)**

---

## 🧪 ML Pipeline Overview

1. **Data Cleaning**  
   - Handled missing values and duplicate entries  
   - Cleaned 'Dependents' column (e.g., replaced '3+' with 3)

2. **Label Encoding**  
   - Converted categorical columns to numeric

3. **Class Balancing**  
   - Used **SMOTE** to address imbalanced dataset

4. **Dimensionality Reduction**  
   - Applied **PCA** to improve model performance and reduce overfitting

5. **Model Training**  
   - **Logistic Regression**  
   - **XGBoost Classifier**  
   - **Soft Voting Ensemble** for final prediction

6. **Model Evaluation**  
   - Confusion Matrix  
   - Accuracy, Precision, Recall, F1-Score  
   - ROC-AUC Curve

---

## 📊 Results

- Achieved **93% accuracy** using the soft voting ensemble method  
- Logistic Regression and XGBoost combined improved stability across classes  
- PCA and SMOTE significantly boosted generalization

---

## 📚 References

- [SMOTE in Imbalanced-learn](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)  
- [PCA – Scikit-learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)  
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)  
- [Voting Classifier – Scikit-learn](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier)

---
