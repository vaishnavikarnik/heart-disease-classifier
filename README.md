# ❤️ Heart Disease Classifier

## 📌 Problem Statement

Heart disease is one of the world’s most pressing health concerns, where early prediction can significantly improve patient outcomes. In this project, I built a machine learning model to classify whether a patient is at risk of heart disease based on clinical features like age, cholesterol levels, blood pressure, and more.

My objective was to explore how data-driven tools can assist healthcare professionals in making faster, more accurate decisions. Through this project, I applied a full data science pipeline — from EDA and model training to evaluation and tuning — while working on a problem that has the potential to impact real lives.

---

## 🎯 Project Objectives

- Analyze the **UCI Heart Disease dataset**
- Perform **EDA** to understand feature distributions and relationships
- Build and evaluate classification models using:
  - ✅ Decision Tree
  - ✅ Random Forest
- Apply **GridSearchCV** to tune Random Forest hyperparameters
- Use **model evaluation metrics** like Accuracy, Precision, Recall, and F1-Score

---

## 🧰 Tools & Libraries

- Python
- Jupyter Notebook
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (`DecisionTreeClassifier`, `RandomForestClassifier`, `GridSearchCV`)

---

## 📊 Dataset
- Source: [UCI Heart Disease Dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data/data)
- Target: `num` → converted to binary (0 = no disease, 1 = heart disease)
- Key features: age, sex, chest pain type, cholesterol, resting BP, fasting blood sugar, ECG, etc.

---

## 🧪 Modeling Workflow

1. **Data Cleaning & Preprocessing**
   - Verified data types and converted target column
   - Handled categorical encoding if needed

2. **Exploratory Data Analysis (EDA)**
   - Correlation matrix
   - Distribution plots
   - Feature-target relationships

3. **Model Training**
   - Trained Decision Tree and Random Forest
   - Compared baseline performance

4. **Hyperparameter Tuning**
   - Used `GridSearchCV` to tune Random Forest

5. **Evaluation**
   - Confusion Matrix
   - Accuracy, Precision, Recall, F1-score

---

## 📈 Results Summary

| Model            | Train Accuracy | Test Accuracy |
|------------------|----------------|---------------|
| Decision Tree    | ~100%          | ~77.17%       |
| Random Forest    | ~99.1%         | **83.12%**    |
| Tuned RF (GridCV)| ~99.4%         | 83.12%        |

- Random Forest was more stable and generalizable.
- Decision Tree overfit the training data.
- GridSearchCV helped explore tuning but yielded marginal gain.

---

## 🔍 Final Interpretation

I found that the **Random Forest model provided better generalization** and outperformed the Decision Tree on unseen data. The ensemble technique helped reduce overfitting and increased reliability for binary classification.

This project reflects my commitment to using machine learning for real-world, meaningful impact — particularly in healthcare.

---

## 🧠 Key Learnings

- How to compare classification algorithms using real metrics
- Gini Index vs Entropy in tree-based splits
- Hyperparameter tuning using `GridSearchCV`
- Importance of EDA in understanding data before modeling

---

## 🗂️ Repository Structure

heart-disease-classifier/
├── Heart Disease Classification.ipynb
├── README.md
└── dataset/

## 🙋‍♀️ About Me

I'm **Vaishnavi Karnik**, an aspiring data analyst passionate about building meaningful projects that combine **technical depth with social impact**.

> 📫 Connect with me: [LinkedIn](https://www.linkedin.com/in/vaishnavi-karnik/)

