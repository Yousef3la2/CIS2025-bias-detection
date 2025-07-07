# Bias Detection and Explainability in AI Hiring Model

## 🔍 Overview

This project was built for a 48-hour challenge focused on bias detection and explainability in AI models. The model classifies job applicants as **Hire** or **Not Hire** based on structured features (e.g., age, experience, interview score, etc.).

We explore potential **gender bias**, explain model decisions using SHAP, and apply **reweighing** to mitigate unfair outcomes.

---

## 📁 Dataset

The dataset includes the following features:
- Age
- Gender (sensitive attribute)
- EducationLevel
- ExperienceYears
- PreviousCompanies
- DistanceFromCompany
- InterviewScore
- SkillScore
- PersonalityScore
- RecruitmentStrategy
- HiringDecision (target)

---

## 🛠️ Technologies Used

- Python 3.10+
- pandas, numpy
- scikit-learn
- SHAP (Explainability)
- Matplotlib (for plots)

---

## 🧠 Model

We used a **Random Forest Classifier** due to the structured nature of the dataset. Data was split into training and testing sets (80/20).

```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
```

---

## ⚖️ Fairness Analysis

We measured **Demographic Parity** to compare acceptance rates between Male and Female applicants. Initial analysis revealed a bias favoring Male candidates.

---

## 🔁 Bias Mitigation

Applied **reweighing** to increase the influence of underrepresented gender (Female) during training:

```python
sample_weights = gender_train.apply(lambda g: 1.5 if g == 'Female' else 1.0).values
clf.fit(X_train, y_train, sample_weight=sample_weights)
```

This reduced the demographic parity difference significantly.

---

## 📊 Explainability with SHAP

SHAP was used to interpret model decisions and visualize feature importance.

```python
import shap
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[:, :, 1], X_test)
```

---

## 📈 Results Summary

- **Accuracy**: Balanced across classes
- **Bias Reduced**: Demographic disparity between genders was minimized
- **Most Influential Features**: InterviewScore, SkillScore, ExperienceYears

---

## ▶️ How to Run

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the notebook or Python script:
```bash
python main.py
```

---

## 📄 Report

A 2-page PDF summary of findings is included: `Bias_Detection_Report_Yousef.pdf`

---

## 👨‍💻 Author

- Yousef Alaa
