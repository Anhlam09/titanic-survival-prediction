# Titanic Survival Prediction

Predicting which passengers survived the Titanic disaster with binary classification in scikit-learn.

## Files
- `my_titanic.py`: the project I'm building step by step
- `titanic.py`: the full reference solution (EDA, feature engineering, pipeline, model comparison, evaluation)
- `data/titanic.csv`: the Kaggle training set (891 passengers)
- `figures/`: charts the scripts save

## Run it
```bash
pip install pandas scikit-learn matplotlib seaborn
python3 titanic.py
```

## Results (reference solution)
| Model | 5-fold CV accuracy |
|---|---|
| Baseline (always "died") | 61.7% |
| Logistic Regression | 82.3% |
| Random Forest | 81.5% |
| Decision Tree (depth 4) | 79.8% |

Accuracy on the held-out test set: **83.2%** (logistic regression), ROC AUC 0.87.

## Key decisions
- **Title extracted from Name** (Mr, Mrs, Miss, Master) captures sex, age, and marital status in one feature.
- **HasCabin** instead of Cabin: Cabin is 77% missing, but whether one was recorded at all is a signal.
- **Imputation inside a Pipeline** so the medians are learned from training data only (no data leakage).
- **Pclass one-hot encoded**: passenger classes are categories, not quantities.
- **Logistic regression chosen** because it scores as well as the random forest and is easier to explain.

