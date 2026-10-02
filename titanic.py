# %% [markdown]
# Titanic Survival Prediction
# ===========================
# Goal: predict whether a passenger survived (1) or died (0).
# This is BINARY CLASSIFICATION: the answer is one of two categories.
#
# Run the whole thing:   python3 titanic.py
# Or run cell by cell in VS Code (each "# %%" is a cell).

# %% STEP 0 — Imports
import matplotlib
matplotlib.use("Agg")  # save plots to files instead of opening windows
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

RANDOM_STATE = 42  # a fixed seed means you get the same results on every run

# %% STEP 1 — Load the data and look at it
# Always look at the raw data before doing anything else.
df = pd.read_csv("data/titanic.csv")
print("Shape (rows, columns):", df.shape)
print(df.head(), "\n")
print(df.dtypes, "\n")

# How many values are missing in each column?
print("Missing values per column:")
print(df.isna().sum().sort_values(ascending=False).head(4), "\n")
# Age is missing ~20%, Cabin ~77%, Embarked just 2 rows.

# %% STEP 2 — Exploratory Data Analysis (EDA)
# Question: what is the base survival rate? This matters later, because
# a model is only useful if it beats "always guess the most common answer".
print(f"Overall survival rate: {df['Survived'].mean():.1%}\n")

# Survival rate by group. groupby().mean() on a 0/1 column gives a rate.
for col in ["Sex", "Pclass", "Embarked"]:
    print(df.groupby(col)["Survived"].mean().round(3), "\n")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.barplot(data=df, x="Sex", y="Survived", ax=axes[0])
axes[0].set_title("Survival rate by sex")
sns.barplot(data=df, x="Pclass", y="Survived", ax=axes[1])
axes[1].set_title("Survival rate by class")
sns.histplot(data=df, x="Age", hue="Survived", bins=30, multiple="stack", ax=axes[2])
axes[2].set_title("Age distribution by survival")
plt.tight_layout()
plt.savefig("figures/eda.png", dpi=120)
plt.close()
# FINDING: "women and children first" shows up in the data. Sex is the
# strongest signal, then class. Young children survived at a higher rate.

# %% STEP 3 — Feature engineering
# Make new columns that hold signal the raw columns hide.
def add_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    # Title ("Mr", "Mrs", "Miss", "Master") is inside the Name column.
    # It combines sex, age, and marital status. "Master" = young boy.
    data["Title"] = data["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
    data["Title"] = data["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    common = ["Mr", "Mrs", "Miss", "Master"]
    data.loc[~data["Title"].isin(common), "Title"] = "Rare"

    # Family size = siblings/spouses + parents/children + yourself.
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)

    # Cabin is 77% missing: too sparse to fill in. But whether a cabin was
    # recorded at all is itself a signal (it mostly happened in 1st class).
    data["HasCabin"] = data["Cabin"].notna().astype(int)
    return data

df = add_features(df)
print(df.groupby("Title")["Survived"].agg(["mean", "count"]).round(3), "\n")

# %% STEP 4 — Pick features and split into train and test sets
# Dropped: PassengerId (just a row number), Name (used to make Title),
# Ticket (messy free text), Cabin (replaced by HasCabin).
numeric_features = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
categorical_features = ["Sex", "Pclass", "Embarked", "Title"]
binary_features = ["IsAlone", "HasCabin"]

X = df[numeric_features + categorical_features + binary_features]
y = df["Survived"]

# Hold back 20% as a TEST set the model never sees during training.
# stratify=y keeps the same survival rate in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows\n")

# %% STEP 5 — Data cleaning inside a Pipeline
# WHY a pipeline: if you fill missing Age with the median of the WHOLE
# dataset, test-set information leaks into training. That is called "data
# leakage". A Pipeline learns the median from the training data only, then
# reuses that same value on the test data.
preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),  # median: robust to outliers (Fare has huge ones)
        ("scale", StandardScaler()),                   # put features on a similar scale (helps LogReg)
    ]), numeric_features),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),  # fills the 2 missing Embarked values
        ("onehot", OneHotEncoder(handle_unknown="ignore")),   # text -> 0/1 columns
    ]), categorical_features),
    ("bin", "passthrough", binary_features),
])
# Pclass is a number, but it is treated as a category: class 3 is not
# "3x" class 1. One-hot encoding lets the model learn each class separately.

# %% STEP 6 — Train and compare models with cross-validation
models = {
    "Baseline (always 'died')": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree (depth 4)": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=6, random_state=RANDOM_STATE),
}

# Cross-validation: split the TRAINING set into 5 folds, train on 4 and
# score on the 5th, five times. That gives a more reliable estimate than a
# single split, and the test set stays untouched.
print("5-fold cross-validation accuracy on the training set:")
cv_results = {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
    cv_results[name] = scores.mean()
    print(f"  {name:28s} {scores.mean():.3f} ± {scores.std():.3f}")
print()
# The tree is limited to max_depth=4 on purpose. An unlimited tree memorizes
# the training data (OVERFITTING) and does worse on new passengers.

# %% STEP 7 — Final evaluation on the held-out test set
# Use the test set only once, at the end. Logistic regression is chosen
# here because it scores close to the best model and is the easiest to explain.
final = Pipeline([("prep", preprocess), ("model", LogisticRegression(max_iter=1000))])
final.fit(X_train, y_train)
y_pred = final.predict(X_test)
y_prob = final.predict_proba(X_test)[:, 1]

print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Test ROC AUC:  {roc_auc_score(y_test, y_prob):.3f}\n")
print(classification_report(y_test, y_pred, target_names=["Died", "Survived"]))
# Precision (Survived): of the people predicted to survive, how many did.
# Recall (Survived):    of the people who actually survived, how many were found.
# ROC AUC: how well the model RANKS survivors above non-survivors (0.5 = random, 1 = perfect).

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["Died", "Survived"], cmap="Blues"
)
plt.title("Logistic Regression — test set")
plt.savefig("figures/confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.close()

# %% STEP 8 — Interpret the model: what did it learn?
feature_names = final.named_steps["prep"].get_feature_names_out()
coefs = pd.Series(final.named_steps["model"].coef_[0], index=feature_names).sort_values()
print("Logistic regression coefficients (+ raises survival odds, - lowers them):")
print(coefs.round(2).to_string(), "\n")

# A shallow decision tree is easy to read as a flowchart of rules.
tree = Pipeline([("prep", preprocess),
                 ("model", DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE))])
tree.fit(X_train, y_train)
plt.figure(figsize=(18, 7))
plot_tree(tree.named_steps["model"], feature_names=feature_names,
          class_names=["Died", "Survived"], filled=True, rounded=True, fontsize=9)
plt.savefig("figures/decision_tree.png", dpi=120, bbox_inches="tight")
plt.close()
print("Figures saved to figures/")
