# LOAD REQUIRED LIBRARIES
import numpy as np # works w/ numbers and calculations
import pandas as pd # works w/ tables/datasets

# Visualisation
from matplotlib import pyplot as plt # used it to create graphs, such as the practical’s boxplots

# Datasets
from sklearn.datasets import load_breast_cancer, make_classification

# sklearn	A library full of machine-learning tools
# .datasets	The section of that library containing dataset tools
# load_breast_cancer	The particular tool we want
# from ... import ...	“From this section, bring in this tool so I can use it”

# Data splitting and validation
# Data splitting and validation
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RepeatedStratifiedKFold
)

# Pre-processing
from sklearn.preprocessing import LabelEncoder
# “From the data-preparation section inside sklearn, bring in the tool called LabelEncoder.”
# LabelEncoder converts the answer categories (y) into numbers.

# Evaluation
from sklearn.metrics import accuracy_score, f1_score
# accuracy_score measures how many predictions are correct overall.
# f1_score balances two things:
# - Precision: when the model says “malignant”, how often is it right?
# - Recall: out of all the actual malignant cases, how many does it find?

# Base learners
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Base learners: individual models that can be combined in an ensemble.

# Decision tree: predicts a category by following a series of questions.
from sklearn.tree import DecisionTreeClassifier

# KNN: predicts a category using the nearest, most similar training examples.
# K is the number of neighbours it checks.
from sklearn.neighbors import KNeighborsClassifier

# SVC: predicts a category by learning a boundary separating the categories.
# SVC stands for Support Vector Classifier.
from sklearn.svm import SVC

# These lines only import the tools; they do not train the models.

# Ensemble methods: combine several models to make a final prediction.
# From the ensemble section inside sklearn, import these tools:

from sklearn.ensemble import (
    AdaBoostClassifier,     # Trains models one after another, focusing on earlier mistakes.
    RandomForestClassifier, # Combines decision trees trained using random samples and feature choices.
    BaggingClassifier,      # Trains copies of a chosen model on different random samples, then combines predictions.
    VotingClassifier        # Combines models using votes (hard) or average probabilities (soft).
)

# Brackets let us spread the import across several lines.
# These lines only import the tools; they do not train any models.

# LOAD THE DATASET
breast_cancer = load_breast_cancer()

# Create feature DataFrame
X = pd.DataFrame(
    breast_cancer.data,
    columns=breast_cancer.feature_names
)

# Create categorical target
y = pd.Categorical.from_codes(
    breast_cancer.target,
    breast_cancer.target_names
)

# Convert target labels to integers
encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("Dataset shape:", X.shape)
print("Target classes:", encoder.classes_)

# 1. **How many observations are in the dataset?**
   # 569 observations — each row represents one tumour sample.

# 2. **How many features are available?**
   # 30 features — each column represents a measurement of the tumour.

# 3. **How many target classes are there?**
   # 2 target classes.

# 4. **What are the class labels?**
  # Benign (non-cancerous) and malignant (cancerous).

# Step 2 — Create train/test sets:
# Split the inputs (X) and correct answers (y) into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,                # Features: the measurements used to make predictions.
    y,                # Target: the correct category for each sample.
    test_size=0.25,   # Use 25% for testing and the remaining 75% for training.
    random_state=1,   # Makes the random split repeatable when we rerun this code.
    stratify=y        # Keeps roughly the same proportion of each class in both sets.
)

# Show the number of rows in each set. shape[0] means the number of rows.
print("Training observations:", X_train.shape[0])
print("Testing observations:", X_test.shape[0])

# Why stratify=y?
# It helps maintain approximately the same class proportions in the training and test sets


# Step 3 — Build the AdaBoost model
# A decision stump is a small decision tree that asks one question before predicting a category.
# AdaBoost combines these simple trees, with later trees focusing more on earlier mistakes.

# Create an AdaBoost model and store it as boosting_model.
boosting_model = AdaBoostClassifier(

    # Choose a decision tree as the individual model (weak learner).
    estimator=DecisionTreeClassifier(
        max_depth=1,     # Allow only one split: this makes it a decision stump.
        random_state=1  # Makes the tree's random choices repeatable.
    ),

    n_estimators=200,   # Train up to 200 weak learners.
    random_state=1      # Makes AdaBoost's random choices repeatable.
)

# Train the model using the training measurements and their correct answers.
boosting_model.fit(X_train, y_train)

# Predict categories for the test measurements.
# Store these predictions in boost_predictions.
boost_predictions = boosting_model.predict(X_test)

# Compare the correct test answers with the predictions to calculate accuracy.
boost_accuracy = accuracy_score(
    y_test,             # Correct answers.
    boost_predictions   # Model's predicted answers.
)

# Compare the same answers to calculate F1.
# By default, F1 measures class 1 — malignant in this practical.
boost_f1 = f1_score(
    y_test,
    boost_predictions
)

# Display both scores rounded to two decimal places.
print(f"AdaBoost Accuracy: {boost_accuracy:.2f}")
print(f"AdaBoost F1 Score: {boost_f1:.2f}")

# Code	Simple meaning
# estimator=	Which individual model should AdaBoost use?
# max_depth=1	Each tree can make only one split
# n_estimators=200	Use up to 200 trees; training can stop early
# .fit(X_train, y_train)	Learn from the training data and answers
# .predict(X_test)	Predict answers for the test data

# ACTIVITY 2 - BAGGING WITH RANDOM FOREST
# Random Forest trains trees independently, then combines their predictions. Each tree receives a random sample of training rows,
# allowing repeated rows (bootstrap sampling), and considers random selections of features when choosing a split.

# Create a Random Forest model and name it bagging_model.
bagging_model = RandomForestClassifier(
    n_estimators=200,  # Create 200 decision trees.
    max_depth=1,      # Each tree makes only one split, like in our AdaBoost model.
    random_state=1,   # Makes the random choices repeatable.
    n_jobs=-1         # Use all available CPU cores to help run the work faster.
)

# Train the model using the training measurements and correct answers.
bagging_model.fit(X_train, y_train)

# Predict categories for the test measurements.
# Save the predictions as bagging_predictions.
bagging_predictions = bagging_model.predict(X_test)

# Calculate how many predictions are correct overall.
bagging_accuracy = accuracy_score(
    y_test,             # Actual answers.
    bagging_predictions # Model's predicted answers.
)

# Calculate F1, which balances precision and recall.
bagging_f1 = f1_score(
    y_test,             # Actual answers.
    bagging_predictions # Model's predicted answers.
)
print(f"Random Forest Accuracy: {bagging_accuracy:.2f}")
print(f"Random Forest F1 Score: {bagging_f1:.2f}")

# - boost_predictions — the categories predicted by AdaBoost for the test samples.
# - boost_accuracy — the proportion of AdaBoost predictions that are correct. For example, 0.95 means 95%.
# - boost_f1 — AdaBoost’s balance between precision and recall for malignant cases. Closer to 1 is better.
# - bagging_predictions — the categories predicted by Random Forest for the test samples.
# - bagging_accuracy — the proportion of Random Forest predictions that are correct.
# - bagging_f1 — Random Forest’s balance between precision and recall for malignant cases. Closer to 1 is better.
# - y_test — the actual correct categories, used to check both models’ predictions.
# Here, predicted 0 means benign and 1 means malignant.

# 1. **Which model performed better?**
#    Compare your results. The model with higher accuracy and F1 performed better on both measures. We need your actual scores to name the winner.
#
# 2. **Is accuracy telling the same story as F1?**
#    Yes, if both scores favour the same model. If accuracy favours one model and F1 favours the other, they tell different stories: accuracy measures overall correctness, while F1 focuses on precision and recall for malignant cases.
#
# 3. **Why is AdaBoost considered sequential?**
#    It trains trees one after another. Each new tree focuses more on the cases earlier trees got wrong, so it depends on their results.
#
# 4. **Why can Random Forest trees be trained independently?**
#    Each tree uses its own random training sample and feature choices. It doesn’t need to know how the other trees performed.
#
# 5. **Which approach would you expect to be easier to parallelise?**
#    Random Forest, because its trees can train at the same time on different CPU cores. AdaBoost’s trees depend on earlier trees, so they train in sequence.





