import argparse
import pandas as pd
from ecg import *
from sklearn import metrics
from sklearn import linear_model
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, classification_report


"""
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--detector', nargs='?', type=str, default='hamilton', help='R-peak detector name')
parser.add_argument('-clf', '--classifier', nargs='?', type=str, default='logreg', help='Classification algorithm name')
parser.add_argument("-V", "--version", help="show program version", action="store_true")
args = parser.parse_args()

classifiers = { 
    'logreg': linear_model.LogisticRegression(max_iter=2000), 
    'decisiontree': DecisionTreeClassifier(criterion='entropy'),
    'xgboost': XGBClassifier(), 
    'randomforest': RandomForestClassifier(criterion='entropy'), 
    'extratrees': ExtraTreesClassifier(criterion='entropy'), 
    'bagging': BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=0.5) 
    }


ecg_preprocessor(args.detector)
df = pd.read_csv("ecg_processed.csv")
X = df[['mean_hr', 'std','NFD', 'NSD', 'HRV', 'avNN', 'sdNN', 'RMSSD', 'NN50', 'pNN50', 'pNN20']]
y = df["anxiety"]

estimator = RandomForestClassifier(criterion='entropy')
featureSelector = RFE(estimator) 
featureSelector = featureSelector.fit(X, y)

featuresRanked = list(featureSelector.ranking_) 
selectedFeatures = [] 
for i in range(len(featuresRanked)): 
    if(featuresRanked[i] == 1): 
        selectedFeatures.append(X.columns[i])


X = X[selectedFeatures]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = classifiers[args.classifier]
clf = clf.fit(X_train, y_train)
print(f'Training accuracy: {clf.score(X_train, y_train)}')
print(f'Testing accuracy: {metrics.accuracy_score(y_test, clf.predict(X_test))}')
print(f'Confusion Matrix: {confusion_matrix(y_test, clf.predict(X_test))}')
print(f'Classification Report: {classification_report(y_test, clf.predict(X_test))}')


"""

import argparse
import pandas as pd
from ecg import *
from sklearn import metrics, linear_model
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, classification_report
import traceback
import os

# -------------------- Argument Parsing --------------------
parser = argparse.ArgumentParser(description="ECG Anxiety Detection Pipeline")
parser.add_argument('-d', '--detector', nargs='?', type=str, default='hamilton', help='R-peak detector name')
parser.add_argument('-clf', '--classifier', nargs='?', type=str, default='logreg', help='Classification algorithm name')
args = parser.parse_args()

# -------------------- Classifiers --------------------
classifiers = { 
    'logreg': linear_model.LogisticRegression(max_iter=2000), 
    'decisiontree': DecisionTreeClassifier(criterion='entropy'),
    'xgboost': XGBClassifier(), 
    'randomforest': RandomForestClassifier(criterion='entropy'), 
    'extratrees': ExtraTreesClassifier(criterion='entropy'), 
    'bagging': BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=0.5) 
}

print("\n🚀 Starting ECG Anxiety Detection Pipeline...\n")

# -------------------- Step 1: Preprocess ECG --------------------
try:
    print("[INFO] Running ECG preprocessor...")
    ecg_preprocessor(args.detector)
except Exception as e:
    print(f"❌ ECG preprocessing failed: {e}")
    traceback.print_exc()
    exit(1)

# -------------------- Step 2: Load Processed ECG Data --------------------
if not os.path.exists("ecg_processed.csv"):
    print("❌ 'ecg_processed.csv' not found. Preprocessing may have failed.")
    exit(1)

print("\n[INFO] Loading processed ECG features from 'ecg_processed.csv'...")
try:
    df = pd.read_csv("ecg_processed.csv")
    print(f"[INFO] Total samples loaded: {len(df)}")
except Exception as e:
    print(f"❌ Failed to load 'ecg_processed.csv': {e}")
    traceback.print_exc()
    exit(1)

# -------------------- Step 3: Check for required columns --------------------
feature_columns = ['mean_hr', 'std', 'NFD', 'NSD', 'HRV', 'avNN', 'sdNN', 'RMSSD', 'NN50', 'pNN50', 'pNN20']

# Keep only columns that exist
existing_features = [col for col in feature_columns if col in df.columns]

missing_cols = [col for col in feature_columns if col not in df.columns]

if missing_cols:
    print(f"Missing expected columns: {missing_cols}")
    print("Skipping samples without complete HRV features...")

if not existing_features:
    print("No HRV features available in this dataset. Exiting.")
    exit(0)


# Drop rows with NaNs
df = df.dropna(subset=existing_features)
if df.empty:
    print("⚠️ No valid samples remain after cleaning — cannot continue.")
    exit(0)

# -------------------- Step 4: Prepare Data --------------------
X = df[[col for col in feature_columns if col in df.columns]]
y = df["anxiety"]

print(f"Using {len(existing_features)} features for training.")
print(f"Data shape: {X.shape}, Labels shape: {y.shape}")

# -------------------- Step 5: Feature Selection --------------------
try:
    print("[INFO] Performing feature selection using RFE...")
    estimator = RandomForestClassifier(criterion='entropy')
    featureSelector = RFE(estimator)
    featureSelector = featureSelector.fit(X, y)

    featuresRanked = list(featureSelector.ranking_)
    selectedFeatures = [X.columns[i] for i in range(len(featuresRanked)) if featuresRanked[i] == 1]

    if not selectedFeatures:
        print("⚠️ No top features selected by RFE — using all available features instead.")
        selectedFeatures = X.columns.tolist()

    print(f"[INFO] Selected features: {selectedFeatures}")
    X = X[selectedFeatures]
except Exception as e:
    print(f"⚠️ Feature selection failed: {e}")
    print("⚠️ Continuing with all features.")
    X = X  # keep all

# -------------------- Step 6: Train/Test Split --------------------
print("[INFO] Splitting dataset into training/testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(f"[INFO] Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# -------------------- Step 7: Train Classifier --------------------
try:
    print(f"[INFO] Training classifier: {args.classifier}")
    clf = classifiers.get(args.classifier, linear_model.LogisticRegression(max_iter=2000))
    clf = clf.fit(X_train, y_train)
except Exception as e:
    print(f"❌ Classifier training failed: {e}")
    traceback.print_exc()
    exit(1)

# -------------------- Step 8: Evaluate --------------------
print("\n📊 Model Evaluation Results:")
try:
    train_acc = clf.score(X_train, y_train)
    test_acc = metrics.accuracy_score(y_test, clf.predict(X_test))
    print(f"[RESULT] Training Accuracy: {train_acc:.4f}")
    print(f"[RESULT] Testing Accuracy: {test_acc:.4f}")
    print(f"[RESULT] Confusion Matrix:\n{confusion_matrix(y_test, clf.predict(X_test))}")
    print(f"[RESULT] Classification Report:\n{classification_report(y_test, clf.predict(X_test))}")
except Exception as e:
    print(f"❌ Evaluation failed: {e}")
    traceback.print_exc()

print("\n✅ ECG Anxiety Detection Pipeline completed successfully.")
