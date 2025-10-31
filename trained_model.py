# =====================================================
# train_model.py — Create a real trained model
# =====================================================

import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------
# 1. Generate synthetic dataset simulating GO prediction
# -----------------------------------------------------
np.random.seed(42)
num_samples = 500

# Fake sequence features (length, GC%, AT%)
X = np.column_stack([
    np.random.randint(50, 2000, num_samples),        # length
    np.random.rand(num_samples),                     # GC content
    np.random.rand(num_samples)                      # AT content
])

# Fake GO labels
go_labels = ['Label_0', 'Label_1', 'Label_2', 'Label_3', 'Label_4']
y = np.random.choice(go_labels, num_samples)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# -----------------------------------------------------
# 2. Split data
# -----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# -----------------------------------------------------
# 3. Train Random Forest model
# -----------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# -----------------------------------------------------
# 4. Save model (with correct classes)
# -----------------------------------------------------
model.classes_ = le.classes_
with open("trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ trained_model.pkl created successfully.")
