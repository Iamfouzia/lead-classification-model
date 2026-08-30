
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# 1. Load Data
df = pd.read_excel('leads.xlsx')


# 2. Data Cleaning
# Reviews Count was stored inconsistently (some values had commas or "+"
# signs), which caused Pandas to treat the column as text instead of a
# number. Clean and convert it to a proper integer column.
df['Reviews Count'] = (
    df['Reviews Count']
    .astype(str)
    .str.replace(',', '', regex=False)
    .str.replace('+', '', regex=False)
    .astype(int)
)


# 3. Feature Engineering
# Convert the raw "Social Media Links" column (mostly missing/unique text)
# into a simple binary signal: does the business have a social media
# presence at all.
df['Has_Social_Media'] = df['Social Media Links'].notna().astype(int)


# 4. Feature Selection
# These columns were chosen based on business logic — they realistically
# influence how "digitally ready" or "high priority" a lead is. Unique
# identifiers (name, address, phone) and outcome-related columns
# (call status, follow-up date) are excluded to avoid noise and data
# leakage.
FEATURES = ['Website Available', 'Automation Status', 'Google Rating',
            'Reviews Count', 'Has_Social_Media']
TARGET = 'Lead Priority'


# 5. Encoding
# Convert categorical text columns into numeric labels the model can use.
le_website = LabelEncoder()
df['Website_Available_Encoded'] = le_website.fit_transform(df['Website Available'])

le_automation = LabelEncoder()
df['Automation_Status_Encoded'] = le_automation.fit_transform(df['Automation Status'])

le_priority = LabelEncoder()
df['Lead_Priority_Encoded'] = le_priority.fit_transform(df[TARGET])


# 6. Prepare Features (X) and Target (y)
X = df[['Website_Available_Encoded', 'Automation_Status_Encoded',
        'Google Rating', 'Reviews Count', 'Has_Social_Media']]
y = df['Lead_Priority_Encoded']


# 7. Scaling
# Bring all numeric features to a similar scale so that large-range
# features (e.g. Reviews Count) don't dominate the model unfairly.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 8. Train-Test Split
# stratify=y keeps the High/Medium/Low class proportions consistent
# across both sets — important given the small, imbalanced dataset.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


# 9. Model Training
# class_weight='balanced' gives underrepresented classes (e.g. "Low")
# more attention during training.
model = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight='balanced'
)
model.fit(X_train, y_train)


# 10. Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("=== Model Evaluation ===")
print(f"Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_priority.classes_))


# 11. Cross-Validation
# A single train-test split can be "lucky" or "unlucky" on a small
# dataset. 5-fold cross-validation gives a more reliable performance
# estimate by testing across 5 different splits.
cv_scores = cross_val_score(model, X_scaled, y, cv=5)

print("=== Cross-Validation Results ===")
print(f"Scores per split: {cv_scores}")
print(f"Average Accuracy: {cv_scores.mean() * 100:.2f}%\n")


# 12. Feature Importance
feature_importance_df = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("=== Feature Importance ===")
print(feature_importance_df.to_string(index=False))
print()


# 13. Save Model and Preprocessing Tools
# Save the trained model along with the scaler and encoders so new leads
# can be transformed the exact same way at prediction time.
joblib.dump(model, 'lead_priority_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_website, 'le_website.pkl')
joblib.dump(le_automation, 'le_automation.pkl')
joblib.dump(le_priority, 'le_priority.pkl')

print("=== Model and preprocessing tools saved ===")
print("Files: lead_priority_model.pkl, scaler.pkl, le_website.pkl, "
      "le_automation.pkl, le_priority.pkl")