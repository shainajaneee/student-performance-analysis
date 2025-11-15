import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

# --- Example dataset & model outputs (replace with your actual data) ---
# y_true = actual pass/fail labels
# y_pred = model predictions
# feature_importance = logistic regression coefficients as a DataFrame

# Example feature importance DataFrame
feature_importance = pd.DataFrame({
    'Feature': ['school_MS', 'failures', 'higher_yes', 'age', 'studytime', 'absences', 'goout', 'Medu', 'Fedu', 'address_U'],
    'Coefficient': [-0.8, -1.2, 1.1, 0.3, 0.5, -0.2, -0.3, 0.4, 0.2, 0.1]
})

# Example pass/fail distribution
pass_fail_counts = pd.Series({'pass': 200, 'fail': 50})

# Example confusion matrix
y_true = ['pass']*200 + ['fail']*50
y_pred = ['pass']*190 + ['fail']*10 + ['pass']*15 + ['fail']*35  # Example predictions
cm = confusion_matrix(y_true, y_pred, labels=['pass', 'fail'])

# --- Plotting ---
fig = plt.figure(figsize=(18, 12))

# 1. Top Features (Logistic Regression coefficients)
ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
sns.barplot(
    x='Coefficient', 
    y='Feature', 
    data=feature_importance.sort_values(by='Coefficient', ascending=False), 
    palette='viridis', 
    ax=ax1
)
ax1.set_title('Top Features Influencing Pass/Fail (Logistic Regression)', fontsize=16)
ax1.set_xlabel('Coefficient', fontsize=12)
ax1.set_ylabel('Feature', fontsize=12)

# 2. Confusion Matrix
ax2 = plt.subplot2grid((2, 2), (1, 0))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=['pass', 'fail'], 
    yticklabels=['pass', 'fail'], 
    ax=ax2
)
ax2.set_title('Confusion Matrix', fontsize=14)
ax2.set_xlabel('Predicted', fontsize=12)
ax2.set_ylabel('Actual', fontsize=12)

# 3. Pass vs Fail Distribution
ax3 = plt.subplot2grid((2, 2), (1, 1))
sns.barplot(
    x=pass_fail_counts.index, 
    y=pass_fail_counts.values, 
    palette='pastel', 
    ax=ax3
)
ax3.set_title('Pass vs Fail Distribution', fontsize=14)
ax3.set_ylabel('Number of Students', fontsize=12)

plt.tight_layout()

# --- Save the figure as high-resolution image ---
plt.savefig('student_performance_charts.png', dpi=300, bbox_inches='tight')

# Show the figure
plt.show()
