# ML From Scratch: Linear & Logistic Regression

Two machine learning models built from scratch using only NumPy and Pandas (no scikit-learn).

## Files
- `Logistic_Regression.ipynb` — predicts Titanic passenger survival (Pclass, Sex, Age → Survived)
- `Linear_Regression.ipynb` — predicts scores from study hours

## What I did

**Logistic Regression:**
- Cleaned the data (filled missing ages, converted Sex to numbers)
- Standardized features so gradient descent wouldn't blow up
- Trained with gradient descent using sigmoid + cross-entropy loss
- Got 78.77% test accuracy, and 80.45% after tuning the threshold using an ROC curve (AUC 0.86)

**Linear Regression:**
- Trained with gradient descent using MSE loss
- Final model: score ≈ 5.1 × study_hours + 39.2
- Test MSE: 95.35, close to training loss, so it generalized well

Both models use the same core loop: predict → compute loss → compute gradient → update weights → repeat.
