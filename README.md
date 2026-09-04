# ML From Scratch: Linear & Logistic Regression

Two ML models implemented directly from their mathematical formulas — no scikit-learn, no black boxes. I didn't want to just call `.fit()` and get a number back; I wanted to derive the update rules myself and watch the math actually move the model toward a solution.

## What's in here

- `Linear_Regression.ipynb` — predicts exam scores from study hours
- `Logistic_Regression.ipynb` — predicts Titanic passenger survival from Pclass, Sex, and Age

## Linear Regression

Implemented the full gradient descent update from scratch:

**Hypothesis:** ŷ = θ₀ + θ₁x
**Loss (MSE):** J(θ) = (1/2m) Σ(ŷᵢ − yᵢ)²
**Gradient:** ∂J/∂θⱼ = (1/m) Σ(ŷᵢ − yᵢ)xᵢⱼ
**Update rule:** θⱼ := θⱼ − α · ∂J/∂θⱼ

Every parameter update in this notebook comes directly from that gradient — no library computing it for me.

Final model: `score ≈ 5.1 × study_hours + 39.2`
Test MSE: 95.35, close to training loss — so it generalized, not just memorized.

## Logistic Regression

Same principle, different math. Instead of MSE, this uses the sigmoid function to squash outputs into probabilities, and cross-entropy loss to measure how wrong those probabilities are:

**Hypothesis:** ŷ = σ(θᵀx), where σ(z) = 1 / (1 + e⁻ᶻ)
**Loss (cross-entropy):** J(θ) = −(1/m) Σ [yᵢ log(ŷᵢ) + (1−yᵢ) log(1−ŷᵢ)]
**Gradient:** ∂J/∂θⱼ = (1/m) Σ(ŷᵢ − yᵢ)xᵢⱼ

The gradient formula looks nearly identical to linear regression's — which is one of the more satisfying things I noticed. The sigmoid and cross-entropy setup makes them fall out to the same clean form.

Before training, I had to clean the data (missing ages, encoding `Sex` numerically) and standardize the features — without it, gradient descent was unstable and slow to converge.

Result: **78.77% test accuracy**, improved to **80.45%** by tuning the decision threshold using an ROC curve (AUC 0.86).

## What I actually learned

- Implementing the gradient by hand made it obvious *why* linear and logistic regression are
