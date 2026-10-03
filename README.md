# Boston Housing Regression — Neural Network from Scratch (MATLAB)

An end-to-end implementation of a Multi-Layer Perceptron (MLP) for regression in MATLAB without using machine learning toolboxes.

## 📌 Problem Description
The goal is to predict median house prices (MEDV) using the Boston Housing dataset based on socio-economic, environmental, and structural features. Instead of relying on automated frameworks, the objective was to implement the underlying mathematics manually: data preprocessing, forward propagation, exact backpropagation, and first-order parameter updates.

## 💡 Solution Architecture
- Model: Single hidden-layer MLP (13 features + 1 bias -> 10 hidden units -> 1 output).
- Custom Activation: z = u * cos(u) with analytical derivative dz/du = cos(u) - u * sin(u).
- Loss Function: Mean Squared Error (MSE).
- Optimizers Compared:
  - Momentum Gradient Descent: Accelerates convergence and dampens oscillations using velocity accumulation (beta = 0.9).
  - Mini-batch SGD: Evaluates gradients over randomized subsets (batch_size = 32) to reduce computational cost per iteration.
- Evaluation: Convergence tracking (loss & gradient norm history) and test evaluation using MSE and R^2 score.

## 📂 Project Structure
- `main.m` — Pipeline entry point (data loading, Z-score normalization, 80/20 train-test split, optimization loops, and plots).
- `BostonHousing.csv` — Dataset containing housing features and target values.
- `activation.m` — Vectorized custom activation function.
- `forward.m` — Forward pass computation.
- `loss.m` — MSE objective function.
- `gradient.m` — Analytical gradients derived via backpropagation.
- `momentum.m` — Gradient Descent with Momentum optimizer.
- `sgd.m` — Mini-batch Stochastic Gradient Descent optimizer.

## 🚀 How to Run
1. Open the repository folder in MATLAB.
2. Run the main script in the Command Window:
   `main`
3. The script outputs final training/testing metrics (MSE, R^2) to the console and generates comparative convergence plots.