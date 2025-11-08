# SparseGaussian_lorenz
Experimental code for fitting sparse variational Gaussian processes to Lorenz system dynamics using `GPJax`

This repository contains experimental code for fitting a sparse variational Gaussian process (SVGP) to time-series data from the Lorenz attractor using GPJax and JAX. The implementation demonstrates scalable Bayesian inference on chaotic dynamical systems through variational approximation methods.

Problem Setup:
The code models the derivative (vector field) of the Lorenz attractor by learning the finite-difference mapping from 3D state vectors to their temporal increments. Training data consists of approximately 20,000 state-derivative pairs generated from two distinct Lorenz orbits integrated over 50 time units.

Technical Approach:
Sparse Variational Inference: Implements the collapsed variational Gaussian framework following Titsias (2009) variational_families, using 1000 inducing points to approximate the full GP posterior
Inducing Point Initialization: K-means clustering on the training states to strategically place inducing points in regions of high data density
Objective Function: Maximizes the evidence lower bound (ELBO) via the collapsed formulation objectives, which analytically marginalizes variational parameters for computational efficiency
Numerical Stability: Data normalization (zero mean, unit variance), increased jitter (1e-4) for Cholesky decompositions, and conservative learning rate (1e-4) to prevent training collapse
Precision: All computations use float64 precision with JAX's x64 mode enabled
Implementation Details:
The model uses an RBF kernel with learned lengthscale and variance parameters, a zero mean function, and Gaussian likelihood with learned observation noise. Training proceeds for 3000 iterations using the Adam optimizer. The CollapsedVariationalGaussian variational family is appropriate for datasets under 50k points.

Performance:
On held-out test data, the trained model achieves R² values of 0.83, 0.76, and 0.69 across the three output dimensions, with mean absolute errors of 0.20, 0.20, and 0.19 (in normalized space). The final ELBO of approximately -3.05 million indicates successful convergence for this dataset size.

Limitations:
This is experimental code developed through iterative debugging of numerical instabilities inherent to sparse GP inference on chaotic systems. The approach required careful tuning of inducing point count, jitter, learning rate, and data preprocessing. Results may not generalize to other dynamical systems without similar adjustments. The code prioritizes clarity and reproducibility over production-level optimization.

Dependencies:
GPJax (latest version), JAX with x64 support, Optax, scikit-learn (for k-means), NumPy, SciPy, and Matplotlib.
