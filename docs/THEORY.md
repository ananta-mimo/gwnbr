# Theory and Methodology

## Table of Contents

1. [Why standard regression fails for spatial count data](#1-why-standard-regression-fails-for-spatial-count-data)
2. [Negative Binomial Regression](#2-negative-binomial-regression)
3. [From global to local: Geographically Weighted Regression](#3-from-global-to-local-geographically-weighted-regression)
4. [Geographically Weighted Poisson Regression (GWPR)](#4-geographically-weighted-poisson-regression-gwpr)
5. [Geographically Weighted Negative Binomial Regression (GWNBR)](#5-geographically-weighted-negative-binomial-regression-gwnbr)
6. [GWNBRg: Global overdispersion variant](#6-gwnbrg-global-overdispersion-variant)
7. [Kernel functions and bandwidth](#7-kernel-functions-and-bandwidth)
8. [Bandwidth selection](#8-bandwidth-selection)
9. [Parameter estimation: NR and IRLS](#9-parameter-estimation-nr-and-irls)
10. [Model diagnostics and fit statistics](#10-model-diagnostics-and-fit-statistics)
11. [Significance testing](#11-significance-testing)
12. [References](#12-references)

---

## 1. Why standard regression fails for spatial count data

Standard regression models make two assumptions that are routinely violated
in spatial count data:

**Stationarity.** A global model produces one coefficient per predictor,
implying the relationship is identical everywhere. In reality, the effect of
income on crash frequency in a dense urban core may be entirely different from
its effect in a rural county. Forcing a single coefficient masks this spatial
heterogeneity and can produce misleading inferences.

**Equidispersion.** The Poisson distribution assumes mean = variance. Count
data — crashes, disease cases, crime incidents — almost always have variance
substantially greater than the mean (overdispersion). Fitting a Poisson model
to overdispersed data underestimates standard errors, producing artificially
significant results.

GWNBR addresses both problems simultaneously: it relaxes stationarity through
the geographically weighted framework and handles overdispersion through the
Negative Binomial distribution.

---

## 2. Negative Binomial Regression

The NB-2 parameterization models the conditional mean as:

```
E[Y_j | X_j] = mu_j = exp(X_j * beta + log(E_j))
```

where `E_j` is an exposure offset (e.g. log population), and the variance is:

```
Var[Y_j] = mu_j + alpha * mu_j^2
```

The overdispersion parameter `alpha >= 0` captures the excess variance beyond
Poisson. When `alpha = 0` the model reduces exactly to Poisson regression.

The log-likelihood for observation j is:

```
l_j = y_j * log(alpha * mu_j)
      - (y_j + 1/alpha) * log(1 + alpha * mu_j)
      + lgamma(y_j + 1/alpha)
      - lgamma(1/alpha)
      - lgamma(y_j + 1)
```

Parameters beta and alpha are estimated by alternating Newton-Raphson (for
alpha) and Iteratively Reweighted Least Squares (for beta) until convergence.

---

## 3. From global to local: Geographically Weighted Regression

Geographically Weighted Regression (GWR), introduced by Fotheringham,
Brunsdon & Charlton (2002), estimates a separate regression for each spatial
unit i by weighting all observations by their proximity to i:

```
beta_i = (X' W_i X)^{-1}  X' W_i y
```

where `W_i = diag(w_i1, w_i2, ..., w_in)` is a diagonal matrix of spatial
kernel weights. Observations close to i receive high weight; distant
observations receive low weight.

The result is a surface of local coefficients — one estimate per spatial unit
— that reveals how relationships vary across the study area.

---

## 4. Geographically Weighted Poisson Regression (GWPR)

GWPR (Nakaya et al., 2005) extends GWR to count data using a Poisson
log-likelihood. For each focal location i, the locally weighted log-likelihood
is:

```
L(beta_i) = sum_j [ w_ij * (y_j * eta_ij - exp(eta_ij)) ]
```

where `eta_ij = X_j * beta_i + log(E_j)` is the linear predictor and
`w_ij` is the spatial kernel weight for observation j at focal location i.

GWPR resolves the stationarity problem but not overdispersion. If the data
are overdispersed, GWPR underestimates standard errors for all local
coefficients.

---

## 5. Geographically Weighted Negative Binomial Regression (GWNBR)

GWNBR (Silva & Rodrigues, 2014) generalises GWPR by replacing the Poisson
with a Negative Binomial distribution, allowing both beta_i and alpha_i to
vary spatially.

The local model for focal location i is:

```
Y_j ~ NB(mu_ij, alpha_i)

log(mu_ij) = X_j * beta_i + log(E_j)
```

The locally weighted log-likelihood is:

```
L(beta_i, alpha_i) = sum_j w_ij * l_j(beta_i, alpha_i)
```

where `l_j` is the NB log-likelihood for observation j (Equation 11,
Silva & Rodrigues 2014) and `w_ij` is the spatial kernel weight.

### Local IRLS weight matrix

The IRLS working weight for the NB-2 distribution (Equation 8, Silva &
Rodrigues 2014) is:

```
A_ij = mu_ij / (1 + alpha_i * mu_ij)
       + (y_j - mu_ij) * alpha_i * mu_ij / (1 + alpha_i * mu_ij)^2
```

### Local covariance matrix

The covariance matrix of the local beta estimates is:

```
Cov(beta_i) = (X' W_i A_i X)^{-1}
```

where `A_i = diag(A_i1, ..., A_in)` is the GLM weight matrix evaluated at
convergence.

### Standard error of alpha

Using the delta method (Equation 15, Silva & Rodrigues 2014), where
`theta_i = 1 / alpha_i`:

```
se(alpha_i) = se(theta_i) / theta_i^2

se(theta_i) = sqrt(1 / |H(theta_i)|)
```

and `H(theta_i)` is the Hessian of the local log-likelihood with respect to
`theta_i`.

---

## 6. GWNBRg: Global overdispersion variant

Full GWNBR produces a local surface of alpha values, but the effective number
of parameters contributed by this surface (k_2) is not analytically
tractable. This prevents the use of AICc for bandwidth selection.

GWNBRg resolves this by estimating a single global alpha from a standard
(non-spatial) NB regression, then using that fixed value in the local IRLS
for beta_i. Because alpha is scalar, its contribution to the effective
parameter count is exactly 1, and AICc bandwidth selection is valid.

```
Global step : fit NB regression -> alpha_global
Local step  : for each i, fit IRLS with alpha = alpha_global -> beta_i
```

GWNBRg is the recommended starting model. It is computationally lighter,
supports AICc bandwidth selection, and in practice produces similar
coefficient surfaces to full GWNBR when overdispersion is relatively uniform
across space.

---

## 7. Kernel functions and bandwidth

The spatial kernel `w_ij` controls how quickly the influence of observation j
on the estimate at i decays with distance.

### Gaussian kernel (fixed bandwidth)

```
w_ij = exp(-0.5 * (d_ij / h)^2)
```

All observations receive positive weight; those beyond ~2h receive negligible
weight. `h` is in the same units as the distances (km for lat/lon data).

### Bisquare kernel (adaptive)

```
w_ij = (1 - (d_ij / h)^2)^2   if d_ij <= h
w_ij = 0                        if d_ij >  h
```

Observations beyond bandwidth h receive exactly zero weight. Produces sharper
local boundaries than Gaussian.

### Adaptive k-NN kernel

The bandwidth adapts per focal location: `h_i` = distance to the k-th
nearest neighbour of i. Useful when data density varies strongly across the
study area (e.g. dense urban tracts vs sparse rural tracts).

### Distance calculation

For latitude/longitude coordinates, distances are computed using the
Haversine formula (great-circle distance):

```
a   = sin^2(dlat/2) + cos(lat_i) * cos(lat_j) * sin^2(dlon/2)
d   = 2 * R * arcsin(sqrt(a))
```

where R = 6371 km. For projected coordinates (metres, feet), Euclidean
distance is used. The package auto-detects coordinate type based on whether
|longitude| < 180.

---

## 8. Bandwidth selection

The bandwidth h controls the degree of spatial smoothing. A small h produces
highly localised (potentially noisy) estimates; a large h approaches the
global model.

### AICc criterion (GWNBRg, GWPR)

The corrected AIC (Hurvich et al., 1998) penalises model complexity:

```
AICc = -2 * L(beta, alpha) + 2k + 2k(k+1) / (n - k - 1)
```

where k = trace(S) + 1 is the effective number of parameters and S is the
hat matrix. AICc is only valid when k_2 = 1 (GWNBRg) or k_2 = 0 (GWPR).

### Cross-validation (GWNBR)

For full GWNBR where k_2 is unknown, bandwidth is selected by minimising
leave-one-out cross-validation error:

```
CV(h) = sum_j (y_j - y_hat_j(-j))^2
```

where `y_hat_j(-j)` is the predicted value for j when j is excluded from
its own local regression.

### Golden Section Search

Both criteria are minimised using the Golden Section Search algorithm
(Fotheringham et al., 2002), which brackets the optimum without requiring
derivatives. The search range is [0, max_distance] for fixed kernels and
[5, n] for adaptive k-NN.

---

## 9. Parameter estimation: NR and IRLS

For each focal location i, GWNBR alternates between two estimation steps
until convergence:

### Newton-Raphson for alpha_i

Maximises the locally weighted log-likelihood with respect to theta_i = 1/alpha_i:

```
theta_i^(m+1) = theta_i^(m) - g(theta_i^(m)) / H(theta_i^(m))
```

Score (first derivative, Equation 13):

```
g(theta) = sum_j w_ij * [psi(theta + y_j) - psi(theta)
                         + log(theta) + 1
                         - log(theta + mu_j)
                         - (theta + y_j) / (theta + mu_j)]
```

Hessian (second derivative, Equation 14):

```
H(theta) = sum_j w_ij * [psi'(theta + y_j) - psi'(theta)
                          + 1/theta
                          - 2/(theta + mu_j)
                          + (y_j + theta) / (theta + mu_j)^2]
```

where `psi` and `psi'` are the digamma and trigamma functions.

### IRLS for beta_i

Given the current alpha_i, updates beta_i by solving the weighted normal
equations:

```
beta_i^(m+1) = (X' W_s A^(m) X)^{-1}  X' W_s A^(m) z^(m)
```

where `W_s = diag(w_i1, ..., w_in)` are the spatial kernel weights,
`A^(m)` is the GLM working weight matrix, and `z^(m)` is the working
response:

```
z_j^(m) = eta_j^(m) + (y_j - mu_j^(m)) / (A_j^(m) * (1 + alpha * mu_j^(m)))
```

Convergence is declared when the change in deviance between iterations
is less than 1e-6.

---

## 10. Model diagnostics and fit statistics

### Deviance

```
D = 2 * sum_j [ y_j * log(y_j / mu_j)
                - (y_j + 1/alpha_j) * log((1 + alpha_j*y_j) / (1 + alpha_j*mu_j)) ]
```

### Pseudo-R2 (deviance-based)

Following Cameron & Windmeijer (1996):

```
R2_dev      = 1 - D / D_null
adj_R2_dev  = 1 - ((n - 1) / (n - k)) * (1 - R2_dev)
```

where `D_null` is the deviance of the intercept-only model.

### Information criteria

```
AIC  = 2k - 2L
AICc = AIC + 2k(k+1) / (n - k - 1)
BIC  = k * log(n) - 2L
```

where `k = trace(S) + 1` is the effective number of parameters.

### Hat matrix

The hat matrix S maps observed to fitted values. Its trace gives the
effective degrees of freedom consumed by the local fitting:

```
S[i, :] = x_i' (X' W_i A_i X)^{-1} (X W_i A_i)'
```

---

## 11. Significance testing

### t-statistics

```
t_ij = beta_ij / se(beta_ij)
```

Two-sided p-values are computed from the standard normal distribution,
following Nakaya et al. (2005).

### Multiple testing correction

Because n separate tests are conducted (one per spatial unit), the
family-wise error rate inflates. Following Silva & Fotheringham (2015),
the adjusted significance level is:

```
alpha_adj = alpha_nominal * (p / k)
```

where p is the number of predictors (including intercept) and k is the
effective number of parameters. A coefficient at location i is declared
locally significant if its p-value < alpha_adj.

---

## 12. References

Cameron, A. C. and Windmeijer, F. A. G. (1996). R-Squared Measures for
Count Data Regression Models with Applications to Health-Care Utilization.
*Journal of Business and Economic Statistics*, 14(2), 209–220.

Fotheringham, A. S., Brunsdon, C. and Charlton, M. (2002).
*Geographically Weighted Regression*. Wiley.

Nakaya, T., Fotheringham, A. S., Brunsdon, C. and Charlton, M. (2005).
Geographically Weighted Poisson Regression for Disease Association Mapping.
*Statistics in Medicine*, 24, 2695–2717.

Silva, A. R. and Rodrigues, T. C. V. (2014). Geographically Weighted
Negative Binomial Regression — Incorporating Overdispersion.
*Statistics and Computing*, 24, 769–783.

Silva, A. R. and Fotheringham, A. S. (2015). The Multiple Testing Issue
in Geographically Weighted Regression.
*Geographical Analysis*, 47(2), 118–136.
