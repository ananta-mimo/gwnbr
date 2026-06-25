import numpy as np
from gwnbr.models import GWNBRg, GWPR
from gwnbr.bandwidth import BandwidthSelector

# ── 1. Synthetic Maryland-like data ──────────────────────────────────
rng = np.random.default_rng(42)
n = 100

# Fake lat/lon grid (Maryland bounding box)
lon = rng.uniform(-79.5, -75.0, n)
lat = rng.uniform(37.9, 39.7,  n)
coords = np.column_stack([lon, lat])

# Two predictors: income and unemployment (standardized)
income     = rng.standard_normal(n)
unemploy   = rng.standard_normal(n)
X = np.column_stack([income, unemploy])

# Population offset
population = rng.integers(2000, 80000, n).astype(float)
offset     = np.log(population)

# Simulate crash counts (NB with overdispersion)
mu_true = np.exp(-2.0 + (-0.3 * income) + (0.2 * unemploy) + offset)
alpha_true = 0.5
r = 1.0 / alpha_true
p = r / (r + mu_true)
y = rng.negative_binomial(r, p).astype(float)

print(f"y: mean={y.mean():.1f}  var={y.var():.1f}  "
      f"overdispersion ratio={y.var()/y.mean():.2f}")

# ── 2. Fit GWNBRg at a fixed bandwidth first ─────────────────────────
model = GWNBRg(
    coords=coords,
    y=y,
    X=X,
    offset=offset,
    variable_names=["income", "unemployment"]
)

model.fit(bandwidth=150.0, kernel="gaussian")
print(model.summary())

# ── 3. Export results ─────────────────────────────────────────────────
df = model.to_dataframe()
df.to_csv("results_synthetic.csv", index=False)
print("\nSaved: results_synthetic.csv")
print(df[["x", "y_coord", "y_obs", "y_hat",
          "beta_income", "beta_unemployment"]].head(10))