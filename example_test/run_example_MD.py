import pandas as pd
import numpy as np
from gwnbr.models import GWNBRg, GWPR
from gwnbr.bandwidth import BandwidthSelector

# Load your census tract data
df = pd.read_csv("your_maryland_data.csv")

# ── Coordinates (lon first, then lat) ────────────────────────────────
coords = df[["longitude", "latitude"]].values

# ── Response ──────────────────────────────────────────────────────────
y = df["total_fi_crashes"].values.astype(float)

# ── Predictors (standardize them — matches your poster methodology) ───
from sklearn.preprocessing import StandardScaler
predictor_cols = [
    "log_pop_density",
    "median_household_income",
    "unemployment_rate",
    "pct_black",
    "pct_female"
]
X_raw = df[predictor_cols].values
X = StandardScaler().fit_transform(X_raw)

# ── Offset: log(population) ───────────────────────────────────────────
offset = np.log(df["population"].values.astype(float))

# ── Step 1: find optimal bandwidth ───────────────────────────────────
selector = BandwidthSelector(
    GWNBRg,
    coords, y, X,
    offset=offset,
    variable_names=predictor_cols,
    kernel="gaussian",
    criterion="aicc",
    verbose=True
)
optimal_bw = selector.search()
print(f"\nOptimal bandwidth: {optimal_bw:.2f} km")

# ── Step 2: fit final model ───────────────────────────────────────────
model = GWNBRg(
    coords=coords,
    y=y,
    X=X,
    offset=offset,
    variable_names=predictor_cols
)
model.fit(bandwidth=optimal_bw, kernel="gaussian")
print(model.summary())

# ── Step 3: save for mapping in R / GeoPandas ─────────────────────────
results = model.to_dataframe()
results["GEOID"] = df["GEOID"].values   # attach tract ID for joining
results.to_csv("gwnbrg_maryland_results.csv", index=False)
print("Saved: gwnbrg_maryland_results.csv")