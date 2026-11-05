# Knesset Elections — Dimensionality Reduction

Assignment solution: data processing, aggregation, sparse-column filtering, and a
from-scratch PCA implementation applied to the Knesset 25 elections ballot-box
dataset, plus an optional Streamlit UI.

## Files

| File | Description |
|---|---|
| `dimensionality_reduction.py` | Core, reusable functions: `load_data`, `group_and_aggregate_data`, `remove_sparse_columns`, `dimensionality_reduction` (PCA from scratch, numpy only). Every function has a docstring. |
| `Knesset_Dimensionality_Reduction.ipynb` | Jupyter notebook demonstrating every function on `knesset_25.xlsx`, including all required tables and the three required visualizations (compare cities, compare parties, Jerusalem vs. Tel Aviv), with written interpretation of the clustering patterns. |
| `streamlit_app.py` | Bonus interactive UI: upload a dataset, group/aggregate it, filter sparse columns, run PCA, and view an interactive scatter plot. |
| `knesset_25.xlsx` | The dataset used in the notebook. |

## Requirements

```
numpy
pandas
matplotlib
plotly
streamlit
openpyxl      # needed by pandas to read .xlsx files
jupyter       # only needed to open/run the notebook
```

Install everything with:

```bash
pip install numpy pandas matplotlib plotly streamlit openpyxl jupyter
```

`scikit-learn` (or any other high-level ML library) is intentionally **not** used —
PCA is implemented from scratch in `dimensionality_reduction.py` using only `numpy`
(mean-centering → covariance matrix → eigen-decomposition → projection onto the
top eigenvectors).

## How to run the Python script(s)

The functions in `dimensionality_reduction.py` are meant to be imported, not run
directly. From the same folder:

```python
from dimensionality_reduction import (
    load_data, group_and_aggregate_data, remove_sparse_columns, dimensionality_reduction
)

df = load_data("knesset_25.xlsx")
agg = group_and_aggregate_data(df.drop(columns=["ballot_code"]), "city_name", "sum")
filtered = remove_sparse_columns(agg, threshold=1000)
reduced = dimensionality_reduction(filtered, num_components=2, meta_columns=["city_name"])
```

## How to run the Jupyter Notebook

```bash
jupyter notebook Knesset_Dimensionality_Reduction.ipynb
```

Then run all cells (`Cell → Run All`). The notebook expects `knesset_25.xlsx` and
`dimensionality_reduction.py` to be in the same directory. All charts are shown with
Matplotlib (pre-computed, so results are visible even without re-running); the final
section of the notebook contains optional Plotly cells that reproduce the same charts
interactively (hover tooltips) if you have `plotly` installed and run those cells.

## How to interact with the Streamlit UI

Run:

```bash
streamlit run streamlit_app.py
```

This opens the app in your browser (usually `http://localhost:8501`). Steps in the UI:

1. **Upload data** — upload `knesset_25.xlsx` (or any similarly-shaped CSV/Excel file)
   using the file picker in the sidebar.
2. **Group & aggregate** — pick the column to group by (e.g. `city_name`), an
   aggregation function (`sum`, `mean`, `count`, ...), and optionally exclude
   ID-like columns (e.g. `ballot_code`) that shouldn't be aggregated together with
   vote counts.
3. **Remove sparse columns** — use the slider to drop numeric columns (parties) whose
   total is below a chosen threshold.
4. **Dimensionality reduction** — choose **City-wise** (rows = cities) or **Party-wise**
   (the data is transposed so rows = parties, columns = cities) processing, and pick
   the number of PCA components (2 for a 2D scatter, 3+ for a 3D scatter).
5. The app displays the reduced dataset as a table and an interactive Plotly scatter
   plot (2D or 3D depending on the number of components), with hover tooltips showing
   the city/party name. You can also download the reduced data as a CSV.

## Notes on assumptions

- The dataset has no explicit "eligible voters" / turnout column, so **total votes
  cast in a city** is used as a proxy for voter turnout when ranking the "10 cities
  with the highest voter turnout."
- `מעטפות חיצוניות` ("external envelopes") is a special aggregated ballot-box category
  for absentee-style votes, not an actual city — it's included in the aggregated
  tables for completeness but excluded from the city-clustering scatter plots since it
  isn't a real geographic locality.
- PCA is run on **mean-centered but not standardized** vote counts (classic covariance-
  based PCA), so the first principal component is heavily influenced by each city's /
  party's overall vote volume — this is discussed directly in the notebook's
  interpretation sections.
