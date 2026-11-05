"""
streamlit_app.py

A minimal Streamlit UI for exploring dimensionality reduction on the
Knesset elections dataset (or any similarly-shaped city/party dataset).

Run locally with:
    streamlit run streamlit_app.py

See README.md for full setup and usage instructions.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from dimensionality_reduction import (
    load_data,
    group_and_aggregate_data,
    remove_sparse_columns,
    dimensionality_reduction,
)

st.set_page_config(page_title="Knesset Elections - Dimensionality Reduction", layout="wide")

st.title("🗳️ Knesset Elections — Dimensionality Reduction Explorer")
st.caption(
    "Upload a ballot-box level dataset (city, ballot box, party vote columns), "
    "aggregate it, reduce its dimensionality with a from-scratch PCA implementation, "
    "and visualize how cities / parties cluster."
)

# ---------------------------------------------------------------------------
# 1. Upload dataset
# ---------------------------------------------------------------------------
st.sidebar.header("1. Upload data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is None:
    st.info("👈 Upload a CSV or Excel file in the sidebar to get started.")
    st.stop()

# load_data expects a filepath; Streamlit gives us a file-like object, so we
# dispatch on the uploaded file's extension ourselves here.
if uploaded_file.name.lower().endswith(".csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

st.subheader("Raw data preview")
st.dataframe(df.head(20), use_container_width=True)
st.caption(f"{df.shape[0]:,} rows x {df.shape[1]:,} columns")

# ---------------------------------------------------------------------------
# 2. Group & aggregate
# ---------------------------------------------------------------------------
st.sidebar.header("2. Group & aggregate")

all_columns = df.columns.tolist()
non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns.tolist() or all_columns

group_by_column = st.sidebar.selectbox(
    "Group by column", options=non_numeric_columns, index=0
)
agg_func_name = st.sidebar.selectbox(
    "Aggregation function", options=["sum", "mean", "count", "median", "max", "min"], index=0
)

# drop any leftover numeric identifier columns (e.g. ballot box code) that
# shouldn't be summed/averaged together with vote counts
id_like_columns = st.sidebar.multiselect(
    "Columns to exclude before aggregating (e.g. an ID/code column)",
    options=[c for c in all_columns if c != group_by_column],
)

df_for_agg = df.drop(columns=id_like_columns) if id_like_columns else df
agg_df = group_and_aggregate_data(df_for_agg, group_by_column, agg_func_name)

st.subheader(f"Aggregated by `{group_by_column}` ({agg_func_name})")
st.dataframe(agg_df, use_container_width=True)

# ---------------------------------------------------------------------------
# 3. Remove sparse columns
# ---------------------------------------------------------------------------
st.sidebar.header("3. Remove sparse columns")
numeric_cols = agg_df.select_dtypes(include=[np.number]).columns.tolist()
max_possible = int(agg_df[numeric_cols].sum().max()) if numeric_cols else 0

threshold = st.sidebar.slider(
    "Minimum total sum to keep a column",
    min_value=0,
    max_value=max(max_possible, 1),
    value=min(1000, max(max_possible, 1)),
    step=50,
)
filtered_df = remove_sparse_columns(agg_df, threshold)
st.caption(
    f"Kept {filtered_df.shape[1] - 1} of {agg_df.shape[1] - 1} numeric columns "
    f"(threshold = {threshold:,})."
)

# ---------------------------------------------------------------------------
# 4. Compare cities vs. parties, and PCA settings
# ---------------------------------------------------------------------------
st.sidebar.header("4. Dimensionality reduction")

view_mode = st.sidebar.radio("Processing mode", options=["City-wise", "Party-wise (transpose)"])

if view_mode == "Party-wise (transpose)":
    meta_col = filtered_df.columns[0]
    wide = filtered_df.set_index(meta_col).T
    wide.index.name = "feature_name"
    pca_input_df = wide.reset_index()
    meta_columns = ["feature_name"]
else:
    meta_columns = [filtered_df.columns[0]]
    pca_input_df = filtered_df

max_components = max(1, min(10, len(pca_input_df.columns) - len(meta_columns)))
num_components = st.sidebar.slider(
    "Number of PCA components", min_value=2, max_value=max_components, value=2
)

reduced_df = dimensionality_reduction(pca_input_df, num_components, meta_columns)

st.subheader(f"Reduced data ({view_mode}, {num_components} components)")
st.dataframe(reduced_df, use_container_width=True)

# ---------------------------------------------------------------------------
# 5. Visualize
# ---------------------------------------------------------------------------
st.subheader("Visualization")
label_col = meta_columns[0]

if num_components >= 3:
    fig = px.scatter_3d(
        reduced_df, x="PC1", y="PC2", z="PC3", hover_name=label_col,
        title="First 3 principal components"
    )
else:
    fig = px.scatter(
        reduced_df, x="PC1", y="PC2", hover_name=label_col,
        title="First 2 principal components"
    )

st.plotly_chart(fig, use_container_width=True)

st.download_button(
    "Download reduced data as CSV",
    data=reduced_df.to_csv(index=False).encode("utf-8"),
    file_name="reduced_data.csv",
    mime="text/csv",
)
