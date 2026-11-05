"""
dimensionality_reduction.py

Core data-processing and dimensionality-reduction functions for the
Knesset (Israeli parliament) elections ballot-box dataset.

This module implements everything "from scratch" using only numpy and
pandas -- no scikit-learn or other high-level ML libraries are used for
the PCA implementation, per the assignment requirements.

Functions
---------
load_data(filepath)
    Load a CSV or Excel file into a pandas DataFrame.
group_and_aggregate_data(df, group_by_column, agg_func)
    Group a DataFrame by a column and aggregate the remaining columns.
remove_sparse_columns(df, threshold)
    Drop numeric columns whose total sum is below a threshold.
dimensionality_reduction(df, num_components, meta_columns)
    Reduce the dimensionality of a DataFrame using PCA implemented
    from scratch with numpy.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load a dataset from a CSV or Excel file into a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        The file path to a CSV (.csv) or Excel (.xlsx / .xls) dataset.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the loaded data.

    Raises
    ------
    ValueError
        If the file extension is not a recognized CSV/Excel extension.
    """
    _, ext = os.path.splitext(filepath.lower())

    if ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Expected .csv, .xlsx or .xls"
        )

    return df


def group_and_aggregate_data(
    df: pd.DataFrame, group_by_column: str, agg_func
) -> pd.DataFrame:
    """
    Group a DataFrame by a given column and aggregate all other
    (numeric) columns using the supplied aggregation function.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame containing the dataset.
    group_by_column : str
        The column to group data by (e.g. 'city_name').
    agg_func : callable or str
        The aggregation function to apply to each group
        (e.g. 'sum', 'mean', 'count', np.sum, ...).

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with aggregated results, indexed by
        ``group_by_column`` values reset into a regular column.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    grouped = df.groupby(group_by_column, as_index=False)[numeric_cols].agg(agg_func)

    return grouped


def remove_sparse_columns(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Remove numeric columns whose total sum is below a given threshold.

    Non-numeric columns (e.g. identifier / metadata columns such as
    city names) are always kept, regardless of the threshold.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame.
    threshold : int
        The minimum total sum for a (numeric) column to be retained
        in the DataFrame.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with sparse columns removed.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    keep_numeric = [c for c in numeric_cols if df[c].sum() >= threshold]

    return df[non_numeric_cols + keep_numeric]


def dimensionality_reduction(
    df: pd.DataFrame, num_components: int, meta_columns: list[str]
) -> pd.DataFrame:
    """
    Reduce the dimensionality of a DataFrame using Principal Component
    Analysis (PCA), implemented from scratch with numpy (no
    scikit-learn).

    The algorithm:
        1. Separate the metadata columns (kept unchanged) from the
           numeric feature columns (used for PCA).
        2. Mean-center the numeric features.
        3. Compute the covariance matrix of the centered features.
        4. Compute eigenvalues / eigenvectors of the covariance matrix.
        5. Sort eigenvectors by descending eigenvalue and keep the
           top ``num_components``.
        6. Project the centered data onto the selected eigenvectors.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame containing the data to be reduced.
    num_components : int
        The number of principal components to retain.
    meta_columns : list[str]
        A list of metadata columns to exclude from dimensionality
        reduction. These are included in the final output unchanged.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with columns ``PC1..PCn`` (the reduced
        dimensions) followed by the metadata columns.
    """
    meta_columns = list(meta_columns)
    feature_cols = [c for c in df.columns if c not in meta_columns]

    X = df[feature_cols].to_numpy(dtype=float)

    # 1. mean-center the data
    mean_vec = X.mean(axis=0)
    X_centered = X - mean_vec

    # 2. covariance matrix (features x features)
    cov_matrix = np.cov(X_centered, rowvar=False)

    # 3. eigen-decomposition (covariance matrix is symmetric -> eigh)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 4. sort eigenvectors by descending eigenvalue
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # 5. keep the top num_components eigenvectors
    top_eigenvectors = eigenvectors[:, :num_components]

    # 6. project the centered data onto the principal components
    reduced = X_centered @ top_eigenvectors

    component_names = [f"PC{i + 1}" for i in range(num_components)]
    reduced_df = pd.DataFrame(reduced, columns=component_names, index=df.index)

    result = pd.concat([reduced_df, df[meta_columns]], axis=1)

    return result
