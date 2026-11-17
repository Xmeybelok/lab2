"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    A = pandas.read_csv(data.csv_path)
    row_count = int(len(A))
    
    missing_by_column = {}
    for col in A.columns:
        missing_by_column[str(col)] = int(A[col].isna().sum())
        
    adults_over_30_count = int((A['Age'] > 30).sum())
    
    mean_age_by_pclass = {}
    survival_rate_by_pclass = {}
    
    for pclass in sorted(A['Pclass'].dropna().unique()):
        subset = A[A['Pclass'] == pclass]
        mean_age_by_pclass[pclass] = float(subset['Age'].mean())
        survival_rate_by_pclass[pclass] = float(subset['Survived'].mean())
        
    top_fares_series = A['Fare'].dropna().sort_values(ascending=False).head(5)
    highest_fares = []

    for val in top_fares_series: highest_fares.append(float(val))
        
    return TitanicSummary(
        row_count = row_count,
        missing_by_column = missing_by_column,
        adults_over_30_count = adults_over_30_count,
        mean_age_by_pclass = mean_age_by_pclass,
        survival_rate_by_pclass = survival_rate_by_pclass,
        highest_fares = highest_fares
    )