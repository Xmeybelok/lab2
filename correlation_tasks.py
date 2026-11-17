"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    dif = pandas.read_csv(data.csv_path, sep='\t', na_values=['NA'])
    
    m_dif = dif[dif['Gender'] == 'Male']
    w_dif = dif[dif['Gender'] == 'Female']
    m_count = int(len(m_dif))
    w_count = int(len(w_dif))
    
    features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height']
    m_mri = {}
    w_mri = {}
    
    for feat in features:
        m_mri[feat] = float(m_dif[feat].corr(m_dif['MRI_Count']))
        w_mri[feat] = float(w_dif[feat].corr(w_dif['MRI_Count']))
        
    strongest_feature = ""
    max_abs_corr = -1.0
    
    for feat in features:
        if abs(m_mri[feat]) > max_abs_corr:
            max_abs_corr = abs(m_mri[feat])
            strongest_feature = feat
            
        if abs(w_mri[feat]) > max_abs_corr:
            max_abs_corr = abs(w_mri[feat])
            strongest_feature = feat
            
    return BrainCorrelationSummary(
        m_count = m_count, w_count = w_count,
        w_mri = w_mri, m_mri = m_mri,
        strongest_mri_feature = strongest_feature
    )