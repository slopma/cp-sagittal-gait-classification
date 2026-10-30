"""
sagittal.py — Procesamiento, particionamiento y modelado para ciclos de marcha sagitales.

Diseñado para clasificar marcha de niños con parálisis cerebral vs controles sanos
utilizando las curvas cinemáticas normalizadas (0–100% ciclo de marcha) de:
  - Cadera (flexión / extensión)
  - Rodilla (flexión / extensión)
  - Tobillo (dorsiflexión / plantiflexión)
en ambos lados (izquierdo y derecho) a partir de los archivos:
  gait_cycles_normalized_sagittal.parquet
"""

from __future__ import annotations

import os
import glob
import csv
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix
)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object


SIGNALS = [
    ('left', 'hip'),
    ('left', 'knee'),
    ('left', 'ankle'),
    ('right', 'hip'),
    ('right', 'knee'),
    ('right', 'ankle'),
]

SIGNAL_NAMES = [
    f"{side}_{joint}" for side, joint in SIGNALS
]


# ──────────────────────────────────────────────────────────────────────────────
# 1. Carga del Dataset Sagital
# ──────────────────────────────────────────────────────────────────────────────

def load_sagittal_dataset(
    derivatives_dir: str = 'derivatives',
    split_csv_path: str = 'derivatives/time_series_prediction/subject_split.csv',
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Carga todos los archivos gait_cycles_normalized_sagittal.parquet y extrae:
      - Curvas cinemáticas completas (N, 6, 101)
      - Características biomecánicas tabulares (N, D)
      - Etiquetas binarias y metadatos por ciclo y sujeto

    Returns:
        df_cycles: DataFrame con metadatos por ciclo (sujeto, trial, ciclo, grupo, is_best).
        X_curves: np.ndarray (N, 6, 101) con ángulos en grados.
        X_tabular: np.ndarray (N, D) con features estadísticos y clínicos.
        y: np.ndarray (N,) (0: control, 1: parálisis/case).
        subjects: np.ndarray (N,) con IDs de los sujetos.
        feature_names: List[str] nombres de las columnas de X_tabular.
    """
    # 1. Mapeo de sujetos a grupo (case vs control)
    subject_group = {}
    if os.path.exists(split_csv_path):
        with open(split_csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                subject_group[row['subject_id']] = row['group']

    parquet_files = sorted(glob.glob(os.path.join(derivatives_dir, '**', 'gait_cycles_normalized_sagittal.parquet'), recursive=True))
    if not parquet_files:
        raise FileNotFoundError(f"No se encontraron archivos gait_cycles_normalized_sagittal.parquet en {derivatives_dir}")

    cycle_metadata = []
    curves_list = []
    tabular_list = []

    for p in parquet_files:
        df_trial = pd.read_parquet(p)
        if df_trial.empty:
            continue

        subj_id = df_trial['subject_id'].iloc[0]
        trial_lbl = df_trial['trial_label'].iloc[0]
        group = subject_group.get(subj_id, 'unknown')
        if group == 'unknown':
            # Determinar por nombre si falta en CSV
            group = 'case' if 'CASO' in str(subj_id).upper() else 'control'

        label = 1 if group == 'case' else 0

        # Procesar por ciclo
        for cid, cdf in df_trial.groupby('cycle_id'):
            # Verificar presencia de ambos lados
            sides_present = set(cdf['side'].unique())
            if not {'left', 'right'}.issubset(sides_present):
                # Omitir ciclos incompletos para consistencia bilateral
                continue

            cycle_signals = []
            valid_cycle = True

            for side, joint in SIGNALS:
                subset = cdf[(cdf['side'] == side) & (cdf['joint'] == joint)].sort_values('normalized_point_index')
                if len(subset) != 101:
                    valid_cycle = False
                    break
                cycle_signals.append(subset['angle_deg'].values)

            if not valid_cycle:
                continue

            # (6, 101)
            cycle_mat = np.array(cycle_signals, dtype=np.float32)
            curves_list.append(cycle_mat)

            # Extraer características clínicas tabulares
            # Para cada una de las 6 señales: mean, std, min, max, ROM
            means = np.mean(cycle_mat, axis=1)
            stds = np.std(cycle_mat, axis=1)
            mins = np.min(cycle_mat, axis=1)
            maxs = np.max(cycle_mat, axis=1)
            roms = maxs - mins

            # Simetría inter-extremidades: |Left - Right|
            sym_rom = np.abs(roms[:3] - roms[3:])      # Hip, Knee, Ankle ROM diff
            sym_mean = np.abs(means[:3] - means[3:])    # Hip, Knee, Ankle Mean diff

            feat_vec = np.concatenate([means, stds, mins, maxs, roms, sym_rom, sym_mean])
            tabular_list.append(feat_vec)

            is_best = bool(cdf['is_best'].iloc[0])
            cycle_metadata.append({
                'subject_id': subj_id,
                'trial_label': trial_lbl,
                'cycle_id': cid,
                'group': group,
                'label': label,
                'is_best': is_best,
                'source_file': p
            })

    df_cycles = pd.DataFrame(cycle_metadata)
    X_curves = np.array(curves_list, dtype=np.float32)
    X_tabular = np.array(tabular_list, dtype=np.float32)
    y = df_cycles['label'].values.astype(int)
    subjects = df_cycles['subject_id'].astype(str).values

    # Generar nombres descriptivos de características
    feature_names = []
    for stat in ['mean', 'std', 'min', 'max', 'rom']:
        for sname in SIGNAL_NAMES:
            feature_names.append(f"{sname}_{stat}")
    for j in ['hip', 'knee', 'ankle']:
        feature_names.append(f"asymmetry_rom_{j}")
    for j in ['hip', 'knee', 'ankle']:
        feature_names.append(f"asymmetry_mean_{j}")

    return df_cycles, X_curves, X_tabular, y, subjects, feature_names


# ──────────────────────────────────────────────────────────────────────────────
# 2. Particionamiento Patient-Wise (Train / Val / Test)
# ──────────────────────────────────────────────────────────────────────────────

def patient_wise_stratified_split(
    y: np.ndarray,
    subjects: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, List[str]]]:
    """
    Divide los datos en conjuntos de Train, Validación y Prueba a nivel SUJETO,
    garantizando estratificación por clase (control vs parálisis).

    Ningún paciente/sujeto tendrá trials o ciclos compartidos entre diferentes particiones.

    Returns:
        train_idx: Índices de los ciclos para entrenamiento.
        val_idx: Índices de los ciclos para validación.
        test_idx: Índices de los ciclos para prueba.
        subjects_split: Diccionario con la lista de sujetos por partición.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Las razones deben sumar 1.0"

    rng = np.random.RandomState(seed)

    # Identificar clase única por sujeto
    df_subjs = pd.DataFrame({'subject': subjects, 'y': y}).drop_duplicates()

    split_subjects = {'train': [], 'val': [], 'test': []}

    for label in [0, 1]:
        label_subjs = list(df_subjs[df_subjs['y'] == label]['subject'])
        rng.shuffle(label_subjs)
        n = len(label_subjs)

        n_test = max(1, int(np.round(n * test_ratio)))
        n_val = max(1, int(np.round(n * val_ratio)))
        n_train = n - n_val - n_test

        split_subjects['train'].extend(label_subjs[:n_train])
        split_subjects['val'].extend(label_subjs[n_train:n_train + n_val])
        split_subjects['test'].extend(label_subjs[n_train + n_val:])

    train_set = set(split_subjects['train'])
    val_set = set(split_subjects['val'])
    test_set = set(split_subjects['test'])

    train_idx = np.where(np.isin(subjects, list(train_set)))[0]
    val_idx = np.where(np.isin(subjects, list(val_set)))[0]
    test_idx = np.where(np.isin(subjects, list(test_set)))[0]

    return train_idx, val_idx, test_idx, split_subjects


# ──────────────────────────────────────────────────────────────────────────────
# 3. Modelos PyTorch (1D-CNN y MLP)
# ──────────────────────────────────────────────────────────────────────────────

class GaitCNN1D(nn.Module if HAS_TORCH else object):
    """
    Red Convolucional 1D diseñada para clasificar series temporales cinemáticas normalizadas.
    Entrada: (B, 6, 101) -> 6 canales cinemáticos, 101 puntos de ciclo.
    """
    def __init__(self, in_channels: int = 6, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class GaitMLP(nn.Module if HAS_TORCH else object):
    """
    Red Perceptrón Multicapa (MLP) para características tabulares de ciclos de marcha.
    """
    def __init__(self, input_dim: int = 36, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Evaluación a Nivel Paciente y Nivel Ciclo
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_patient_level_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    subjects: np.ndarray,
    threshold: float = 0.5
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Evalúa el desempeño a nivel de sujeto promediando las predicciones
    de todos los ciclos/trials de cada paciente.

    Returns:
        metrics: Diccionario con accuracy, balanced_acc, sensitivity, specificity, f1, auc.
        df_patients: DataFrame con el diagnóstico real y predicho por paciente.
    """
    df_eval = pd.DataFrame({
        'subject': subjects,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob
    })

    patient_rows = []
    for subj, group_df in df_eval.groupby('subject'):
        true_label = group_df['y_true'].iloc[0]
        mean_prob = group_df['y_prob'].mean()
        # Votación mayoritaria o umbral de probabilidad promedio
        pred_label = int(mean_prob >= threshold)
        n_cycles = len(group_df)
        patient_rows.append({
            'subject': subj,
            'true_label': true_label,
            'pred_label': pred_label,
            'mean_prob': mean_prob,
            'n_cycles': n_cycles
        })

    df_patients = pd.DataFrame(patient_rows)
    p_true = df_patients['true_label'].values
    p_pred = df_patients['pred_label'].values
    p_prob = df_patients['mean_prob'].values

    acc = accuracy_score(p_true, p_pred)
    bal_acc = balanced_accuracy_score(p_true, p_pred)
    sens = recall_score(p_true, p_pred, pos_label=1, zero_division=0)
    spec = recall_score(p_true, p_pred, pos_label=0, zero_division=0)
    f1 = f1_score(p_true, p_pred, pos_label=1, zero_division=0)
    try:
        auc = roc_auc_score(p_true, p_prob)
    except Exception:
        auc = float('nan')

    metrics = {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'sensitivity': sens,
        'specificity': spec,
        'f1_score': f1,
        'roc_auc': auc,
        'n_patients': len(df_patients)
    }

    return metrics, df_patients
