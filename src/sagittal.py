"""
sagittal.py — Procesamiento, validación cruzada y modelado para ciclos de marcha sagitales.

Diseñado para clasificar marcha de niños con parálisis cerebral vs controles sanos
utilizando las curvas cinemáticas normalizadas (0–100% ciclo de marcha) de:
  - Cadera (flexión / extensión)
  - Rodilla (flexión / extensión)
  - Tobillo (dorsiflexión / plantiflexión)
en ambos lados (izquierdo y derecho) a partir de los archivos:
  gait_cycles_normalized_sagittal.parquet

La evaluación se hace con validación cruzada repetida y agrupada por sujeto: con ~56 sujetos
un único split train/val/test deja 3 casos en test y sus métricas no son concluyentes.
"""

from __future__ import annotations

import copy
import csv
import glob
import os
import warnings
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit

try:
    import torch
    import torch.nn as nn
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

    Un "ciclo" de salida empareja el ciclo `cycle_id = k` del lado izquierdo con el del derecho.
    Esos dos ciclos NO son el mismo paso: sus frames de inicio difieren en torno a medio ciclo,
    por lo que las features de asimetría comparan zancadas consecutivas, no simultáneas.

    Sujetos presentes en `split_csv_path` sin ciclos normalizados se omiten con un warning.

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
    else:
        warnings.warn(f"No existe {split_csv_path}: el grupo se inferirá del nombre del sujeto.")

    parquet_files = sorted(glob.glob(os.path.join(derivatives_dir, '**', 'gait_cycles_normalized_sagittal.parquet'), recursive=True))
    if not parquet_files:
        raise FileNotFoundError(f"No se encontraron archivos gait_cycles_normalized_sagittal.parquet en {derivatives_dir}")

    cycle_metadata = []
    curves_list = []
    tabular_list = []
    inferred_group = set()

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
            inferred_group.add(subj_id)

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

            # Asimetría inter-extremidades: |Left - Right|
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

    if inferred_group:
        warnings.warn(f"Grupo inferido por nombre (ausentes en el CSV): {sorted(inferred_group)}")

    loaded = {m['subject_id'] for m in cycle_metadata}
    missing = sorted(set(subject_group) - loaded)
    if missing:
        warnings.warn(
            f"{len(missing)} de {len(subject_group)} sujetos del CSV no tienen ciclos normalizados "
            f"bilaterales y quedan fuera: {missing}"
        )

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
# 2. Modelo de Deep Learning (1D-CNN)
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


# ──────────────────────────────────────────────────────────────────────────────
# 3. Pesos y métricas a nivel paciente
# ──────────────────────────────────────────────────────────────────────────────

def sample_weights(y: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """
    Peso por ciclo = 1 / n_ciclos del sujeto, reescalado para que cada clase aporte lo mismo.

    Evita que los niños con más ciclos pesen más (hay 5.1 ciclos por trial en controles vs 3.7
    en casos) y reemplaza al `class_weight='balanced'` a nivel ciclo.
    """
    counts = pd.Series(subjects).value_counts()
    w = 1.0 / pd.Series(subjects).map(counts).values.astype(float)
    for c in np.unique(y):
        m = y == c
        w[m] *= 0.5 / w[m].sum()
    return w * len(y)


def patient_probs(subjects: np.ndarray, y: np.ndarray, prob: np.ndarray) -> pd.DataFrame:
    """Promedia la probabilidad de clase positiva de los ciclos de cada sujeto."""
    df = pd.DataFrame({'subject': subjects, 'y': y, 'p': prob})
    return df.groupby('subject').agg(y=('y', 'first'), p=('p', 'mean'), n_cycles=('y', 'size'))


def patient_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """AUC (independiente del umbral) y métricas a un umbral fijo, a nivel sujeto."""
    pred = (p >= threshold).astype(int)
    return {
        'auc': roc_auc_score(y, p) if len(np.unique(y)) == 2 else float('nan'),
        'bal_acc': balanced_accuracy_score(y, pred),
        'sens': recall_score(y, pred, pos_label=1, zero_division=0),
        'spec': recall_score(y, pred, pos_label=0, zero_division=0),
    }


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 2000, seed: int = 0) -> Tuple[float, float]:
    """IC95% percentil del AUC remuestreando sujetos con reemplazo."""
    rng = np.random.RandomState(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Ajuste de modelos (firma común: fit_predict(Xtr, ytr, Xte, w, subjects_tr, seed) -> P(case))
# ──────────────────────────────────────────────────────────────────────────────

def fit_predict_rf(Xtr, ytr, Xte, w=None, subjects_tr=None, seed: int = 42) -> np.ndarray:
    """Random Forest sobre features tabulares, con pesos por sujeto y clase."""
    clf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=seed)
    clf.fit(Xtr, ytr, sample_weight=w)
    return clf.predict_proba(Xte)[:, 1]


def _cnn_epoch(model, loader, loss_fn, optimizer=None) -> float:
    """Una época; devuelve la pérdida media ponderada. Entrena solo si hay optimizador."""
    model.train(optimizer is not None)
    total, norm = 0.0, 0.0
    with torch.set_grad_enabled(optimizer is not None):
        for bx, by, bw in loader:
            loss = (loss_fn(model(bx), by) * bw).sum() / bw.sum()
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * bw.sum().item()
            norm += bw.sum().item()
    return total / norm


def fit_predict_cnn(
    Ctr, ytr, Cte, w=None, subjects_tr=None, seed: int = 42,
    epochs: int = 60, patience: int = 10,
) -> np.ndarray:
    """
    Entrena GaitCNN1D con early stopping sobre un split interno de SUJETOS de train.

    Normaliza por canal con estadísticos del train interno. Devuelve P(case) por ciclo de test.
    """
    torch.manual_seed(seed)

    # Split interno estratificado por sujeto para early stopping (nunca se toca el test).
    s_df = pd.DataFrame({'s': subjects_tr, 'y': ytr}).drop_duplicates('s')
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, i_va = next(sss.split(s_df['s'].values, s_df['y'].values))
    va_mask = np.isin(subjects_tr, s_df['s'].values[i_va])
    tr_mask = ~va_mask

    mu = Ctr[tr_mask].mean(axis=(0, 2), keepdims=True)
    sd = Ctr[tr_mask].std(axis=(0, 2), keepdims=True) + 1e-6
    to_tensor = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32)

    def make_loader(mask, shuffle):
        ds = TensorDataset(
            to_tensor(Ctr[mask]),
            torch.tensor(ytr[mask], dtype=torch.long),
            torch.tensor(w[mask], dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=16, shuffle=shuffle)

    train_loader, val_loader = make_loader(tr_mask, True), make_loader(va_mask, False)
    model = GaitCNN1D(in_channels=6, num_classes=2, dropout=0.3)
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    best_loss, best_state, bad = np.inf, None, 0
    for _ in range(epochs):
        _cnn_epoch(model, train_loader, loss_fn, optimizer)
        val_loss = _cnn_epoch(model, val_loader, loss_fn)
        if val_loss < best_loss - 1e-4:
            best_loss, best_state, bad = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(to_tensor(Cte)), dim=1)[:, 1].numpy()


# ──────────────────────────────────────────────────────────────────────────────
# 5. Validación cruzada repetida y agrupada por sujeto
# ──────────────────────────────────────────────────────────────────────────────

def _subject_table(y: np.ndarray, subjects: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({'s': subjects, 'y': y}).drop_duplicates('s').reset_index(drop=True)


def cross_validate_subjects(
    models: Dict[str, Tuple[Callable, np.ndarray]],
    y: np.ndarray,
    subjects: np.ndarray,
    n_splits: int = 4,
    n_repeats: int = 5,
    seed: int = 42,
) -> Dict[str, List[Dict[str, float]]]:
    """
    CV estratificada repetida a nivel SUJETO. En cada repetición cada sujeto cae una vez en test.

    Args:
        models: {nombre: (fit_predict, X)} con X alineado con `y` y `subjects` (por ciclo).

    Returns:
        oof[nombre][repetición] = {sujeto: P(case) promedio de sus ciclos en test}.
    """
    s_df = _subject_table(y, subjects)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    oof = {name: [dict() for _ in range(n_repeats)] for name in models}

    for k, (tr_s, te_s) in enumerate(cv.split(s_df['s'].values, s_df['y'].values)):
        tr_mask = np.isin(subjects, s_df['s'].values[tr_s])
        te_mask = np.isin(subjects, s_df['s'].values[te_s])
        assert not set(subjects[tr_mask]) & set(subjects[te_mask]), 'Fuga de sujetos entre train y test'
        w_tr = sample_weights(y[tr_mask], subjects[tr_mask])
        for name, (fit_predict, X) in models.items():
            prob = fit_predict(X[tr_mask], y[tr_mask], X[te_mask], w=w_tr, subjects_tr=subjects[tr_mask], seed=seed + k)
            pp = patient_probs(subjects[te_mask], y[te_mask], prob)
            oof[name][k // n_splits].update(pp['p'].to_dict())
    return oof


def summarize_cv(
    oof: Dict[str, List[Dict[str, float]]], y: np.ndarray, subjects: np.ndarray, threshold: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """
    Resume la CV a nivel paciente.

    AUC: media ± sd entre repeticiones (cada repetición evalúa a todos los sujetos una vez) e IC95%
    por bootstrap de sujetos sobre la probabilidad promediada entre repeticiones. Ese IC es
    algo optimista porque el promedio reduce la varianza de las predicciones.
    El resto de métricas depende del umbral fijo y de la calibración de cada modelo.

    Returns:
        df_summary: una fila por modelo.
        p_avg: {nombre: Series (índice = sujeto) con la probabilidad promedio entre repeticiones}.
    """
    y_subj = _subject_table(y, subjects).set_index('s')['y']
    rows, p_avg = [], {}
    for name, reps in oof.items():
        per_rep = pd.DataFrame([
            patient_metrics(y_subj.loc[list(r)].values, np.array(list(r.values())), threshold) for r in reps
        ])
        p = pd.Series({s: np.mean([r[s] for r in reps]) for s in y_subj.index})
        lo, hi = bootstrap_auc_ci(y_subj.loc[p.index].values, p.values)
        p_avg[name] = p
        rows.append({
            'modelo': name,
            'auc_mean': per_rep['auc'].mean(), 'auc_sd': per_rep['auc'].std(),
            'auc_ci_lo': lo, 'auc_ci_hi': hi,
            'bal_acc': per_rep['bal_acc'].mean(), 'sens': per_rep['sens'].mean(), 'spec': per_rep['spec'].mean(),
        })
    return pd.DataFrame(rows), p_avg


def cv_permutation_importance(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    feature_names: List[str],
    n_splits: int = 4,
    n_repeats: int = 5,
    seed: int = 42,
    n_perm: int = 10,
) -> pd.DataFrame:
    """
    Importancia por permutación del Random Forest, medida como caída de AUC (por ciclo) en los
    folds de TEST y promediada entre folds. Más estable que la importancia Gini con features
    correlacionadas, y no se calcula sobre datos vistos en el entrenamiento.
    """
    s_df = _subject_table(y, subjects)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    drops = []
    for k, (tr_s, te_s) in enumerate(cv.split(s_df['s'].values, s_df['y'].values)):
        tr_mask = np.isin(subjects, s_df['s'].values[tr_s])
        te_mask = np.isin(subjects, s_df['s'].values[te_s])
        w_tr = sample_weights(y[tr_mask], subjects[tr_mask])
        clf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=seed + k)
        clf.fit(X[tr_mask], y[tr_mask], sample_weight=w_tr)
        res = permutation_importance(
            clf, X[te_mask], y[te_mask], scoring='roc_auc', n_repeats=n_perm, random_state=seed + k,
        )
        drops.append(res.importances_mean)
    drops = np.array(drops)
    return (
        pd.DataFrame({'feature': feature_names, 'auc_drop_mean': drops.mean(axis=0), 'auc_drop_sd': drops.std(axis=0)})
        .sort_values('auc_drop_mean', ascending=False)
        .reset_index(drop=True)
    )
