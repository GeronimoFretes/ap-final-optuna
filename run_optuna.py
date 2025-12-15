# import numpy as np
# import pandas as pd

# from catboost import CatBoostRegressor, Pool
# from sklearn.model_selection import KFold
# from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

# import optuna

# TARGET_YEAR = 2023

# df = pd.read_parquet('data/final_dataset.parquet')

# df_model = df[df["totpobla_2022"].notna()].select_dtypes('number')

# target_incid_col = f'dengue_incid_{TARGET_YEAR}'

# y_abs_incidence = df_model[target_incid_col]

# df_model['y_log_incidence'] = np.log1p(y_abs_incidence)

# for year in range(2018,2024) :
#     df_model.drop(columns=[f'dengue_reg_{year}',f'dengue_incid_{year}'], inplace=True)
    
# cols_to_exclude = [
#     'y_log_incidence',
#     target_incid_col,
#     'dengue_reg_2023'
# ]

# cols_to_exclude = [c for c in cols_to_exclude if c is not None and c in df_model.columns]

# feature_cols = [c for c in df_model.columns if c not in cols_to_exclude]

# X = df_model[feature_cols].copy()
# y = df_model['y_log_incidence'].copy()

# orthogonal_order = pd.read_csv('data/orthogonal_ordered_features.csv')['features'].tolist()

# # Fix your cross-validation scheme
# N_SPLITS = 5
# RANDOM_SEED = 42

# kf = KFold(
#     n_splits=N_SPLITS,
#     shuffle=True,
#     random_state=RANDOM_SEED
# )

# def objective(trial: optuna.Trial) -> float:
#     # ---------- 1) Top-K feature selection ----------
#     max_k = orthogonal_order
#     top_k = trial.suggest_int("top_k", 20, max_k)
#     selected_features = orthogonal_order[:top_k]
#     X_sub = X[selected_features].copy()
    
#     # ---------- 2) Sample CatBoost hyperparameters ----------
#     # You can tighten/loosen these ranges later if you want
#     params = {
#         "loss_function": "RMSE",
#         "eval_metric": "RMSE",
#         "random_seed": RANDOM_SEED,
#         "verbose": False,
#         "allow_writing_files": False,
#         "task_type": "CPU",  # change to "GPU" if you want & can
        
#         # Search space
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "depth": trial.suggest_int("depth", 4, 10),
#         "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
#         "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
#         "border_count": trial.suggest_int("border_count", 32, 255),
#         "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
#         "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
#         "grow_policy": trial.suggest_categorical(
#             "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
#         ),
#         # training length + early stopping
#         "iterations": trial.suggest_int("iterations", 400, 1500),
#         "od_type": "Iter",
#         "od_wait": 50,
#     }
    
#     # ---------- 3) Cross-validation ----------
#     rmse_scores = []
    
#     for train_idx, valid_idx in kf.split(X_sub):
#         X_train, X_valid = X_sub.iloc[train_idx], X_sub.iloc[valid_idx]
#         y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
#         model = CatBoostRegressor(**params)
        
#         model.fit(
#             X_train, y_train,
#             eval_set=(X_valid, y_valid),
#             use_best_model=True
#         )
        
#         preds = model.predict(X_valid)
#         rmse = np.sqrt(root_mean_squared_error(y_valid, preds))
#         rmse_scores.append(rmse)
    
#     rmse_mean = float(np.mean(rmse_scores))
#     rmse_std = float(np.std(rmse_scores))
    
#     # Save extra info for later analysis
#     trial.set_user_attr("rmse_std", rmse_std)
#     trial.set_user_attr("rmse_folds", rmse_scores)
#     trial.set_user_attr("selected_features", selected_features)
    
#     return rmse_mean


import os
import json
import math
import time
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.trial import TrialState

# ============================================================
# Config via environment variables
# ============================================================
TARGET_YEAR = int(os.getenv("TARGET_YEAR", 2023))
N_SPLITS = int(os.getenv("N_SPLITS", 5))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

PATIENCE_TRIALS = int(os.getenv("PATIENCE_TRIALS", "20"))
MIN_IMPROVE = float(os.getenv("MIN_IMPROVE", "1e-3"))

TIMEOUT = os.getenv("TIMEOUT", "19800")
TIMEOUT = int(TIMEOUT) if TIMEOUT else None

STUDY_STORAGE = os.getenv("STUDY_STORAGE", "sqlite:///optuna_dengue.db")
STUDY_NAME = os.getenv("STUDY_NAME", "dengue_catboost_topk_corr0p7")

DATA_PATH = Path("data/final_dataset.parquet")
ORTHO_FEATURES_CSV = Path("data/orthogonal_ordered_features.csv")
ARTIFACTS_DIR = Path("artifacts")

# Plateau state key in study.user_attrs
PLATEAU_KEY = "plateau_state_rmse_v1"


# ============================================================
# Utilities: param signature (duplicate guard)
# ============================================================
def param_signature(params: dict) -> str:
    """
    Stable signature for a hyperparameter dict.
    Floats are normalized to avoid tiny repr differences.
    """
    def _norm(v):
        if isinstance(v, float):
            return float(
                np.format_float_positional(v, unique=True, precision=12, trim="-")
            )
        return v

    payload = {k: _norm(v) for k, v in sorted(params.items())}
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


# ============================================================
# Plateau helpers (for minimization: lower RMSE is better)
# ============================================================
def _plateau_scan(trials, min_improve: float):
    """
    Scan COMPLETED trials (minimization) and compute:
      - best_value: lowest RMSE seen
      - best_trial: trial number
      - stale: completed trials since last improvement >= min_improve
    """
    complete = [t for t in trials if t.state == TrialState.COMPLETE]
    complete = sorted(complete, key=lambda t: t.number)
    if not complete:
        return {
            "best_value": float("inf"),
            "best_trial": None,
            "stale": 0,
            "min_improve": float(min_improve),
        }

    best = complete[0].value
    best_trial = complete[0].number
    stale = 0

    for t in complete[1:]:
        v = t.value
        if v is None or not math.isfinite(v):
            stale += 1
            continue

        # minimization: improvement if RMSE decreases by at least min_improve
        if v <= best - float(min_improve):
            best = v
            best_trial = t.number
            stale = 0
        else:
            stale += 1

    return {
        "best_value": float(best),
        "best_trial": best_trial,
        "stale": int(stale),
        "min_improve": float(min_improve),
    }


def get_plateau_state(study: optuna.Study, min_improve: float):
    """
    Load plateau state from study.user_attrs[PLATEAU_KEY].
    If missing or min_improve changed, recompute from history and persist.
    """
    ua = study.user_attrs or {}

    if PLATEAU_KEY in ua:
        try:
            state = json.loads(ua[PLATEAU_KEY])
            # if state matches current min_improve, reuse it
            if (
                {"best_value", "best_trial", "stale", "min_improve"} 
                <= set(state)
                and abs(state["min_improve"] - float(min_improve)) < 1e-12
            ):
                return state
        except Exception:
            pass

    # recompute from trial history with current min_improve
    state = _plateau_scan(study.get_trials(deepcopy=False), min_improve)
    study.set_user_attr(PLATEAU_KEY, json.dumps(state))
    return state


def set_plateau_state(study: optuna.Study, state: dict):
    # ensure min_improve is always stored
    if "min_improve" not in state:
        state["min_improve"] = float(MIN_IMPROVE)
    study.set_user_attr(PLATEAU_KEY, json.dumps(state))



# ============================================================
# Data loading
# ============================================================
def load_data():
    df = pd.read_parquet(DATA_PATH)

    # Use only departments with population
    df_model = (
        df[df["totpobla_2022"].notna()]
        .select_dtypes("number")
        .copy()
    )

    target_incid_col = f"dengue_incid_{TARGET_YEAR}"
    if target_incid_col not in df_model.columns:
        raise ValueError(f"Target column {target_incid_col} not found in df_model.")

    # Define target: log(1 + annual incidence)
    y_abs_incidence = df_model[target_incid_col]
    df_model["y_log_incidence"] = np.log1p(y_abs_incidence)

    # Drop yearly incidence + reg series (2018..TARGET_YEAR)
    drop_cols = []
    for year in range(2018, TARGET_YEAR + 1):
        drop_cols.extend([f"dengue_reg_{year}", f"dengue_incid_{year}"])

    drop_cols = [c for c in drop_cols if c in df_model.columns]
    if drop_cols:
        df_model = df_model.drop(columns=drop_cols)

    # Explicit leakage / target cols to exclude as features
    cols_to_exclude = [
        "y_log_incidence",
        target_incid_col,
        "dengue_reg_2023",  # may or may not exist, safe filter below
    ]
    cols_to_exclude = [c for c in cols_to_exclude if c in df_model.columns]

    feature_cols = [c for c in df_model.columns if c not in cols_to_exclude]

    X = df_model[feature_cols].copy()
    y = df_model["y_log_incidence"].copy()

    print(f"X shape: {X.shape}, y shape: {y.shape}")
    return X, y


def load_orthogonal_order(X: pd.DataFrame):
    ortho_df = pd.read_csv(ORTHO_FEATURES_CSV)
    col_name = "features"
    if col_name not in ortho_df.columns:
        raise ValueError(
            f"Expected column '{col_name}' in {ORTHO_FEATURES_CSV}, "
            f"found {ortho_df.columns.tolist()}"
        )

    orthogonal_order = ortho_df[col_name].tolist()
    orthogonal_order = [f for f in orthogonal_order if f in X.columns]

    print(f"Original number of features in X: {X.shape[1]}")
    print(f"Orthogonal order length (intersection): {len(orthogonal_order)}")

    if len(orthogonal_order) == 0:
        raise ValueError("orthogonal_order ended up empty after intersecting with X.columns")

    return orthogonal_order


# ============================================================
# Optuna Objective
# ============================================================
def make_objective(X, y, orthogonal_order):
    kf = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    def objective(trial: optuna.Trial) -> float:
        # ---------- 1) choose top-K features ----------
        max_k = min(len(orthogonal_order), 90)  # small cap; adjust if needed
        top_k = trial.suggest_int("top_k", 20, max_k)
        selected_features = orthogonal_order[:top_k]
        X_sub = X[selected_features].copy()

        # ---------- 2) CatBoost hyperparameters ----------
        params = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_seed": RANDOM_SEED,
            "verbose": False,
            "allow_writing_files": False,
            "task_type": "CPU",

            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
            "grow_policy": trial.suggest_categorical(
                "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
            ),
            "iterations": trial.suggest_int("iterations", 400, 1500),
            "od_type": "Iter",
            "od_wait": 50,
        }

        # ---------- 3) duplicate guard ----------
        sig = param_signature(trial.params)
        trial.set_user_attr("param_sig", sig)

        completed = trial.study.get_trials(
            states=(TrialState.COMPLETE,),
            deepcopy=False,
        )
        seen_sigs = set()
        for t in completed:
            ps = t.user_attrs.get("param_sig")
            if ps is None:
                ps = param_signature(t.params)  # for older trials
            seen_sigs.add(ps)

        if sig in seen_sigs:
            # exact repetition of an existing COMPLETE trial
            raise optuna.TrialPruned("duplicate-params")

        # ---------- 4) CV loop ----------
        rmse_scores = []
        for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X_sub)):
            X_train, X_valid = X_sub.iloc[train_idx], X_sub.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            model = CatBoostRegressor(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=(X_valid, y_valid),
                use_best_model=True,
            )

            preds = model.predict(X_valid)
            rmse = root_mean_squared_error(y_valid, preds)
            rmse_scores.append(rmse)

            # Report intermediate value per fold so pruner can act
            trial.report(float(rmse), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        rmse_mean = float(np.mean(rmse_scores))
        rmse_std = float(np.std(rmse_scores))

        trial.set_user_attr("rmse_mean", rmse_mean)
        trial.set_user_attr("rmse_std", rmse_std)
        trial.set_user_attr("rmse_folds", rmse_scores)
        trial.set_user_attr("selected_features", selected_features)

        return rmse_mean  # Optuna minimizes this

    return objective


# ============================================================
# Main
# ============================================================
def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data()
    orthogonal_order = load_orthogonal_order(X)

    objective = make_objective(X, y, orthogonal_order)

    # Sampler + pruner
    sampler = TPESampler(
        seed=RANDOM_SEED,
        multivariate=True,
        consider_prior=True,
        n_startup_trials=10,
    )
    pruner = MedianPruner(n_warmup_steps=2)

    # Study with storage → can resume
    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        storage=STUDY_STORAGE,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    
    state0 = get_plateau_state(study, MIN_IMPROVE)
    print(
        f"[plateau] resume status → stale={state0['stale']} "
        f"best={state0['best_value']:.6f} (trial {state0['best_trial']}) "
        f"min_improve={state0['min_improve']}"
    )

    def plateau_callback(study_obj: optuna.Study, trial: optuna.trial.FrozenTrial):
        # load current plateau state
        st = get_plateau_state(study_obj, MIN_IMPROVE)
        best = st["best_value"]
        stale = int(st["stale"])

        val = trial.value
        if val is None or not math.isfinite(val):
            return

        improved = (val <= best - float(MIN_IMPROVE))
        if improved:
            best = float(val)
            st["best_value"] = best
            st["best_trial"] = trial.number
            stale = 0
        else:
            stale += 1

        st["stale"] = stale
        set_plateau_state(study_obj, st)

        print(
            f"[plateau] trial={trial.number} value={val:.6f} "
            f"best={best:.6f} stale={stale}/{PATIENCE_TRIALS}"
        )

        if PATIENCE_TRIALS > 0 and stale >= PATIENCE_TRIALS:
            print("[plateau] patience reached → stopping study.")
            study_obj.stop()


    start_time = time.time()

    def progress_callback(study_obj: optuna.Study, trial: optuna.trial.FrozenTrial):
        dur = trial.duration.total_seconds() if trial.duration else 0.0
        val = trial.value
        val_str = f"{val:.6f}" if isinstance(val, (int, float)) and math.isfinite(val) else "NA"
        elapsed = (time.time() - start_time) / 60.0
        print(
            f"[trial {trial.number}] state={trial.state.name} "
            f"RMSE={val_str} dur={dur:.1f}s elapsed={elapsed:.1f}min"
        )

    callbacks = [plateau_callback, progress_callback]

    print(
        f"Running Optuna study '{STUDY_NAME}' with storage '{STUDY_STORAGE}' "
        f"(timeout={TIMEOUT}, patience={PATIENCE_TRIALS}, min_improve={MIN_IMPROVE})"
    )

    # Run until plateau or timeout; n_trials=None means "unbounded"
    study.optimize(
        objective,
        n_trials=None,
        timeout=TIMEOUT,
        gc_after_trial=True,
        callbacks=callbacks,
        show_progress_bar=False,
    )

    if study.best_trial is None or study.best_value is None:
        raise RuntimeError("Study ended without any COMPLETED trial.")

    best_trial = study.best_trial
    best_info = {
        "best_value_rmse": best_trial.value,
        "best_params": best_trial.params,
        "rmse_std": best_trial.user_attrs.get("rmse_std"),
        "rmse_folds": best_trial.user_attrs.get("rmse_folds"),
        "top_k": best_trial.params.get("top_k"),
        "selected_features": best_trial.user_attrs.get("selected_features"),
        "plateau_state": get_plateau_state(study, MIN_IMPROVE),
    }

    best_json_path = ARTIFACTS_DIR / "optuna_best.json"
    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(best_info, f, indent=2)

    print(f"\nBest RMSE: {best_info['best_value_rmse']:.4f}")
    print("Best params:")
    for k, v in best_info["best_params"].items():
        print(f"  {k}: {v}")
    print(f"Saved best trial info to {best_json_path}")

    trials_df = study.trials_dataframe()
    trials_csv_path = ARTIFACTS_DIR / "optuna_trials.csv"
    trials_df.to_csv(trials_csv_path, index=False)
    print(f"Saved all trials to {trials_csv_path}")


if __name__ == "__main__":
    main()
