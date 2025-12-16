import os
import json
import math
import time
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import root_mean_squared_error

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.trial import TrialState

# ============================================================
# Config
# ============================================================
TARGET_YEAR = int(os.getenv("TARGET_YEAR", 2023))
N_SPLITS = int(os.getenv("N_SPLITS", 5))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

REG_STRAT_BINS = int(os.getenv("REG_STRAT_BINS", "10"))

CV_SEEDS_ENV = os.getenv("CV_SEEDS", "").strip()
N_CV_SEEDS = int(os.getenv("N_CV_SEEDS", "3"))
CV_SEED_STEP = int(os.getenv("CV_SEED_STEP", "1000"))

if CV_SEEDS_ENV:
    CV_SEEDS = [int(s.strip()) for s in CV_SEEDS_ENV.split(",") if s.strip()]
else:
    CV_SEEDS = [RANDOM_SEED + i * CV_SEED_STEP for i in range(N_CV_SEEDS)]

PATIENCE_TRIALS = int(os.getenv("PATIENCE_TRIALS", "20"))
MIN_IMPROVE = float(os.getenv("MIN_IMPROVE", "1e-3"))

TIMEOUT = os.getenv("TIMEOUT", "19800")
TIMEOUT = int(TIMEOUT) if TIMEOUT else None

STUDY_STORAGE = os.getenv("STUDY_STORAGE", "sqlite:///optuna_dengue.db")
STUDY_NAME = os.getenv("STUDY_NAME", "dengue_catboost_topk_corr0p7")

DATA_PATH = Path("data/final_dataset.parquet")
ORTHO_FEATURES_CSV = Path("data/orthogonal_ordered_features.csv")
ARTIFACTS_DIR = Path("artifacts")

PLATEAU_KEY = "plateau_state_rmse_v1"


# ============================================================
# Utilities
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


def make_regression_strat_bins(y: pd.Series, n_bins: int, n_splits: int):
    """
    Create quantile bins for regression stratification.
    Returns integer bin labels or None if we can't create valid bins.
    """
    y = pd.Series(y).reset_index(drop=True)

    for q in range(int(n_bins), 1, -1):
        try:
            b = pd.qcut(y, q=q, duplicates="drop")
            # Must have enough samples per bin to stratify into n_splits
            vc = b.value_counts()
            if (vc < n_splits).any():
                continue
            return b.cat.codes.to_numpy()
        except ValueError:
            continue

    return None


def get_cv_splitter(y: pd.Series, seed: int):
    """
    Prefer StratifiedKFold on quantile bins. Fall back to KFold if binning fails.
    Returns (splitter, strat_labels_or_none).
    """
    strat = make_regression_strat_bins(y, REG_STRAT_BINS, N_SPLITS)
    if strat is None:
        return KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed), None
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed), strat


def compute_past_features(df_dengue_annual, target_year):
    df_dengue_annual = df_dengue_annual.copy()
    
    year_cols = [int(c.split('_')[-1]) for c in df_dengue_annual.columns if c.startswith('dengue_reg_')]
    year_cols = sorted(year_cols)
    
    past_years = [y for y in year_cols if y < target_year]

    reg_past_cols = [f'dengue_reg_{y}' for y in past_years]
    inc_past_cols = [f'dengue_incid_{y}' for y in past_years]

    def safe_get(colname, df, default=0.0):
        return df[colname] if colname in df.columns else default

    df_dengue_annual['dengue_reg_prev1'] = safe_get(f'dengue_reg_{target_year-1}', df_dengue_annual, 0.0)
    df_dengue_annual['dengue_reg_prev2'] = safe_get(f'dengue_reg_{target_year-2}', df_dengue_annual, 0.0)
    df_dengue_annual['dengue_reg_prev3'] = safe_get(f'dengue_reg_{target_year-3}', df_dengue_annual, 0.0)

    df_dengue_annual['dengue_incid_prev1'] = safe_get(f'dengue_incid_{target_year-1}', df_dengue_annual, 0.0)
    df_dengue_annual['dengue_incid_prev2'] = safe_get(f'dengue_incid_{target_year-2}', df_dengue_annual, 0.0)
    df_dengue_annual['dengue_incid_prev3'] = safe_get(f'dengue_incid_{target_year-3}', df_dengue_annual, 0.0)

    reg_past_vals = df_dengue_annual[reg_past_cols].to_numpy() if reg_past_cols else None
    inc_past_vals = df_dengue_annual[inc_past_cols].to_numpy() if inc_past_cols else None

    if reg_past_vals is not None:
        df_dengue_annual['dengue_reg_hist_mean'] = reg_past_vals.mean(axis=1)
        df_dengue_annual['dengue_reg_hist_max']  = reg_past_vals.max(axis=1)
        df_dengue_annual['dengue_reg_hist_std']  = reg_past_vals.std(axis=1)

    if inc_past_vals is not None:
        df_dengue_annual['dengue_incid_past_mean'] = inc_past_vals.mean(axis=1)
        df_dengue_annual['dengue_incid_past_max']  = inc_past_vals.max(axis=1)
        df_dengue_annual['dengue_incid_past_std']  = inc_past_vals.std(axis=1)

    if reg_past_vals is not None:
        df_dengue_annual['dengue_ever'] = (reg_past_vals.sum(axis=1) > 0).astype(int)

        df_dengue_annual['dengue_n_years_with'] = (reg_past_vals > 0).sum(axis=1)

        positive_inc_vals = inc_past_vals[inc_past_vals > 0]
        if positive_inc_vals.size > 0:
            epidemic_threshold = np.percentile(positive_inc_vals, 75)
        else:
            epidemic_threshold = 0.0

        epidemic_mask = (inc_past_vals >= epidemic_threshold).astype(int)
        df_dengue_annual['dengue_n_epidemic_years'] = epidemic_mask.sum(axis=1)

        last_epi_years = []
        for row in inc_past_vals:
            epi_years_for_row = [y for y, val in zip(past_years, row) if val >= epidemic_threshold and val > 0]
            if len(epi_years_for_row) == 0:
                last_epi_years.append(np.nan)
            else:
                last_epi_years.append(max(epi_years_for_row))

        df_dengue_annual['dengue_years_since_last_epidemic'] = target_year - pd.Series(last_epi_years, index=df_dengue_annual.index)

        eps = 1e-6
        df_dengue_annual['dengue_recent_vs_hist_ratio'] = (
            df_dengue_annual['dengue_incid_prev1'] / (df_dengue_annual['dengue_incid_past_mean'] + eps)
        )

        df_dengue_annual['dengue_volatility_incid'] = (
            df_dengue_annual['dengue_incid_past_std'] / (df_dengue_annual['dengue_incid_past_mean'] + eps)
        )
    else:
        for col in ['dengue_ever_dengue', 'dengue_n_years_with', 'dengue_n_epidemic_years',
                    'dengue_years_since_last_epidemic', 'dengue_recent_vs_hist_ratio', 'dengue_volatility_incid']:
            df_dengue_annual[col] = np.nan
            
    for year in year_cols :
        df_dengue_annual.drop(columns=[f'dengue_reg_{year}',f'dengue_incid_{year}'], inplace=True)
            
    return df_dengue_annual


# ============================================================
# Plateau helpers
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

    y_abs_incidence = df_model[target_incid_col]
    df_model["y_log_incidence"] = np.log1p(y_abs_incidence)

    df_model = compute_past_features(df_model, TARGET_YEAR)

    cols_to_exclude = [
        "y_log_incidence",
        target_incid_col,
        "dengue_reg_2023",
    ]

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
    def objective(trial: optuna.Trial) -> float:
        
        max_k = min(len(orthogonal_order), 90)
        top_k = trial.suggest_int("top_k", 20, max_k)
        selected_features = orthogonal_order[:top_k]
        X_sub = X[selected_features].copy()

        params = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
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
                ps = param_signature(t.params)
            seen_sigs.add(ps)

        if sig in seen_sigs:
            raise optuna.TrialPruned("duplicate-params")

        rmse_seed_means = []
        rmse_all_folds = []
        per_seed_details = []

        global_step = 0
        used_stratified_any = False

        for seed_i, cv_seed in enumerate(CV_SEEDS):
            splitter, strat_labels = get_cv_splitter(y, seed=cv_seed)
            used_stratified = strat_labels is not None
            used_stratified_any = used_stratified_any or used_stratified

            fold_rmses = []

            if strat_labels is None:
                split_iter = splitter.split(X_sub)
            else:
                split_iter = splitter.split(X_sub, strat_labels)

            for fold_idx, (train_idx, valid_idx) in enumerate(split_iter):
                X_train, X_valid = X_sub.iloc[train_idx], X_sub.iloc[valid_idx]
                y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

                params_fold = dict(params)
                params_fold["random_seed"] = int(cv_seed + fold_idx)

                model = CatBoostRegressor(**params_fold)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=(X_valid, y_valid),
                    use_best_model=True,
                )

                preds = model.predict(X_valid)
                rmse = root_mean_squared_error(y_valid, preds)

                fold_rmses.append(float(rmse))
                rmse_all_folds.append(float(rmse))

                # Report running mean (more stable than reporting raw fold RMSE)
                global_step += 1
                trial.report(float(np.mean(rmse_all_folds)), step=global_step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            seed_mean = float(np.mean(fold_rmses))
            rmse_seed_means.append(seed_mean)
            per_seed_details.append(
                {
                    "cv_seed": int(cv_seed),
                    "used_stratified": bool(used_stratified),
                    "fold_rmses": fold_rmses,
                    "seed_mean": seed_mean,
                }
            )

        rmse_mean = float(np.mean(rmse_seed_means))
        rmse_std_across_seeds = float(np.std(rmse_seed_means))

        trial.set_user_attr("rmse_mean", rmse_mean)
        trial.set_user_attr("rmse_std_across_seeds", rmse_std_across_seeds)
        trial.set_user_attr("rmse_seed_means", rmse_seed_means)
        trial.set_user_attr("cv_seeds", [int(s) for s in CV_SEEDS])
        trial.set_user_attr("used_stratified_any", bool(used_stratified_any))
        trial.set_user_attr("per_seed_details", per_seed_details)
        trial.set_user_attr("selected_features", selected_features)

        return rmse_mean

    return objective



# ============================================================
# Main
# ============================================================
def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data()
    orthogonal_order = load_orthogonal_order(X)

    objective = make_objective(X, y, orthogonal_order)

    sampler = TPESampler(
        seed=RANDOM_SEED,
        multivariate=True,
        consider_prior=True,
        n_startup_trials=10,
    )
    pruner = MedianPruner(n_warmup_steps=2)

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

    from optuna.trial import TrialState

    def plateau_callback(study_obj: optuna.Study, trial: optuna.trial.FrozenTrial):
        if trial.state != TrialState.COMPLETE:
            return

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

        if PATIENCE_TRIALS:
            print(
                f"[plateau] trial={trial.number} value={val:.6f} "
                f"best={best:.6f} stale={stale}/{PATIENCE_TRIALS}"
            )
            if stale >= PATIENCE_TRIALS:
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
