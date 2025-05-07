import os 
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import numpy as np


def compare_grouped_feature_importance(models, model_names, group_prefixes, importance_type='gain', output_dir="plots"):
    """
    Compare grouped feature importances across multiple models and save plots.

    Args:
        models: list of trained XGBClassifier or Booster models
        model_names: list of names for each model
        group_prefixes: dict mapping prefix -> group name
        importance_type: 'gain', 'weight', or 'cover'
        output_dir: directory to save plots
    Returns:
        all_grouped: dict of model_name -> grouped importance dict
        combined_plot_path: path to multi-model comparison plot
    """
    os.makedirs(output_dir, exist_ok=True)
    all_grouped = {}

    # Per-model plots
    for model, name in zip(models, model_names):
        print(name)
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        importances = booster.get_score(importance_type=importance_type)
        imp_df = pd.DataFrame(importances.items(), columns=['feature', 'importance'])

        def assign_group(feat):
            for prefix, group_name in group_prefixes.items():
                if feat.startswith(prefix):
                    return group_name
            return "Text"

        imp_df["group"] = imp_df["feature"].apply(assign_group)
        grouped = imp_df.groupby("group")["importance"].sum().sort_values(ascending=False)
        all_grouped[name] = grouped.to_dict()

        # Save individual plot
        plt.figure(figsize=(8, 5))
        grouped.plot(kind="bar")
        plt.title(f"Grouped Feature Importances ({importance_type}) - {name}")
        plt.ylabel("Total Importance")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        path = os.path.join(output_dir, f"grouped_importance_{name}.png")
        plt.savefig(path)
        plt.close()

    # Combined plot
    combined_df = pd.DataFrame(all_grouped).fillna(0)
    combined_df.plot(kind="bar", figsize=(10, 6))
    plt.title(f"Grouped Feature Importances Comparison ({importance_type})")
    plt.ylabel("Total Importance")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    combined_plot_path = os.path.join(output_dir, "grouped_importance_comparison.png")
    plt.savefig(combined_plot_path)
    plt.close()

    return all_grouped

def compare_within_group_feature_importance(models, model_names, group_prefix, importance_type='gain', top_n=15, output_dir="plot"):
    os.makedirs(output_dir, exist_ok=True)
    all_importances = {}

    for model, name in zip(models, model_names):
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        importances = booster.get_score(importance_type=importance_type)
        filtered = {k: v for k, v in importances.items() if k.startswith(group_prefix)}
        df = pd.DataFrame(filtered.items(), columns=["feature", name])
        all_importances[name] = df.set_index("feature")

    # Combine all importance DataFrames
    combined_df = pd.concat(all_importances.values(), axis=1).fillna(0)
    combined_df.columns = model_names

    # Sum across models and get top N features
    top_features = combined_df.sum(axis=1).nlargest(top_n).index
    top_df = combined_df.loc[top_features]

    # Plot
    ax = top_df.plot(kind="barh", figsize=(10, 6))
    plt.title(f"Top {top_n} '{group_prefix}' Features by {importance_type}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.gca().invert_yaxis()

    fig_path = os.path.join(output_dir, f"top_{group_prefix.strip('_')}_features_comparison_{importance_type}.png")
    plt.savefig(fig_path)
    plt.close()

    return top_df

def compute_total_effect_multiclass_per_treatment(
    model: XGBClassifier,
    df: pd.DataFrame,
    treatment_cols: list[str],
    plot_name: str,
    plot_path='plots',
    class_labels=["Left", "Center", "Right"],
    low_q=0.10,
    high_q=0.90,
):
    result = {}

    assert all(f in df.columns for f in model.feature_names_in_), "Mismatch in required feature columns"

    required_features = list(model.feature_names_in_)

    for t_col in treatment_cols:
        df_copy = df[df[t_col] != 0]

        confounders = list(df_copy.columns)
        confounders.remove(t_col)
        
        x_low = -1
        x_high = 1

        features = [t_col] + confounders
        
        df_low = df_copy[features].copy()        
        df_high = df_copy[features].copy()
        
        df_low[t_col] = x_low
        df_high[t_col] = x_high

        X_low = df_low[required_features]
        X_high = df_high[required_features]

        probs_low = model.predict_proba(X_low)
        probs_high = model.predict_proba(X_high)

        TE_diffs = probs_high - probs_low              # shape: (n_samples, n_classes)
        TE_means = TE_diffs.mean(axis=0)               # mean TE per class
        TE_q25 = np.percentile(TE_diffs, 25, axis=0)
        TE_q75 = np.percentile(TE_diffs, 75, axis=0)

        result[t_col] = {
            "mean": dict(zip(class_labels, TE_means)),
            "q25": dict(zip(class_labels, TE_q25)),
            "q75": dict(zip(class_labels, TE_q75))
        }

    plot_path = f"{plot_path}/{plot_name}.png"
    x = np.arange(len(class_labels))
    width = 0.8 / len(treatment_cols)

    plt.figure(figsize=(10, 6))
    for i, t_col in enumerate(treatment_cols):
        offsets = x + (i - len(treatment_cols)/2) * width + width/2
        means = [result[t_col]["mean"][cls] for cls in class_labels]
        errs_lower = []
        errs_upper = []

        for cls in class_labels:
            mean = result[t_col]["mean"][cls]
            q25 = result[t_col]["q25"][cls]
            q75 = result[t_col]["q75"][cls]

            err_low = mean - q25
            err_high = q75 - mean

            if err_low < 0:
                print(f"[Warning] Lower whisker negative for class '{cls}' and treatment '{t_col}'. Clipped to 0.")
                err_low = 0.0
            if err_high < 0:
                print(f"[Warning] Upper whisker negative for class '{cls}' and treatment '{t_col}'. Clipped to 0.")
                err_high = 0.0

            errs_lower.append(err_low)
            errs_upper.append(err_high)

        yerr = [errs_lower, errs_upper]

        plt.bar(offsets, means, width=width, yerr=yerr, capsize=4, label=t_col)

    plt.xticks(x, class_labels)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title(f"Total Effect of Each Treatment (-1 → +1)")
    plt.ylabel("Mean Δ Predicted Probability (with IQR whiskers)")
    plt.legend(title="Treatment", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(plot_path)
    plt.close()


    return result
