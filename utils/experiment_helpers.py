import os 
import pandas as pd
import matplotlib.pyplot as plt


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
