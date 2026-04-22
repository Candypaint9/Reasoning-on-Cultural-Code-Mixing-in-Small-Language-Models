import json
import os
import numpy as np
import matplotlib.pyplot as plt

DATASETS = ["L1", "L2", "L3"]
THRESHOLDS = [0.0, 0.25, 0.5, 1.0, 2.0]

PLOT_CONFIG = [
    {"ds": "L1", "type": "std", "label": "L1 (Standard)", "color": "#1f77b4", "ls": "-", "marker": "o"}, # Blue
    {"ds": "L2", "type": "std", "label": "L2 (Standard)", "color": "#2ca02c", "ls": "-", "marker": "s"}, # Green
    {"ds": "L2", "type": "cm",  "label": "L2 (Code-Mixed)", "color": "#2ca02c", "ls": "--", "marker": "^"}, 
    {"ds": "L3", "type": "std", "label": "L3 (Standard)", "color": "#d62728", "ls": "-", "marker": "D"}, # Red
    {"ds": "L3", "type": "cm",  "label": "L3 (Code-Mixed)", "color": "#d62728", "ls": "--", "marker": "X"}
]

def main():
    db = {}
    unique_models = set()
    
    for ds in DATASETS:
        filename = f"{ds}_master_results.json"
        if os.path.exists(filename):
            print(f"Loaded {filename}")
            with open(filename, "r") as f:
                db[ds] = json.load(f)
                unique_models.update(db[ds].keys())
        else:
            print(f"[WARNING] {filename} not found. Skipping {ds} data.")

    if not db:
        print("No master JSON files found")
        return

    if not unique_models:
        print("The JSON files are empty.")
        return

    print(f"\nFound {len(unique_models)} unique models to analyze: {', '.join(unique_models)}")

    for model_name in unique_models:
        print(f"Plotting graphs for: {model_name}")
        
        # ----------------------------------------------------
        # GRAPH 1: Unified Emergence Trajectory
        # ----------------------------------------------------
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        plotted_trajectory = False
        
        for config in PLOT_CONFIG:
            ds = config["ds"]
            dtype = config["type"]
            
            if ds in db and model_name in db[ds]:
                model_data = db[ds][model_name]
                
                raw_means = model_data.get(f"{dtype}_means", [])
                
                if raw_means and any(x is not None for x in raw_means):
                    clean_means = [x if x is not None else np.nan for x in raw_means]
                    
                    ax1.plot(THRESHOLDS, clean_means, marker=config["marker"], 
                             linestyle=config["ls"], color=config["color"], label=config["label"])
                    plotted_trajectory = True

        if plotted_trajectory:
            ax1.set_xlabel('Confidence Threshold (\u0394 Logit)')
            ax1.set_ylabel('Mean Emergence Layer')
            ax1.set_title(f'Emergence Trajectory Across Datasets\nModel: {model_name}')
            ax1.set_xticks(THRESHOLDS)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, linestyle=':', alpha=0.6)
            
            traj_filename = f"{model_name}_cross_dataset_trajectory.png"
            plt.savefig(traj_filename, bbox_inches='tight')
            print(f"  -> Saved {traj_filename}")
        plt.close(fig1)

        # ----------------------------------------------------
        # GRAPH 2: Confidence Growth Across Layers
        # ----------------------------------------------------
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        plotted_growth = False
        
        for config in PLOT_CONFIG:
            ds = config["ds"]
            dtype = config["type"]
            
            if ds in db and model_name in db[ds]:
                model_data = db[ds][model_name]
                
                deltas = model_data.get(f"avg_{dtype}_deltas", [])
                
                if deltas:
                    ax2.plot(range(1, len(deltas)+1), deltas, 
                             linestyle=config["ls"], color=config["color"], label=config["label"])
                    plotted_growth = True

        if plotted_growth:
            ax2.axhline(y=0, color='gray', linestyle='-')
            ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
            ax2.set_xlabel('Layer Depth')
            ax2.set_ylabel('Average \u0394\u2113')
            ax2.set_title(f'Confidence Growth Across Datasets\nModel: {model_name}')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            growth_filename = f"{model_name}_cross_dataset_confidence_growth.png"
            plt.savefig(growth_filename, bbox_inches='tight')
            print(f"  -> Saved {growth_filename}")
        plt.close(fig2)

    print("\nAll cross-dataset graphs successfully generated")

if __name__ == "__main__":
    main()