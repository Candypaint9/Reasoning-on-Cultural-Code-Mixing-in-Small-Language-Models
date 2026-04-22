import json
import os
import numpy as np
import matplotlib.pyplot as plt

THRESHOLDS = [0.0, 0.25, 0.5, 1.0, 2.0]

def main(QTYPE, MASTER_JSON_FILE):
    if not os.path.exists(MASTER_JSON_FILE):
        print(f"[SKIP] {MASTER_JSON_FILE} not found. Moving to next.")
        return

    print(f"\nProcessing {QTYPE}...")
    with open(MASTER_JSON_FILE, "r") as f:
        model_plot_data = json.load(f)

    names = list(model_plot_data.keys())
    if not names:
        print(f"Warning: {MASTER_JSON_FILE} is empty!")
        return

    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(names))]

    # 1. Trajectory Graph
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for idx, name in enumerate(names):
        data = model_plot_data[name]
        c = colors[idx]
        
        # Determine line thickness: Bold for FT models, normal for Base
        lw = 3.0 if "-FT" in name else 1.5
        alpha = 1.0 if "-FT" in name else 0.7

        std_means = [x if x is not None else np.nan for x in data['std_means']]
        cm_means = [x if x is not None else np.nan for x in data['cm_means']]

        ax1.plot(THRESHOLDS, std_means, marker='o', linestyle='-', color=c, 
                 linewidth=lw, alpha=alpha, label=f'{name} (Orig)')
        ax1.plot(THRESHOLDS, cm_means, marker='s', linestyle='--', color=c, 
                 linewidth=lw, alpha=alpha, label=f'{name} (CM)')

    ax1.set_xlabel('Confidence Threshold (\u0394 Logit)')
    ax1.set_ylabel('Mean Emergence Layer')
    ax1.set_title(f'Unified Emergence Trajectory: [{QTYPE}]')
    ax1.set_xticks(THRESHOLDS)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f"{QTYPE}_unified_trajectory.png", bbox_inches='tight')
    plt.close()

    # 2. Confidence Growth Graph
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    for idx, name in enumerate(names):
        data = model_plot_data[name]
        c = colors[idx]
        lw = 3.0 if "-FT" in name else 1.5

        if data['avg_std_deltas']:
            ax2.plot(range(1, len(data['avg_std_deltas'])+1), data['avg_std_deltas'], 
                     linestyle='-', color=c, linewidth=lw, label=f'{name} (Orig)')
        if data['avg_cm_deltas']:
            ax2.plot(range(1, len(data['avg_cm_deltas'])+1), data['avg_cm_deltas'], 
                     linestyle='--', color=c, linewidth=lw, label=f'{name} (CM)')

    ax2.axhline(y=0, color='gray', linestyle='-')
    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Layer Depth')
    ax2.set_ylabel('Average \u0394\u2113')
    ax2.set_title(f'Unified Confidence Growth: [{QTYPE}]')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f"{QTYPE}_unified_confidence_growth.png", bbox_inches='tight')
    plt.close()

    # 3. Accuracy Bars
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    width = 0.35

    std_accs = [model_plot_data[n]['std_acc'] for n in names]
    cm_accs = [model_plot_data[n]['cm_acc'] for n in names]

    ax3.bar(x - width/2, std_accs, width, label='Standard', color='#1f77b4')
    ax3.bar(x + width/2, cm_accs, width, label='Code-Mixed', color='#ff7f0e')

    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title(f'Final Layer Accuracy: [{QTYPE}]')
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=15, ha='right')
    ax3.legend()
    plt.savefig(f"{QTYPE}_unified_accuracy_bars.png", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    for i in range(1, 4):
        q_label = f"L{i}"
        filename = f"{q_label}_master_results.json"
        main(q_label, filename)