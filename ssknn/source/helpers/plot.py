import matplotlib.pyplot as plt
import numpy as np

def plot_histories(histories, fig_size=(10,7)):
    results = {}
    reported_metrics = ['hr', 'mrr', 'cov']
    for metric in reported_metrics:
        results[metric] = {}
        for model in histories:
            results[metric][model] = {}
            for params in histories[model]:
                if params[0] not in results[metric][model]:
                    results[metric][model][params[0]] = {}
                results[metric][model][params[0]][params[1]] = histories[model][params][metric]
    fig, axes = plt.subplots(nrows=len(reported_metrics), ncols=len(histories), sharex=True, sharey='row', figsize=fig_size, tight_layout=True)
    for i, metric in enumerate(results):
        for j, model in enumerate(results[metric]):
            for similarity in results[metric][model]:
                neighbors, metrics = zip(*results[metric][model][similarity].items())
                neighbors = np.array(neighbors)
                metrics = np.array(metrics)
                order = np.argsort(neighbors)
                axes[i][j].plot(neighbors[order], metrics[order], label=similarity)
                if i == 0:
                    axes[i][j].set_title(model)
                if j == 0:
                    axes[i][j].set_ylabel(metric)
                if i == len(reported_metrics) - 1:
                    axes[i][j].set_xlabel('N neighbors')
    handles, labels = axes[-1][-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))

    