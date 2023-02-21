from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os
import pandas as pd
import numpy as np
import networkx as nx

def graph_roc_curve(location, prob_prediction, classification, evaluation):
    # calculate the fpr and tpr for all thresholds of the classification
    preds = list(prob_prediction[1])
    fpr, tpr, _ = metrics.roc_curve(classification, preds)
    roc_auc = metrics.auc(fpr, tpr)

    plt.clf()
    plt.title('Receiver Operating Characteristic')
    tpr_score = float(evaluation['tp'])/(evaluation['tp'] + evaluation['fn'])
    fpr_score = float(evaluation['fp'])/(evaluation['fp'] + evaluation['tn'])

    plt.plot(fpr_score, tpr_score, marker='o', color='black')
    plt.annotate('(%.2f,%.2f)' %(fpr_score, tpr_score), (fpr_score + 0.02, tpr_score - 0.01))
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0.0, 1.0], [0.0, 1.0],'r--')
    plt.xlim([0.0, 1.0])
    #plt.xticks(np.arange(0, 1, step = 0.1))
    plt.ylim([0.0, 1.0])
    #plt.yticks(np.arange(0, 1, step = 0.1))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    graph_file = 'roc_curve.pdf'
    path_graph_file = os.path.join(location, graph_file)
    plt.savefig(path_graph_file)

def graph_features_convergence(location, fold, rules, columns_list, args):
    plt.clf()
    #rules_list = list(rules[rules[args.class_column] == 1]['rule'])
    #rules_list = rules_list[:50]
    rules_intersection = set(rules[0])
    for i in range(1, len(rules)):
        rules_intersection = rules_intersection.intersection(set(rules[i]))
    rules_list = list(rules_intersection)
    node_s = []
    node_d = []
    for rule in rules_list:
        l = len(rule)
        for s in range(l):
            d = s + 1
            while d < l:
                node_s.append(rule[s])
                node_d.append(rule[d])
                d += 1
    edges = pd.DataFrame({'node_s': node_s, 'node_d': node_d})

    figs, axs = plt.subplots(2, 1, figsize = (10, 15))

    G = nx.from_pandas_edgelist(
        edges,
        "node_s",
        "node_d",
        create_using = nx.Graph()
    )
    nodes_list = list(G.nodes)
    degree_list = [G.degree[node] for node in nodes_list]

    labels_list = list()
    for n in nodes_list:
        labels_list.append(columns_list[n])
    dict_node_label = dict(list(zip(nodes_list, labels_list)))

    cm = plt.get_cmap('Blues')
    min_degree = min(degree_list) if degree_list else 0
    max_degree = max(degree_list) if degree_list else 0
    color_norm  = colors.Normalize(vmin = min_degree, vmax = max_degree)
    scalar_map = cmx.ScalarMappable(norm = color_norm, cmap = cm)

    handles_dict = {patches.Patch(color = scalar_map.to_rgba(G.degree[k]), label = f'{k}. {v}') for k, v in dict_node_label.items()}
    axs[0].axis('off')
    axs[0].legend(handles = handles_dict, scatterpoints = 1, loc = 'lower center', ncol = 2, frameon = False)

    size_list = [degree * 100 for degree in degree_list]
    nx.draw(G, ax = axs[1], with_labels = True, node_size = size_list, node_color = degree_list, font_weight = 'bold', cmap = plt.cm.Blues)
    graph_file = f'{fold}_features_convergence.pdf'
    path_graph_file = os.path.join(location, graph_file)
    plt.savefig(path_graph_file)
