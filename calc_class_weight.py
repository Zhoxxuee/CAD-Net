import numpy as np
import torch
import math
from TrafficFlowClassification.utils.setConfig import setup_config

def build_cost_matrix(costs):
    num_classes = len(costs)
    cost_matrix = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                cost_matrix[i, j] = costs[i] / (costs[i] + costs[j])

    return cost_matrix

def calculate_weights_1(labels_count_x):
    total_samples = np.sum(labels_count_x)
    weights = []
    max_1 = max(labels_count_x)
    for count in labels_count_x:
        proportion = math.sqrt(0.15 * max_1 / count)
        weight = proportion
        weights.append(weight)

    return weights


def cal_fina_weight(labels_count):
    cost_ma = build_cost_matrix(labels_count)
    weights_1 = calculate_weights_1(labels_count)
    average_costs = np.mean(cost_ma, axis=1)
    fina_weight = np.multiply(weights_1, average_costs)
    return fina_weight
