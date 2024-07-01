import json
import math
import os
import numpy as np
def calculate_weights(forgotten_path, never_forgotten_path, output_json_path):
    # Step 1: Read forgotten counts
    with open(forgotten_path, 'r') as f:
        next(f)  # Skip header
        forgotten_samples = {line.split('\t')[0]: int(line.split('\t')[1]) for line in f}

    # Step 2: Identify never forgotten samples
    with open(never_forgotten_path, 'r') as f:
        next(f)  # Skip header
        never_forgotten_samples = {line.strip(): 0 for line in f}  # Assign 0 forget count

    # Combine both dictionaries
    all_samples = {**forgotten_samples, **never_forgotten_samples}

    # Step 3: Calculate weights
    total_forgotten_counts = sum(forgotten_samples.values())
    num_never_forgotten = len(never_forgotten_samples)
    weights = {}

    for sample, count in all_samples.items():
        if count == 0:  # Never forgotten
            weights[sample] = 0.4 / num_never_forgotten
        else:  # Forgotten at least once
            #weights[sample] = 0.6 * (count / total_forgotten_counts)
            weights[sample] = 0.6*(1 - math.exp(-count**2))
    # Step 4: Save to JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(weights, json_file, indent=4)

def calculate_weights_log(forgotten_path, never_forgotten_path, output_json_path):
    # Step 1: Read forgotten counts
    with open(forgotten_path, 'r') as f:
        next(f)  # Skip header
        forgotten_samples = {line.split('\t')[0]: int(line.split('\t')[1]) for line in f}

    # Step 2: Identify never forgotten samples
    with open(never_forgotten_path, 'r') as f:
        next(f)  # Skip header
        never_forgotten_samples = {line.strip(): 0 for line in f}  # Assign 0 forget count

    # Combine both dictionaries
    all_samples = {**forgotten_samples, **never_forgotten_samples}

    # Step 3: Calculate weights
    num_never_forgotten = len(never_forgotten_samples)
    weights = {}
    total_log_counts = 0

    if num_never_forgotten > 0:
        weight_for_never_forgotten = 0.4 / num_never_forgotten
    else:
        weight_for_never_forgotten = 0  # or some small epsilon value if you want to give a minimal weight

    for sample, count in all_samples.items():
        if count == 0:
            weights[sample] = weight_for_never_forgotten
        else:
            log_count = math.log(1 + count)  # 使用log(1 + count)避免count为0时的问题
            weights[sample] = log_count
            total_log_counts += log_count

    # Avoid division by zero if total_log_counts is zero
    if total_log_counts == 0:
        total_log_counts = 1  # Avoid division by zero

    for sample, count in all_samples.items():
        if count > 0:  # Adjust weights for forgotten samples
            weights[sample] = 0.6 * (weights[sample] / total_log_counts)

    # Step 4: Save to JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(weights, json_file, indent=4)

    print(f"Weights saved to {output_json_path}")

def calculate_weights_directly(forgotten_path, never_forgotten_path, output_json_path):
    # Step 1: Read forgotten counts
    with open(forgotten_path, 'r') as f:
        next(f)  # Skip header
        forgotten_samples = {line.split('\t')[0]: int(line.split('\t')[1]) for line in f}

    # Step 2: Identify never forgotten samples
    with open(never_forgotten_path, 'r') as f:
        next(f)  # Skip header
        never_forgotten_samples = {line.strip(): 1 for line in f}  # Assign 1 as weight for never forgotten samples for simplicity

    # Combine both dictionaries
    all_samples = {**forgotten_samples, **never_forgotten_samples}

    # Step 3: Directly use forget counts as weights for forgotten samples
    # For never forgotten samples, a weight of 1 is used
    weights = {sample: (count if count > 0 else 1) for sample, count in all_samples.items()}

    # Step 4: Save to JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(weights, json_file, indent=4)


import math


def calculate_smooth_weights(forgotten_path, never_forgotten_path, output_json_path):
    # Step 1: Read forgotten counts
    with open(forgotten_path, 'r') as f:
        next(f)  # Skip header
        forgotten_samples = {line.split('\t')[0]: int(line.split('\t')[1]) for line in f}

    # Step 2: Identify never forgotten samples
    with open(never_forgotten_path, 'r') as f:
        next(f)  # Skip header
        never_forgotten_samples = {line.strip(): 0 for line in f}

    # Combine both dictionaries
    all_samples = {**forgotten_samples, **never_forgotten_samples}

    # Calculate maximum forget count for normalization
    max_count = max(all_samples.values())

    # Step 3: Calculate weights using a smoothing function
    total_weight = 0.0
    weights = {}

    for sample, count in all_samples.items():
        if count == 0:  # Never forgotten
            weights[sample] = 1  # Assign a base weight for never forgotten samples
        else:
            # Apply square root smoothing
            smoothed_weight = math.sqrt(count) / math.sqrt(max_count)
            weights[sample] = smoothed_weight
            total_weight += smoothed_weight

    # Normalize weights so they sum up to 1 (or another desired total weight)
    for sample in weights.keys():
        weights[sample] /= total_weight

    # Step 4: Save to JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(weights, json_file, indent=4)

    print("Weights saved to", output_json_path)


def calculate_weights_exp(forgotten_path, never_forgotten_path, output_json_path):
    # 读取遗忘次数
    with open(forgotten_path, 'r') as f:
        next(f)  # 跳过标题行
        forgotten_samples = {line.split('\t')[0]: int(line.split('\t')[1]) for line in f}

    # 识别从未遗忘的样本
    with open(never_forgotten_path, 'r') as f:
        next(f)  # 跳过标题行
        never_forgotten_samples = {line.strip(): 0 for line in f}

    # 合并两个字典
    all_samples = {**forgotten_samples, **never_forgotten_samples}

    # 计算权重
    max_count = max(all_samples.values()) if all_samples else 1  # 防止除以零
    min_count = min(all_samples.values()) if all_samples else 0
    range_count = max_count - min_count if max_count > min_count else 1

    # 应用指数加权
    weights = {sample: np.exp((count - min_count) / range_count) for sample, count in all_samples.items()}

    # 归一化权重
    total_weight = sum(weights.values())
    normalized_weights = {sample: weight / total_weight for sample, weight in weights.items()}

    # 保存到 JSON 文件
    with open(output_json_path, 'w') as json_file:
        json.dump(normalized_weights, json_file, indent=4)

    print(f"Weights saved to {output_json_path}")

def calculate_weights_gaussian(forgotten_path, never_forgotten_path, output_json_path):
    # 读取遗忘次数
    with open(forgotten_path, 'r') as f:
        next(f)  # 跳过标题行
        forgotten_samples = {line.split('\t')[0]: int(line.split('\t')[1]) for line in f}

    # 识别从未遗忘的样本
    with open(never_forgotten_path, 'r') as f:
        next(f)  # 跳过标题行
        never_forgotten_samples = {line.strip(): 0 for line in f}

    # 合并两个字典
    all_samples = {**forgotten_samples, **never_forgotten_samples}

    # 计算权重
    weights = {sample: 1 - math.exp(-count**2) for sample, count in all_samples.items()}
    #weights = {sample: 1 - math.exp(-count) for sample, count in all_samples.items()}


    # 归一化权重

    # 保存到 JSON 文件
    with open(output_json_path, 'w') as json_file:
        json.dump(weights, json_file, indent=4)

    print(f"Weights saved to {output_json_path}")


def calculate_weights_normalizationExp(forgotten_path, never_forgotten_path, output_json_path):
    # Step 1: Read forgotten counts
    with open(forgotten_path, 'r') as f:
        next(f)  # Skip header
        forgotten_samples = {line.split('\t')[0]: int(line.split('\t')[1]) for line in f}

    # Step 2: Identify never forgotten samples
    with open(never_forgotten_path, 'r') as f:
        next(f)  # Skip header
        never_forgotten_samples = {line.strip(): 0 for line in f}

    # Combine both dictionaries
    all_samples = {**forgotten_samples, **never_forgotten_samples}

    # Step 3: Calculate weights
    num_never_forgotten = len(never_forgotten_samples)
    exp_weights = {}
    total_exp_weights = 0  # Total weights for normalization

    for sample, count in all_samples.items():
        if count > 0:
            exp_weights[sample] = 1 - math.exp(-count**2)
            total_exp_weights += exp_weights[sample]

    weights = {}
    for sample, count in all_samples.items():
        if count == 0:  # Never forgotten
            weights[sample] = 0.4 / num_never_forgotten
        else:  # Forgotten at least once
            weights[sample] = 0.6 * (exp_weights[sample] / total_exp_weights)  # Normalize weights for forgotten samples

    # Step 4: Save to JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(weights, json_file, indent=4)

    print(f"Weights saved to {output_json_path}")


# 计算权重并保存到JSON的逻辑应该放在这里，确保它是在train_model函数执行完之后进行的。
forgotten_path = '/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/models/adversary/cifar10-vgg19-DBplusFilterCaltech-50/forgotten_examples_DBplusFilterCaltech.txt'  # 替换为实际路径
never_forgotten_path = '/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/models/adversary/cifar10-vgg19-DBplusFilterCaltech-50/never_forgotten_examples_DBplusFilterCaltech.txt'  # 替换为实际路径
output_json_path = '/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/models/adversary/cifar10-vgg19-DBplusFilterCaltech-50/output_weights_DBplusFilterCaltech_Exp2.json'  # 指定输出JSON文件的路径
    # 调用函数计算权重并保存到JSON
#calculate_weights_log(forgotten_path, never_forgotten_path, output_json_path)
calculate_weights_gaussian(forgotten_path, never_forgotten_path, output_json_path)
