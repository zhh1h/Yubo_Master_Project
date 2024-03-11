import json

def count_key_value_pairs(obj):
    if isinstance(obj, dict):
        return len(obj) + sum(count_key_value_pairs(v) for v in obj.values())
    elif isinstance(obj, list):
        return sum(count_key_value_pairs(item) for item in obj)
    else:
        # 不是字典或列表，没有键值对要计算
        return 0

# 加载JSON数据
with open('/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/epsilonExpandWeights/weights20_0.9.json', 'r') as file:
    data = json.load(file)

#  Calculate the numbers of keys-values pairs
count = count_key_value_pairs(data)
print(count)

