
import matplotlib.pyplot as plt

# 数据
steps = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]

# 第一组数据（蓝色折线）
data1 = [(5, 14.04), (10, 14.24), (15, 14.92), (20, 13.68), (25, 14.4),
         (30, 15.02), (40, 13.81), (50, 13.64), (60, 14.14), (70, 13.62),
         (80, 13.79), (90, 14.07), (100, 13.59)]

# 第二组数据（红色折线）
data2 = [(5, 19.99), (10, 15.57), (15, 17.65), (20, 19.02), (25, 18.17),
         (30, 19.9), (40, 17.67), (50, 17.56), (60, 18.77), (70, 18.77),
         (80, 18.82), (90, 18.27), (100, 18.36)]

# 提取x和y坐标
x1, y1 = zip(*data1)
x2, y2 = zip(*data2)

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(steps, y1, marker='o', linestyle='-', color='blue', label='pure random linear interpolation')
plt.plot(steps, y2, marker='o', linestyle='-', color='red', label='weighted loss on linear interpolation')
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Steps')
plt.xticks(steps)
plt.legend()
plt.grid(True)
plt.tight_layout()

# 显示图形
plt.show()

