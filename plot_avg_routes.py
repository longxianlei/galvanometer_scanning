import numpy as np
import matplotlib.pyplot as plt

avg_route_axis_scan = np.load('avg_routes_axis.npy')
avg_route_2opt_scan = np.load('avg_routes_2opt.npy')
avg_route_google_scan = np.load('avg_routes_google.npy')
avg_route_defined_scan = np.load('avg_routes_assigned.npy')

print(avg_route_axis_scan)
print(avg_route_2opt_scan)
print(avg_route_google_scan)
print(avg_route_defined_scan)

max_3 = np.max([avg_route_google_scan, avg_route_2opt_scan, avg_route_axis_scan, avg_route_defined_scan])
min_3 = np.min([avg_route_google_scan, avg_route_2opt_scan, avg_route_axis_scan, avg_route_defined_scan])
y_max = np.ceil(max_3)
y_min = np.floor(min_3)

print(max_3)
print(min_3)
max_1 = np.max(avg_route_axis_scan)
min_1 = np.min(avg_route_axis_scan)
print(max_1)
print(min_1)
print('-----------')
max_1 = np.max(avg_route_2opt_scan)
min_1 = np.min(avg_route_2opt_scan)
print(max_1)
print(min_1)
print('-----------')
max_1 = np.max(avg_route_google_scan)
min_1 = np.min(avg_route_google_scan)
print(max_1)
print(min_1)
print('-----------')
max_1 = np.max(avg_route_defined_scan)
min_1 = np.min(avg_route_defined_scan)
print(max_1)
print(min_1)
print('-----------')

len_data = len(avg_route_axis_scan)
x = np.arange(1, len_data+1, 1)*10

# plt.figure('average_scan_routes', figsize=(10, 8), dpi=80)

plt.figure(dpi=150, figsize=(8, 6))
# 改变文字大小参数-fontsize
# 设置坐标轴的取值范围;
plt.xlim((0, len_data*10))
plt.ylim((y_min, y_max))
# 设置坐标轴的label;
plt.xlabel('X voltage', fontsize=15)
plt.ylabel('Y voltage', fontsize=15)
plt.title('The average routes for 4 methods', fontsize=15)
# 设置x坐标轴刻度;
plt.xticks(np.linspace(0, len_data*10, 11), fontsize=15)
plt.yticks(np.linspace(y_min, y_max, 15), fontsize=15)

plt.plot(x, avg_route_axis_scan, '*--', label='X-axis Scan')
plt.plot(x, avg_route_2opt_scan, 'g^-.', label='2opt Scan')
plt.plot(x, avg_route_google_scan, 'yx-', label='Google Scan')
plt.plot(x, avg_route_defined_scan, 'ro--', label='Own Scan')
plt.legend(loc='best', fontsize=15)
plt.savefig('./experiment_results/4_methods_avg_routes_orig_test.jpg')
plt.show()


