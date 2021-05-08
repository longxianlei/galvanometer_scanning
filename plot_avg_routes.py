import numpy as np
import matplotlib.pyplot as plt

# avg_route_axis_scan = np.load('avg_routes_axis.npy')
# avg_route_2opt_scan = np.load('avg_routes_2opt.npy')
# avg_route_google_scan = np.load('avg_routes_google.npy')
# avg_route_defined_scan = np.load('avg_routes_assigned.npy')
#
# print(avg_route_axis_scan)
# print(avg_route_2opt_scan)
# print(avg_route_google_scan)
# print(avg_route_defined_scan)
#
# max_3 = np.max([avg_route_google_scan, avg_route_2opt_scan, avg_route_axis_scan, avg_route_defined_scan])
# min_3 = np.min([avg_route_google_scan, avg_route_2opt_scan, avg_route_axis_scan, avg_route_defined_scan])
# y_max = np.ceil(max_3)
# y_min = np.floor(min_3)
#
# print(max_3)
# print(min_3)
# max_1 = np.max(avg_route_axis_scan)
# min_1 = np.min(avg_route_axis_scan)
# print(max_1)
# print(min_1)
# print('-----------')
# max_1 = np.max(avg_route_2opt_scan)
# min_1 = np.min(avg_route_2opt_scan)
# print(max_1)
# print(min_1)
# print('-----------')
# max_1 = np.max(avg_route_google_scan)
# min_1 = np.min(avg_route_google_scan)
# print(max_1)
# print(min_1)
# print('-----------')
# max_1 = np.max(avg_route_defined_scan)
# min_1 = np.min(avg_route_defined_scan)
# print(max_1)
# print(min_1)
# print('-----------')
#
# len_data = len(avg_route_axis_scan)
# x = np.arange(1, len_data+1, 1)*10
#
# # plt.figure('average_scan_routes', figsize=(10, 8), dpi=80)
#
# plt.figure(dpi=150, figsize=(8, 6))
# # 改变文字大小参数-fontsize
# # 设置坐标轴的取值范围;
# plt.xlim((0, len_data*10))
# plt.ylim((y_min, y_max))
# # 设置坐标轴的label;
# plt.xlabel('Num of Samples', fontsize=15)
# plt.ylabel('Average length', fontsize=15)
# plt.title('The average routes for 4 methods', fontsize=15)
# # 设置x坐标轴刻度;
# plt.xticks(np.linspace(0, len_data*10, 11), fontsize=15)
# plt.yticks(np.linspace(y_min, y_max, 15), fontsize=15)
#
# plt.plot(x, avg_route_axis_scan, '*--', label='X-axis Scan')
# plt.plot(x, avg_route_2opt_scan, 'g^-.', label='2opt Scan')
# plt.plot(x, avg_route_google_scan, 'yx-', label='Google Scan')
# plt.plot(x, avg_route_defined_scan, 'ro--', label='Own Scan')
# plt.legend(loc='best', fontsize=15)
# plt.savefig('./experiment_results/4_methods_avg_routes_orig_test2.jpg')
# plt.show()


axis_scan = [6.1, 4.7, 4.57, 4.15, 4.04, 3.76, 3.64, 3.73, 3.635, 3.629]
two_opt_scan = [5.9, 4.45, 4.1, 3.85, 3.68, 3.66, 3.59, 3.55, 3.566, 3.57]
google_scan = [5.8, 4.25, 4.033, 3.775, 3.68, 3.65, 3.58, 3.52, 3.512, 3.50]
own_scan = [5.3, 4.33, 4.031, 3.74, 3.65, 3.68, 3.60, 3.51, 3.503, 3.49]

len_data = len(axis_scan)
x = np.arange(1, len_data+1, 1)*50

# plt.figure('average_scan_routes', figsize=(10, 8), dpi=80)

plt.figure(dpi=150, figsize=(8, 6))
# 改变文字大小参数-fontsize
# 设置坐标轴的取值范围;
plt.xlim((0, len_data*50))
# plt.ylim((3.00, 6.50))
# plt.ylim((150, 300))
# 设置坐标轴的label;
plt.xlabel('Num of Samples', fontsize=15)
plt.ylabel('Average scan FPS', fontsize=15)
plt.title('The average scanning speed', fontsize=15)
# 设置x坐标轴刻度;
plt.xticks(np.linspace(0, len_data*50, 11), fontsize=15)
# plt.yticks(np.linspace(150, 300, 16), fontsize=15)
# plt.yticks(np.linspace(3.00, 6.50, 15), fontsize=15)

axis_scan_fps = 1000.0/(np.asarray(axis_scan))

axis_total_scan_time = []
for i in range(len(axis_scan)):
    axis_total_scan_time.append(np.round((50*(i+1)*axis_scan[i]), 4))
print(axis_total_scan_time)

two_opt_total_scan_time = []
for i in range(len(axis_scan)):
    two_opt_total_scan_time.append(np.round((50*(i+1)*two_opt_scan[i]), 4))

google_total_scan_time = []
for i in range(len(axis_scan)):
    google_total_scan_time.append(np.round((50*(i+1)*google_scan[i]), 4))

own_total_scan_time = []
for i in range(len(axis_scan)):
    own_total_scan_time.append(np.round((50*(i+1)*own_scan[i]), 4))

plt.plot(x, axis_total_scan_time, '*--', label='X-axis Scan')
plt.plot(x, two_opt_total_scan_time, 'g^-.', label='2opt Scan')
plt.plot(x, google_total_scan_time, 'yx-', label='Google Scan')
plt.plot(x, own_total_scan_time, 'ro--', label='Own Scan')


# print(axis_scan_fps)

# plt.plot(x, 1000.0/np.asarray(axis_scan), '*--', label='X-axis Scan')
# plt.plot(x, 1000.0/np.asarray(two_opt_scan), 'g^-.', label='2opt Scan')
# plt.plot(x, 1000.0/np.asarray(google_scan), 'yx-', label='Google Scan')
# plt.plot(x, 1000.0/np.asarray(own_scan), 'ro--', label='Own Scan')
#
# plt.plot(x, axis_scan, '*--', label='X-axis Scan')
# plt.plot(x, two_opt_scan, 'g^-.', label='2opt Scan')
# plt.plot(x, google_scan, 'yx-', label='Google Scan')
# plt.plot(x, own_scan, 'ro--', label='Own Scan')
plt.legend(loc='best', fontsize=15)
# plt.savefig('./analysis_fig/scan_routes_speed_comparison/4_methods_avg_scanning_fps.jpg')
# plt.savefig('./experiment_results/temp2/4_methods_avg_scanning_fps.jpg')
plt.show()