import numpy as np
import matplotlib.pyplot as plt
from draft import cal_distance

if __name__ == '__main__':

    """
    1. This part plot the Z scan mode.
    """
    total_points = 100
    x = np.arange(-4.5, 5, 1)
    y = np.arange(-4.5, 5, 1)
    # x = np.arange(-4.5, 2.5, 0.5)
    # y = np.arange(-4.5, 2.5, 0.5)
    x_data = np.zeros(total_points)
    y_data = np.zeros(total_points)
    for i in range(total_points):
        if (i//10) % 2 == 0:
            x_data[i] = x[i % 10]
        else:
            x_data[i] = x[10 - 1 - i % 10]
        y_data[i] = y[i//10]
    # print(x_data.shape)
    # print(y_data.size)

    plt.figure(1)
    # 设置坐标轴的取值范围;
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    # 设置坐标轴的label;
    plt.xlabel('X voltage')
    plt.ylabel('Y voltage')
    plt.title('X Scan Mode: 5.3ms')
    # 设置x坐标轴刻度;
    plt.xticks(np.linspace(-5, 5, 11))
    plt.yticks(np.linspace(-5, 5, 11))
    plt.plot(x_data, y_data, '*-')
    # plt.savefig('scan_mode.jpg')
    # plt.show()

    # print('x_data: ', x_data)
    # print('y_data: ', y_data)
    dist_mat = cal_distance(x_data, y_data)
    total_dist = 0
    for k in range(total_points - 1):
        total_dist += dist_mat[k][k + 1]
    # print(total_dist)

    """
    2. This part plot the random scan mode.
    """
    ctrl_vxs = np.round(np.random.uniform(low=-5.0, high=5.0, size=total_points), 4)
    ctrl_vys = np.round(np.random.uniform(low=-5.0, high=5.0, size=total_points), 4)


    plt.figure(2)
    # 设置坐标轴的取值范围;
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    # 设置坐标轴的label;
    plt.xlabel('X voltage')
    plt.ylabel('Y voltage')
    plt.title('Random Scan: 12.5ms')
    # 设置x坐标轴刻度;
    plt.xticks(np.linspace(-5, 5, 11))
    plt.yticks(np.linspace(-5, 5, 11))
    plt.plot(ctrl_vxs, ctrl_vys, '*-')
    dist_mat = cal_distance(ctrl_vxs, ctrl_vys)
    total_dist = 0
    for k in range(total_points - 1):
        total_dist += dist_mat[k][k + 1]
    print(total_dist)
    # plt.savefig('random_scan.jpg')

    # Maybe you can use a different plot method.
    # plt.plot(ctrl_vxs, ctrl_vys, 'g*-', label='Random Scan')
    # plt.yticks(fontname="Times New Roman")
    # plt.xticks(fontname="Times New Roman")
    # plt.legend()

    """
    3. This part plot the sorted scan mode.
    """
    sort_vx = np.sort(ctrl_vxs)
    sort_vy = np.sort(ctrl_vys)

    plt.figure(3)
    # 设置坐标轴的取值范围;
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    # 设置坐标轴的label;
    plt.xlabel('X voltage')
    plt.ylabel('Y voltage')
    plt.title('Sorted Scan: 3.5ms')
    # 设置x坐标轴刻度;
    plt.xticks(np.linspace(-5, 5, 11))
    plt.yticks(np.linspace(-5, 5, 11))
    plt.plot(sort_vx, sort_vy, '*-')
    print('x_data: ', sort_vx)
    print('y_data: ', sort_vy)
    print(cal_distance(sort_vx, sort_vy))
    # plt.savefig('sorted_scan.jpg')
    # plt.show()
    dist_mat = cal_distance(sort_vx, sort_vy)
    total_dist = 0
    for k in range(total_points - 1):
        total_dist += dist_mat[k][k + 1]
    print(total_dist)
    # print(sort_vx)

    """
    4. Just test the calculation of distance. Assert the correctness.
    """
    temp_x = np.array([1.5, 2.3, 4.5, 0.9, 1.3, 2.4])
    temp_y = np.array([0.4, 0.8, 1.2, 1.5, 1.4, 2.2])
    dist_mat = cal_distance(temp_x, temp_y)
    dist_total = 0
    print(dist_mat.size)
    print(dist_mat.shape)
    for i in range(len(temp_x)-1):
        dist_total += dist_mat[i][i+1]
    print(dist_total)
    plt.figure('new_test')
    plt.plot(temp_x, temp_y, 'g*-')
    plt.show()

