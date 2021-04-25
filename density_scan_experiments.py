"""
This part is designed to conduct density related experiments.
Include Samples-Epochs, Samples-FPS, Samples-Times, Objects-Epochs, etc.
"""
import numpy as np
import matplotlib.pyplot as plt
from draft import cal_distance, gen_scan_order, two_opt_solver, TSPSolverGoogle
np.random.seed(0)

# First, compare x-axis based sorted-scan and optimized google solver-scan.
# The metric include statistical average length of points, average scan time of points.

sample_range = [10, 51]  # The sample ranges. From 10 to 1000.
sample_step = 10  # The step of each sampling.
avg_routes_axis = []
avg_routes_2opt = []
avg_routes_google = []

for sample_points in range(sample_range[0], sample_range[1], sample_step):
    # method_1 = 'euclidean' chebyshev
    method_2 = 'chebyshev'

    ctrl_vxs = np.round(np.random.uniform(low=-5.0, high=5.0, size=sample_points), 4)
    ctrl_vys = np.round(np.random.uniform(low=-5.0, high=5.0, size=sample_points), 4)
    # ctrl_vxs = np.load('original_x.npy')
    # ctrl_vys = np.load('original_y.npy')

    is_plot_fig = False  # plot the figure.

    # whether based on the x_ordered_scan/ y_ordered_scan sequence, or just the random generate samples.
    is_based = 0
    based_scan_axis = 'X'  # If TRUE, given the based X/Y-ordered-scan. Snake scan order.
    split_space = 10  # Split the scan space into n row/column. Then, process the points based on the separate space.

    # whether specify the start/end point. If TRUE, given the two determined points.
    is_defined_start_end = False
    START_POINT = 0
    END_POINT = 77

    distance_matrix = cal_distance(ctrl_vxs, ctrl_vys, metric=method_2)
    scaled_distance_matrix = distance_matrix * 1000
    after_x, after_y, cal_dist = gen_scan_order(ctrl_vxs, ctrl_vys, sc_order=based_scan_axis,
                                                total_points=sample_points, separate_grid=split_space, metric=method_2)
    if not is_based:
        two_opt_x, two_opt_y, two_opt_dist = two_opt_solver(ctrl_vxs, ctrl_vys, scaled_distance_matrix, sample_points)
        google_solver = TSPSolverGoogle(ctrl_vxs, ctrl_vys, scaled_distance_matrix, sample_points,
                                        is_defined_points=is_defined_start_end)
        google_x, google_y, google_dist = google_solver.solve_travel()
    else:
        distance_matrix1 = cal_distance(after_x, after_y, metric=method_2)
        scaled_distance_matrix1 = distance_matrix1 * 1000
        two_opt_x, two_opt_y, two_opt_dist = two_opt_solver(after_x, after_y, scaled_distance_matrix1, sample_points)
        google_solver = TSPSolverGoogle(after_x, after_y, scaled_distance_matrix1, sample_points,
                                        is_defined_points=is_defined_start_end)
        google_x, google_y, google_dist = google_solver.solve_travel()
        # print("The 2opt method's distance: ", two_opt_dist)
        # print("The google method distance: ", google_dist)
    print('----------------')
    avg_routes_axis.append(np.round(cal_dist/sample_points, 4))
    avg_routes_2opt.append(np.round(two_opt_dist/sample_points, 4))
    avg_routes_google.append(np.round(google_dist/sample_points, 4))

    if is_plot_fig:
        """
        plot scan grid_based x ordered.
        """
        plt.figure('scan_' + based_scan_axis)
        # 设置坐标轴的取值范围;
        plt.xlim((-5, 5))
        plt.ylim((-5, 5))
        # 设置坐标轴的label;
        plt.xlabel('X voltage')
        plt.ylabel('Y voltage')
        plt.title(based_scan_axis + ' Sorted Scan distance: ' + str(np.round(cal_dist, 4)))
        # 设置x坐标轴刻度;
        plt.xticks(np.linspace(-5, 5, 11))
        plt.yticks(np.linspace(-5, 5, 11))
        plt.plot(after_x, after_y, '*-')
        # plt.savefig('./path_optimize/Y_based_solver_scan_justtest.jpg')

        """
        plot 2 opt solver scan order.
        """
        plt.figure('2_opt_scan')
        # 设置坐标轴的取值范围;
        plt.xlim((-5, 5))
        plt.ylim((-5, 5))
        # 设置坐标轴的label;
        plt.xlabel('X voltage')
        plt.ylabel('Y voltage')
        plt.title('2opt Sorted Scan distance: ' + str(np.round(two_opt_dist, 4)))
        # 设置x坐标轴刻度;
        plt.xticks(np.linspace(-5, 5, 11))
        plt.yticks(np.linspace(-5, 5, 11))
        plt.plot(two_opt_x, two_opt_y, '*-')
        # plt.savefig('./path_optimize/2opt_solver_scan_start_justtest.jpg')

        """
        plot google solver scan order.
        """
        plt.figure('google_scan')
        # 设置坐标轴的取值范围;
        plt.xlim((-5, 5))
        plt.ylim((-5, 5))
        # 设置坐标轴的label;
        plt.xlabel('X voltage')
        plt.ylabel('Y voltage')
        plt.title('Google Sorted Scan distance: ' + str(np.round(google_dist, 4)))
        # 设置x坐标轴刻度;
        plt.xticks(np.linspace(-5, 5, 11))
        plt.yticks(np.linspace(-5, 5, 11))
        plt.plot(google_x, google_y, '*-')
        plt.plot(google_x[0], google_y[0], 'r*')
        plt.plot(google_x[-1], google_y[-1], 'r*')

        # plt.plot(ctrl_vxs[START_POINT], ctrl_vys[START_POINT], 'b^')
        # plt.plot(ctrl_vxs[END_POINT], ctrl_vys[END_POINT], 'm^')
        # plt.savefig('./path_optimize/google_solver_scan_justtest.jpg')
        plt.show()

print(avg_routes_axis)
print(avg_routes_2opt)
print(avg_routes_google)
np.save('avg_routes_axis.npy', avg_routes_axis)
np.save('avg_routes_2opt.npy', avg_routes_2opt)
np.save('avg_routes_google.npy', avg_routes_google)
