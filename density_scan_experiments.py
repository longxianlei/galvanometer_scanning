"""
This part is designed to conduct density related experiments.
Include Samples-Epochs, Samples-FPS, Samples-Times, Objects-Epochs, etc.
"""
import numpy as np
import matplotlib.pyplot as plt
from draft import cal_distance, gen_scan_order, two_opt_solver, TSPSolverGoogle
np.random.seed(0)

# First, compare x-axis based sorted-scan，2opt-scan and optimized google solver-scan.
# The metric include statistical average routes of scanning points, average scan time of points.


def generate_scan_samples():
    sample_range = [10, 501]  # The sample ranges. From 10 to 1000.
    sample_step = 10  # The step of each sampling.
    avg_routes_axis = []
    avg_routes_2opt = []
    avg_routes_google = []
    axis_based_scan_routes = {'x': [], 'y': []}
    two_opt_scan_routes = {'x': [], 'y': []}
    google_scan_routes = {'x': [], 'y': []}

    for sample_points in range(sample_range[0], sample_range[1], sample_step):
        print(sample_points)
        # method_1 = 'euclidean' chebyshev
        method_2 = 'chebyshev'

        ctrl_vxs = np.round(np.random.uniform(low=-5.0, high=5.0, size=sample_points), 4)
        ctrl_vys = np.round(np.random.uniform(low=-5.0, high=5.0, size=sample_points), 4)
        # ctrl_vxs = np.load('original_x.npy')
        # ctrl_vys = np.load('original_y.npy')

        is_plot_fig = True  # plot the figure.

        # whether based on the x_ordered_scan/ y_ordered_scan sequence, or just the random generate samples.
        is_based = 0
        based_scan_axis = 'X'  # If TRUE, given the based X/Y-ordered-scan. Snake scan order.
        split_space = 10  # Split the scan space into n row/column.

        # whether specify the start/end point. If TRUE, given the two determined points.
        is_defined_start_end = True
        add_all = ctrl_vxs + ctrl_vys
        index_max = np.argmax(add_all)
        index_min = np.argmin(add_all)
        START_POINT = index_min
        END_POINT = index_max
        print('start point: ', START_POINT)
        print('end point: ', END_POINT)

        distance_matrix = cal_distance(ctrl_vxs, ctrl_vys, metric=method_2)
        scaled_distance_matrix = distance_matrix * 1000
        if 0:
            after_x, after_y, cal_dist = gen_scan_order(ctrl_vxs, ctrl_vys, sc_order=based_scan_axis, total_points=sample_points, separate_grid=split_space, metric=method_2)
        if not is_based:
            if 0:
                two_opt_x, two_opt_y, two_opt_dist = two_opt_solver(ctrl_vxs, ctrl_vys, scaled_distance_matrix, sample_points)

            google_solver = TSPSolverGoogle(ctrl_vxs, ctrl_vys, scaled_distance_matrix, sample_points,
                                            is_defined_points=is_defined_start_end,
                                            start_point=int(START_POINT), end_point=int(END_POINT))
            google_x, google_y, google_dist = google_solver.solve_travel()
        else:
            distance_matrix1 = cal_distance(after_x, after_y, metric=method_2)
            scaled_distance_matrix1 = distance_matrix1 * 1000
            two_opt_x, two_opt_y, two_opt_dist = two_opt_solver(after_x, after_y, scaled_distance_matrix1, sample_points)

            google_solver = TSPSolverGoogle(after_x, after_y, scaled_distance_matrix1, sample_points,
                                            is_defined_points=is_defined_start_end,
                                            start_point=START_POINT, end_point=END_POINT)
            google_x, google_y, google_dist = google_solver.solve_travel()
        print('----------------')
        # axis_based_scan_routes['x'].append(after_x)
        # axis_based_scan_routes['y'].append(after_y)
        # two_opt_scan_routes['x'].append(two_opt_x)
        # two_opt_scan_routes['y'].append(two_opt_y)
        google_scan_routes['x'].append(google_x)
        google_scan_routes['y'].append(google_y)
        # avg_routes_axis.append(np.round(cal_dist/sample_points, 4))
        # avg_routes_2opt.append(np.round(two_opt_dist/sample_points, 4))
        avg_routes_google.append(np.round(google_dist/sample_points, 4))

        if is_plot_fig:
            """
            plot scan grid_based x ordered.
            """
            # plt.figure('scan_' + based_scan_axis)
            # # 设置坐标轴的取值范围;
            # plt.xlim((-5, 5))
            # plt.ylim((-5, 5))
            # # 设置坐标轴的label;
            # plt.xlabel('X voltage')
            # plt.ylabel('Y voltage')
            # plt.title(based_scan_axis + ' Sorted Scan distance: ' + str(np.round(cal_dist, 4)))
            # # 设置x坐标轴刻度;
            # plt.xticks(np.linspace(-5, 5, 11))
            # plt.yticks(np.linspace(-5, 5, 11))
            # plt.plot(after_x, after_y, '*-')
            # plt.savefig('./experiment_results/' + str(based_scan_axis) + '_based_solver_scan_' + str(sample_points) + '.jpg')
            # plt.clf()

            """
            plot 2 opt solver scan order.
            """
            # plt.figure('2_opt_scan')
            # # 设置坐标轴的取值范围;
            # plt.xlim((-5, 5))
            # plt.ylim((-5, 5))
            # # 设置坐标轴的label;
            # plt.xlabel('X voltage')
            # plt.ylabel('Y voltage')
            # plt.title('2opt Sorted Scan distance: ' + str(np.round(two_opt_dist, 4)))
            # # 设置x坐标轴刻度;
            # plt.xticks(np.linspace(-5, 5, 11))
            # plt.yticks(np.linspace(-5, 5, 11))
            # plt.plot(two_opt_x, two_opt_y, '*-')
            # plt.savefig('./experiment_results/2opt_solver_scan_' + str(sample_points) + '.jpg')
            # plt.clf()

            """
            plot google solver scan order.
            """
            # plt.figure('google_scan')
            # # 设置坐标轴的取值范围;
            # plt.xlim((-5, 5))
            # plt.ylim((-5, 5))
            # # 设置坐标轴的label;
            # plt.xlabel('X voltage')
            # plt.ylabel('Y voltage')
            # plt.title('Google Sorted Scan distance: ' + str(np.round(google_dist, 4)))
            # # 设置x坐标轴刻度;
            # plt.xticks(np.linspace(-5, 5, 11))
            # plt.yticks(np.linspace(-5, 5, 11))
            # plt.plot(google_x, google_y, '*-')
            # plt.plot(google_x[0], google_y[0], 'r*')
            # plt.plot(google_x[-1], google_y[-1], 'r*')
            # # plt.plot(ctrl_vxs[START_POINT], ctrl_vys[START_POINT], 'b^')
            # # plt.plot(ctrl_vxs[END_POINT], ctrl_vys[END_POINT], 'm^')
            # plt.savefig('./experiment_results/google_solver_scan_' + str(sample_points) + '.jpg')
            # # plt.show()
            # plt.clf()

            """
            plot defined points solver scan order.
            """
            plt.figure('defined_points_scan')
            # 设置坐标轴的取值范围;
            plt.xlim((-5, 5))
            plt.ylim((-5, 5))
            # 设置坐标轴的label;
            plt.xlabel('X voltage')
            plt.ylabel('Y voltage')
            plt.title('Defined Points Sorted Scan distance: ' + str(np.round(google_dist, 4)))
            # 设置x坐标轴刻度;
            plt.xticks(np.linspace(-5, 5, 11))
            plt.yticks(np.linspace(-5, 5, 11))
            plt.plot(google_x, google_y, '*-')
            plt.plot(google_x[0], google_y[0], 'r*')
            plt.plot(google_x[-1], google_y[-1], 'r*')
            plt.plot(ctrl_vxs[START_POINT], ctrl_vys[START_POINT], 'b^')
            plt.plot(ctrl_vxs[END_POINT], ctrl_vys[END_POINT], 'm^')
            plt.savefig('./experiment_results/defined_points_solver_scan_' + str(sample_points) + '.jpg')
            # plt.show()
            plt.clf()

    # np.save('avg_routes_axis.npy', avg_routes_axis)
    # np.save('avg_routes_2opt.npy', avg_routes_2opt)
    # np.save('avg_routes_google.npy', avg_routes_google)
    np.save('avg_routes_assigned.npy', avg_routes_google)

    # np.save('axis_based_scan_routes.npy', axis_based_scan_routes)
    # np.save('two_opt_scan_routes.npy', two_opt_scan_routes)
    # np.save('google_scan_routes.npy', google_scan_routes)
    np.save('defined_points_scan_routes.npy', google_scan_routes)


def load_scan_routes(routes_name):
    scan_data = np.load(routes_name, allow_pickle=True)
    scan_routes_x = scan_data.item().get('x')
    scan_routes_y = scan_data.item().get('y')
    scan_nums = len(scan_routes_x)
    return scan_routes_x, scan_routes_y, scan_nums


if __name__ == '__main__':
    # The name of the 3 scan routes are: axis_based_scan_routes, two_opt_scan_routes, google_scan_routes
    # self_load = np.load('axis_based_scan_routes.npy', allow_pickle=True)
    axis_scan_x, axis_scan_y, axis_scan_nums = load_scan_routes('axis_based_scan_routes.npy')
    topt_scan_x, topt_scan_y, topt_scan_nums = load_scan_routes('two_opt_scan_routes.npy')
    google_scan_x, google_scan_y, google_scan_nums = load_scan_routes('google_scan_routes.npy')
    own_scan_x, own_scan_y, own_scan_nums = load_scan_routes('defined_points_scan_routes.npy')

    # print(axis_scan_nums, topt_scan_nums, google_scan_nums, own_scan_nums)
    print(google_scan_x)
    # test1=np.load('avg_routes_assigned.npy')
    # print(test1)
    # test2 = np.load('defined_points_scan_routes.npy', allow_pickle=True)
    # print(test2.item().get('x'))
    # generate_scan_samples()

    """
    total_samples = 100
    data_x = np.round(np.random.uniform(low=0.0, high=1.0, size=total_samples), 4)
    data_y = np.round(np.random.uniform(low=0.0, high=1.0, size=total_samples), 4)
    scan_x = []
    scan_y = []
    index = 0
    # print(data_x)
    # print(data_y)
    for i in range(10):
        for j in range(10):
            if i % 2 == 0:
                scan_x.append((j-5)+data_x[index])
                scan_y.append((i - 5) + data_y[index])
            else:
                scan_x.append(-((j-5)+data_x[index]))
                scan_y.append((i - 5) + data_y[index])
            index += 1
    # print(scan_x)
    # print(scan_y)
    scan_x = np.array(scan_x)
    scan_y = np.array(scan_y)

    order_scan_x, order_scan_y, order_scan_routes = gen_scan_order(scan_x, scan_y, total_points=total_samples)

    distance_matrix = cal_distance(scan_x, scan_y)
    scaled_distance_matrix = distance_matrix * 1000
    after_x, after_y, cal_dist = gen_scan_order(scan_x, scan_y,
                                                    total_points=total_samples, separate_grid=20)

    two_opt_x, two_opt_y, two_opt_dist = two_opt_solver(scan_x, scan_y, scaled_distance_matrix,
                                                            total_points=total_samples)

    google_solver = TSPSolverGoogle(scan_x, scan_y, scaled_distance_matrix, total_points=total_samples,
                                    is_defined_points=False,
                                    start_point=6, end_point=9)
    google_x, google_y, google_dist = google_solver.solve_travel()

    add_all = scan_x + scan_y
    index_max = np.argmax(add_all)
    index_min = np.argmin(add_all)
    START_POINT = index_min
    END_POINT = index_max

    google_solver = TSPSolverGoogle(scan_x, scan_y, scaled_distance_matrix, total_points=total_samples,
                                    is_defined_points=True,
                                    start_point=int(START_POINT), end_point=int(END_POINT))
    own_x, own_y, own_dist = google_solver.solve_travel()


    print('avg_routes: ', order_scan_routes)
    plt.figure(3)
    # 设置坐标轴的取值范围;
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    # 设置坐标轴的label;
    plt.xlabel('X voltage')
    plt.ylabel('Y voltage')
    plt.title('Grid Based Scan:' + str(np.round(order_scan_routes/total_samples, 4)))
    # 设置x坐标轴刻度;
    plt.xticks(np.linspace(-5, 5, 11))
    plt.yticks(np.linspace(-5, 5, 11))

    plt.plot(scan_x, scan_y, '*-')
    plt.savefig('1_grid_based_scan.jpg')
    # plt.show()

    plt.figure(4)
    # 设置坐标轴的取值范围;
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    # 设置坐标轴的label;
    plt.xlabel('X voltage')
    plt.ylabel('Y voltage')
    plt.title('X-axis Order Based Scan: ' + str(np.round(cal_dist/total_samples, 4)))
    # 设置x坐标轴刻度;
    plt.xticks(np.linspace(-5, 5, 11))
    plt.yticks(np.linspace(-5, 5, 11))
    plt.plot(after_x, after_y, 'mo-.')
    plt.savefig('1_X-axis_based_Scan_grid_20.jpg')

    plt.figure(5)
    # 设置坐标轴的取值范围;
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    # 设置坐标轴的label;
    plt.xlabel('X voltage')
    plt.ylabel('Y voltage')
    plt.title('2opt Based Scan: ' + str(np.round(two_opt_dist/total_samples, 4)))
    # 设置x坐标轴刻度;
    plt.xticks(np.linspace(-5, 5, 11))
    plt.yticks(np.linspace(-5, 5, 11))
    plt.plot(two_opt_x, two_opt_y, 'b*--')
    plt.plot(two_opt_x[0], two_opt_y[0], 'r*')
    plt.plot(two_opt_x[-1], two_opt_y[-1], 'r*')
    plt.savefig('1_two_opt_based_scan.jpg')

    plt.figure(6)
    # 设置坐标轴的取值范围;
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    # 设置坐标轴的label;
    plt.xlabel('X voltage')
    plt.ylabel('Y voltage')
    plt.title('Google Based Scan: ' + str(np.round(google_dist/total_samples, 4)))
    # 设置x坐标轴刻度;
    plt.xticks(np.linspace(-5, 5, 11))
    plt.yticks(np.linspace(-5, 5, 11))
    plt.plot(google_x, google_y, 'y^--')
    plt.plot(google_x[0], google_y[0], 'r*')
    plt.plot(google_x[-1], google_y[-1], 'r*')
    plt.savefig('1_google_based_scan_new.jpg')

    plt.figure(7)
    # 设置坐标轴的取值范围;
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    # 设置坐标轴的label;
    plt.xlabel('X voltage')
    plt.ylabel('Y voltage')
    plt.title('Our solver Based Scan: ' + str(np.round(own_dist/total_samples, 4)))
    # 设置x坐标轴刻度;
    plt.xticks(np.linspace(-5, 5, 11))
    plt.yticks(np.linspace(-5, 5, 11))
    plt.plot(own_x, own_y, 'g^--')
    plt.plot(scan_x[START_POINT], scan_y[START_POINT], 'r*')
    plt.plot(scan_x[END_POINT], scan_y[END_POINT], 'r*')
    plt.savefig('1_own_solver_based_scan.jpg')
    plt.show()
    """


