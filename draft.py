"""
This part is integrated with all kinds of TSP solver, route planning algorithms, add different constraints.
X-axis, Y-axis, grid-scan, down-left--> upper right, assign starting point and ending point, reduce cross-over, etc.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from py2opt.routefinder import RouteFinder
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from ortools.graph import pywrapgraph

np.random.seed(0)


def cal_distance(points_x, points_y, metric='chebyshev'):
    """
    Calculation of distance.
    :param points_x: the x coord of the points.
    :param points_y: the y coord of the points.
    :param metric: the metric to calculate the distance.
    :return: cal_distance.
    """
    data_array = np.array([points_x, points_y]).transpose()
    # print(metric)
    return cdist(data_array, data_array, metric=metric)


def gen_scan_order(original_x, original_y, sc_order='X', total_points=10, separate_grid=10, metric='chebyshev'):
    """
    Generate the scan order of the galvanometer scanning route. X-based, Y-based scan mode.
    :param total_points:
    :param original_x:
    :param original_y:
    :param sc_order: X, x-based scan; Y, y-based scan.
    :param separate_grid: split the X, Y scan space into n-grid, then, based on the split space, using snake scan mode.
    :param metric: the metric to calculate the distance.
    :return: processed scan samples, x_samples, y_samples, scan distance.
    """
    proc_scan_x = []
    proc_scan_y = []
    split_mag = 10.0/separate_grid
    if sc_order == 'X':
        scan_x = original_x
        scan_y = original_y
    else:
        scan_x = original_y
        scan_y = original_x
    for grid_i in range(separate_grid):
        temp_index = []
        y_bit_index = np.bitwise_and(scan_y > (grid_i*split_mag-5), scan_y <= ((grid_i+1)*split_mag-5))
        for j, var in enumerate(y_bit_index):
            if var:
                temp_index.append(j)
        select_x_data = [scan_x[d] for d in temp_index]
        if grid_i % 2 == 0:
            index_sort = np.argsort(select_x_data)
        else:
            index_sort = np.argsort(select_x_data)
            index_sort = index_sort[::-1]
        for k in index_sort:
            proc_scan_x.append(scan_x[temp_index[k]])
            proc_scan_y.append(scan_y[temp_index[k]])
    if sc_order == 'X':
        post_scan_x = proc_scan_x
        post_scan_y = proc_scan_y
    else:
        post_scan_x = proc_scan_y
        post_scan_y = proc_scan_x
    # comb_data = np.array([post_scan_x, post_scan_y]).transpose()
    # dist_real = cdist(comb_data, comb_data, metric=metric)
    dist_real = cal_distance(post_scan_x, post_scan_y, metric=metric)
    total_dist = 0
    for k in range(total_points - 1):
        total_dist += dist_real[k][k + 1]
    print("the samples nums: ", len(post_scan_x))
    return post_scan_x, post_scan_y, total_dist


def two_opt_solver(original_x, original_y, dist_matrix, total_points):
    """
    2-opt solver, exchange the 2 sample points when dist[1,3] + dist[2,4] < dist[1,2] + dist[3,4].
    :param total_points:
    :param original_x:
    :param original_y:
    :param dist_matrix:
    :return: processed data_x, data_y.
    """
    orig_x = original_x
    orig_y = original_y
    all_cities_names = []
    for i in range(total_points):
        all_cities_names.append(i)
    route_funded = RouteFinder(dist_matrix, all_cities_names, iterations=1)
    # start_counter = time.perf_counter()
    best_distance, best_route = route_funded.solve()
    # end_counter = time.perf_counter()
    # print("The total time is (ms): ", (end_counter - start_counter) * 1000)
    sum_dist = best_distance/1000
    resort_data_x = np.zeros(total_points)
    resort_data_y = np.zeros(total_points)
    for i in range(total_points):
        resort_data_x[i] = orig_x[best_route[i]]
        resort_data_y[i] = orig_y[best_route[i]]
    return resort_data_x, resort_data_y, sum_dist


class TSPSolverGoogle:
    """
    Using the Google ortools to solve the TSP problem.
    One can specify the start, end points.
    """
    def __init__(self, original_x, original_y, dist_matrix, total_points, is_defined_points=False,
                 start_point=0, end_point=0):
        """
        :param total_points:
        :param original_x:
        :param original_y:
        :param dist_matrix:
        :param is_defined_points: boolean, whether specify the start and end points.
        """
        self.original_x = original_x
        self.original_y = original_y
        self.distance_matrix = dist_matrix
        self.defined_points = is_defined_points
        self.resort_data_x1 = np.zeros(total_points)
        self.resort_data_y1 = np.zeros(total_points)
        self.start_point = start_point
        self.end_point = end_point
        self.total_points = total_points
        self.travel_dist = 0

    def create_data_model(self):
        """Stores the data for the problem."""
        data = dict()
        data['distance_matrix'] = self.distance_matrix
        data['num_vehicles'] = 1
        if self.defined_points:
            data['starts'] = [self.start_point]
            data['ends'] = [self.end_point]
        else:
            data['depot'] = 0  # start at 0 index of the dataset.
        return data

    @staticmethod
    def print_solution(manager, routing, solution):
        """Prints solution on console."""
        index = routing.Start(0)
        result_index = []
        while not routing.IsEnd(index):
            result_index.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        return result_index

    def solve_travel(self):
        """Entry point of the program."""
        # Instantiate the data problem.
        data = self.create_data_model()

        # Create the routing index manager.
        if self.defined_points:
            manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                                   data['num_vehicles'], data['starts'], data['ends'])
        else:
            manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        #  Add distance constraint. Eason.
        # dimension_name = 'Distance'
        # routing.AddDimension(
        #     transit_callback_index,
        #     0,  # no slack.
        #     20000,  # vehicle maximum total distance.
        #     True,  # start cumul to zero.
        #     dimension_name
        # )
        # distance_dimension = routing.GetDimensionOrDie(dimension_name)
        # distance_dimension.SetGlobalSpanCostCoefficient(100)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)
        # Print solution on console.

        if solution:
            final_index = self.print_solution(manager, routing, solution)

        # print("total len of the sorted index: ", len(final_index))
        # print(final_index)

        # Compute the final distance and scan index.
        if self.defined_points:
            for i in range(self.total_points - 1):
                self.resort_data_x1[i] = self.original_x[final_index[i]]
                self.resort_data_y1[i] = self.original_y[final_index[i]]
                if i < self.total_points-2:
                    self.travel_dist += self.distance_matrix[final_index[i]][final_index[i + 1]]
            self.resort_data_x1[self.total_points-1] = self.original_x[self.end_point]
            self.resort_data_y1[self.total_points-1] = self.original_y[self.end_point]
            self.travel_dist = self.travel_dist + self.distance_matrix[final_index[self.total_points-2]][self.end_point]
        else:
            for i in range(self.total_points):
                self.resort_data_x1[i] = self.original_x[final_index[i]]
                self.resort_data_y1[i] = self.original_y[final_index[i]]
                if i < self.total_points-1:
                    self.travel_dist += self.distance_matrix[final_index[i]][final_index[i + 1]]
        # print(resort_data_x1)
        # print(resort_data_y1)
        # np.save('resort_data_x1.npy', resort_data_x1)
        # np.save('resort_data_y1.npy', resort_data_y1)
        # print(distance_matrix.shape)
        # for i in range(total_points - 2):
            # self.travel_dist += self.distance_matrix[final_index[i]][final_index[i + 1]]
        # print("my_self_cal for google: ", np.round(self.travel_dist, 4))

        return self.resort_data_x1, self.resort_data_y1, np.round(self.travel_dist/1000, 4)


if __name__ == '__main__':
    # method_1 = 'euclidean' chebyshev
    method_2 = 'chebyshev'

    total_samples = 100
    ctrl_vxs = np.round(np.random.uniform(low=-5.0, high=5.0, size=total_samples), 4)
    ctrl_vys = np.round(np.random.uniform(low=-5.0, high=5.0, size=total_samples), 4)
    # ctrl_vxs = np.load('original_x.npy')
    # ctrl_vys = np.load('original_y.npy')

    is_plot_fig = 1  # plot the figure.

    # whether based on the x_ordered_scan/ y_ordered_scan sequence, or just the random generate samples.
    is_based = 0
    based_scan_axis = 'Y'  # If TRUE, given the based X/Y-ordered-scan. Snake scan order.
    split_space = 10  # Split the scan space into n row/column. Then, process the points based on the separate space.

    # whether specify the start/end point. If TRUE, given the two determined points.
    is_defined_start_end = False
    START_POINT = 0
    END_POINT = 22

    distance_matrix = cal_distance(ctrl_vxs, ctrl_vys, metric=method_2)
    scaled_distance_matrix = distance_matrix * 1000
    after_x, after_y, cal_dist = gen_scan_order(ctrl_vxs, ctrl_vys, sc_order=based_scan_axis, total_points=total_samples,
                                                separate_grid=split_space, metric=method_2)
    if not is_based:
        two_opt_x, two_opt_y, two_opt_dist = two_opt_solver(ctrl_vxs, ctrl_vys, scaled_distance_matrix, total_samples)
        google_solver = TSPSolverGoogle(ctrl_vxs, ctrl_vys, scaled_distance_matrix, total_samples,
                                        is_defined_points=is_defined_start_end,
                                        start_point=START_POINT, end_point=END_POINT)
        google_x, google_y, google_dist = google_solver.solve_travel()
        print(google_dist)
    else:
        distance_matrix1 = cal_distance(after_x, after_y, metric=method_2)
        scaled_distance_matrix1 = distance_matrix1 * 1000
        two_opt_x, two_opt_y, two_opt_dist = two_opt_solver(after_x, after_y, scaled_distance_matrix1, total_samples)
        google_solver = TSPSolverGoogle(after_x, after_y, scaled_distance_matrix1, total_samples,
                                        is_defined_points=is_defined_start_end,
                                        start_point=START_POINT, end_point=END_POINT)
        google_x, google_y, google_dist = google_solver.solve_travel()
        # print("The 2opt method's distance: ", two_opt_dist)
        # print("The google method distance: ", google_dist)

    # np.save('01_x_order_x.npy', after_x)
    # np.save('01_x_order_y.npy', after_y)

    # np.save('02_y_order_x.npy', after_y)
    # np.save('02_y_order_y.npy', after_x)

    # np.save('./path_optimize/2opt_scan_x_start.npy', two_opt_x)
    # np.save('./path_optimize/2opt_scan_y_start.npy', two_opt_y)
    #
    # np.save('./path_optimize/google_scan_x_start.npy', google_x)
    # np.save('./path_optimize/google_scan_y_start.npy', google_y)

    # after_x = np.load('./path_optimize/Y_based_order_x.npy')
    # after_y = np.load('./path_optimize/Y_based_order_y.npy')

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
        plt.plot(google_x[99], google_y[99], 'r*')

        plt.plot(ctrl_vxs[START_POINT], ctrl_vys[START_POINT], 'b^')
        plt.plot(ctrl_vxs[END_POINT], ctrl_vys[END_POINT], 'm^')
        # plt.savefig('./path_optimize/google_solver_scan_justtest.jpg')
        plt.show()



