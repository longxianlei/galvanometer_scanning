import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from py2opt.routefinder import RouteFinder
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from ortools.graph import pywrapgraph

np.random.seed(0)


def cal_distance(points_x, points_y, metric='chebyshev'):
    data_array = np.array([points_x, points_y]).transpose()
    print(metric)
    return cdist(data_array, data_array, metric=metric)


def gen_scan_order(original_x, original_y, sc_order='X', separate_grid=10, metric='chebyshev'):
    proc_scan_x = []
    proc_scan_y = []
    if sc_order == 'X':
        scan_x = original_x
        scan_y = original_y
    else:
        scan_x = original_y
        scan_y = original_x
    for grid_i in range(separate_grid):
        temp_index = []
        y_bit_index = np.bitwise_and(scan_y > (grid_i-5), scan_y <= (grid_i-4))
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
    comb_data = np.array([post_scan_x, post_scan_y]).transpose()
    dist_real = cdist(comb_data, comb_data, metric=metric)
    total_dist = 0
    for k in range(total_points - 1):
        total_dist += dist_real[k][k + 1]
    return post_scan_x, post_scan_y, total_dist


def two_opt_solver(original_x, original_y, dist_matrix):
    orig_x = original_x
    orig_y = original_y
    # dist_mat = dist_matrix
    all_cities_names = []
    for i in range(total_points):
        all_cities_names.append(i)
    route_funded = RouteFinder(dist_matrix, all_cities_names, iterations=1)
    # start_counter = time.perf_counter()
    best_distance, best_route = route_funded.solve()
    # end_counter = time.perf_counter()
    # print("The total time is (ms): ", (end_counter - start_counter) * 1000)
    print('best_distance: ', best_distance)
    sum_dist = best_distance/1000
    # sum_dist = 0
    # for i in range(total_points - 1):
    #     sum_dist += dist_mat[best_route[i]][best_route[i + 1]]
    # print("my_self_cal: ", np.round(sum_dist, 4))
    resort_data_x = np.zeros(total_points)
    resort_data_y = np.zeros(total_points)
    for i in range(total_points):
        resort_data_x[i] = orig_x[best_route[i]]
        resort_data_y[i] = orig_y[best_route[i]]
    return resort_data_x, resort_data_y, sum_dist


class TSPSolverGoogle:
    def __init__(self, original_x, original_y, dist_matrix, is_defined_points = False):
        self.original_x = original_x
        self.original_y = original_y
        self.distance_matrix = dist_matrix
        self.defined_points = is_defined_points
        self.resort_data_x1 = np.zeros(total_points)
        self.resort_data_y1 = np.zeros(total_points)
        self.travel_dist = 0

    def create_data_model(self):
        """Stores the data for the problem."""
        data = dict()
        data['distance_matrix'] = self.distance_matrix
        data['num_vehicles'] = 1
        if self.defined_points:
            data['starts'] = [START_POINT]
            data['ends'] = [END_POINT]
        else:
            data['depot'] = 0
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
        print('begin to solve google')
        solution = routing.SolveWithParameters(search_parameters)
        print('end of google')
        # Print solution on console.

        if solution:
            final_index = self.print_solution(manager, routing, solution)

        print(len(final_index))
        if self.defined_points:
            for i in range(total_points - 1):
                self.resort_data_x1[i] = self.original_x[final_index[i]]
                self.resort_data_y1[i] = self.original_y[final_index[i]]
                if i < total_points-2:
                    self.travel_dist += self.distance_matrix[final_index[i]][final_index[i + 1]]
            self.resort_data_x1[99] = self.original_x[END_POINT]
            self.resort_data_y1[99] = self.original_y[END_POINT]
            self.travel_dist = self.travel_dist + self.distance_matrix[final_index[total_points-2]][END_POINT]
        else:
            for i in range(total_points):
                self.resort_data_x1[i] = self.original_x[final_index[i]]
                self.resort_data_y1[i] = self.original_y[final_index[i]]
                if i < total_points-1:
                    self.travel_dist += self.distance_matrix[final_index[i]][final_index[i + 1]]
        # print(resort_data_x1)
        # print(resort_data_y1)
        # np.save('resort_data_x1.npy', resort_data_x1)
        # np.save('resort_data_y1.npy', resort_data_y1)
        # print(distance_matrix.shape)
        # for i in range(total_points - 2):
            # self.travel_dist += self.distance_matrix[final_index[i]][final_index[i + 1]]
        print("my_self_cal for google: ", np.round(self.travel_dist, 4))

        return self.resort_data_x1, self.resort_data_y1, self.travel_dist/1000


if __name__ == '__main__':
    # method_1 = 'euclidean' chebyshev
    method_2 = 'chebyshev'

    total_points = 100
    ctrl_vxs = np.round(np.random.uniform(low=-5.0, high=5.0, size=total_points), 4)
    ctrl_vys = np.round(np.random.uniform(low=-5.0, high=5.0, size=total_points), 4)
    # ctrl_vxs = np.load('original_x.npy')
    # ctrl_vys = np.load('original_y.npy')

    is_plot_fig = 1
    # whether based on the x_ordered_scan sequence or y_ordered_scan sequence, or just the random generate samples.
    is_based = 0
    is_defined_start_end = 0
    START_POINT = 0
    END_POINT = 22
    distance_matrix = cal_distance(ctrl_vxs, ctrl_vys, metric=method_2)
    scaled_distance_matrix = distance_matrix * 1000
    after_x, after_y, cal_dist = gen_scan_order(ctrl_vxs, ctrl_vys, sc_order='X', metric=method_2)
    if not is_based:
        two_opt_x, two_opt_y, two_opt_dist = two_opt_solver(ctrl_vxs, ctrl_vys, scaled_distance_matrix)
        google_solver = TSPSolverGoogle(ctrl_vxs, ctrl_vys, scaled_distance_matrix, is_defined_points= is_defined_start_end)
        google_x, google_y, google_dist = google_solver.solve_travel()
        print(google_dist)
    else:
        distance_matrix1 = cal_distance(after_x, after_y, metric=method_2)
        scaled_distance_matrix1 = distance_matrix1 * 1000
        two_opt_x, two_opt_y, two_opt_dist = two_opt_solver(after_x, after_y, scaled_distance_matrix1)
        google_solver = TSPSolverGoogle(after_x, after_y, scaled_distance_matrix1, is_defined_points=is_defined_start_end)
        google_x, google_y, google_dist = google_solver.solve_travel()

    # Calculate the distance of the ordered sequence.
    # man_dist = 0
    # for i in range(total_points-1):
    #     man_dist += max(abs(after_x[i]-after_x[i+1]), abs(after_y[i] - after_y[i+1]))

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
        '''
        plot scan grid_based x ordered .
        '''
        plt.figure('scan_X')
        # 设置坐标轴的取值范围;
        plt.xlim((-5, 5))
        plt.ylim((-5, 5))
        # 设置坐标轴的label;
        plt.xlabel('X voltage')
        plt.ylabel('Y voltage')
        plt.title('X Sorted Scan distance: ' + str(np.round(cal_dist, 4)))
        # 设置x坐标轴刻度;
        plt.xticks(np.linspace(-5, 5, 11))
        plt.yticks(np.linspace(-5, 5, 11))
        plt.plot(after_x, after_y, '*-')
        plt.savefig('./path_optimize/Y_based_solver_scan_justtest.jpg')

        '''
        plot 2 opt solver scan order.
        '''
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
        plt.savefig('./path_optimize/2opt_solver_scan_start_justtest.jpg')

        '''
        plot google solver scan order.
        '''
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
        plt.plot(google_x[0], google_y[0], 'g*')
        plt.plot(google_x[99], google_y[99], 'r*')
        plt.savefig('./path_optimize/google_solver_scan_justtest.jpg')
        plt.show()



