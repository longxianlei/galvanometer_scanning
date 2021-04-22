"""Simple travelling salesman problem between cities."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from py2opt.routefinder import RouteFinder
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from draft import cal_distance

np.random.seed(0)
total_points = 100

"""
1. Generate the X-ordered scanning routes.
"""
x = np.arange(-4.5, 5, 1.0)
y = np.arange(-4.5, 5, 1.0)
ctrl_vxs = np.zeros(total_points)
ctrl_vys = np.zeros(total_points)
for i in range(total_points):
    if (i // 10) % 2 == 0:
        ctrl_vxs[i] = x[i % 10]
    else:
        ctrl_vxs[i] = x[10 - 1 - i % 10]
    ctrl_vys[i] = y[i // 10]


"""
2. Generate 100 random samples, then, random scan the samples one by one. 
"""
ctrl_vxs = np.round(np.random.uniform(low=-5.0, high=5.0, size=total_points), 4)
ctrl_vys = np.round(np.random.uniform(low=-5.0, high=5.0, size=total_points), 4)
# print(ctrl_vxs)
# print(ctrl_vys)
dist_mat = cal_distance(ctrl_vxs, ctrl_vys)
total_dist = 0
for k in range(total_points - 1):
    total_dist += dist_mat[k][k + 1]
print(total_dist)

# plot the figure.
plt.figure('random scan')
# 设置坐标轴的取值范围;
plt.xlim((-5, 5))
plt.ylim((-5, 5))
# 设置坐标轴的label;
plt.xlabel('X voltage')
plt.ylabel('Y voltage')
plt.title('Random Scan routes: ' + str(np.round(total_dist, 4)))
# 设置x坐标轴刻度;
plt.xticks(np.linspace(-5, 5, 11))
plt.yticks(np.linspace(-5, 5, 11))
plt.plot(ctrl_vxs, ctrl_vys, '*-')

# np.save('original_x.npy', ctrl_vxs)
# np.save('original_y.npy', ctrl_vys)


"""
3. Using the 2opt method: Route finder to get the optimized scanning routes.
    First, construct the distance matrix. In our case, this is a symmetric distance matrix. DistMtx = DistMtx^T.
    Using the chebyshev distance to measure the distance of two 2D points. distance = Max(|x1 - x2|, |y1-y2|).
"""
method_1 = 'euclidean'
method_2 = 'chebyshev'
chebyshev_dist_real = cal_distance(ctrl_vxs, ctrl_vys)
chebyshev_dist_real = np.round(chebyshev_dist_real*1000)
ab_citi_names = []
for i in range(total_points):
    ab_citi_names.append(i)
dist_mat = chebyshev_dist_real
# Instantiate an object called route_found, given the distance matrix, city names, iterations.
route_found = RouteFinder(dist_mat, ab_citi_names, iterations=1)
# start_counter = time.perf_counter()
best_distance, best_route = route_found.solve()
# end_counter = time.perf_counter()
# print("The total time is (ms): ", (end_counter - start_counter) * 1000)
print(best_distance)
print(best_route)

# This is our manual calculation of the routines' distance. Which is the same as the solver returned.
# sum_dist = 0
# for i in range(total_points-1):
#     sum_dist += chebyshev_dist_real[best_route[i]][best_route[i+1]]
# print("my_self_cal: ", np.round(sum_dist/1000, 4))

resort_data_x = np.zeros(total_points)
resort_data_y = np.zeros(total_points)
for i in range(total_points):
    resort_data_x[i] = ctrl_vxs[best_route[i]]
    resort_data_y[i] = ctrl_vys[best_route[i]]

# plot the scan routes of routefinder.
plt.figure('2opt route finder scan.')
# 设置坐标轴的取值范围;
plt.xlim((-5, 5))
plt.ylim((-5, 5))
# 设置坐标轴的label;
plt.xlabel('X voltage')
plt.ylabel('Y voltage')
plt.title('RouteFinder Sorted Scan routes: '+str(np.round(best_distance/1000, 4)))
# 设置x坐标轴刻度;
plt.xticks(np.linspace(-5, 5, 11))
plt.yticks(np.linspace(-5, 5, 11))
plt.plot(resort_data_x, resort_data_y, '*-')


# Construct the google solver for TSP.
class GoogleSolver:
    """
    4. Construct the GoogleSolver based on ortools for solving constrain optimization problems.
    The solver Class only need to specify the distance matrix. Then, the solver will do all the remains.
    """
    @staticmethod
    def create_data_model(self):
        """Stores the data for the problem."""
        data = dict()
        data['distance_matrix'] = chebyshev_dist_real
        data['num_vehicles'] = 1
        data['depot'] = 0
        return data

    @staticmethod
    def print_solution(manager, routing, solution):
        """Prints solution on console."""
        print('Objective: {} miles'.format(solution.ObjectiveValue()))
        index = routing.Start(0)
        plan_output = 'Route for vehicle 0:\n'
        route_distance = 0
        result_index = []
        while not routing.IsEnd(index):
            plan_output += ' {} ->'.format(manager.IndexToNode(index))
            result_index.append(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        plan_output += ' {}\n'.format(manager.IndexToNode(index))
        print(plan_output)
        print(result_index)
        plan_output += 'Route distance: {}miles\n'.format(route_distance)
        return result_index

    def main(self):
        """Entry point of the program."""
        # Instantiate the data problem.
        data = self.create_data_model(self)

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                               data['num_vehicles'], data['depot'])
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

        # Setting first solution heuristic.
        # The first solution strategy is the method the solver uses to find an initial solution.
        # Saving, CHRISTOFIDES, LOCAL_CHEAPEST_INSERTION, GLOBAL_CHEAPEST_ARC

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        # Add some fun. Just for test. Eason.
        # The guided local search, which enables the solver to escape a local minimum.
        # The belows are local search strategies (also called metaheuristics.
        # The methods included Automatic, Greedy_search, simulated_annealing, tabu_search, ....
        # search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        # search_parameters.local_search_metaheuristic = (
        #     routing_enums_pb2.LocalSearchMetaheuristic.OBJECTIVE_TABU_SEARCH
        # )
        # search_parameters.time_limit.seconds = 10
        # search_parameters.solution_limit = 159
        # search_parameters.log_search = True

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        # Print solution on console.
        if solution:
            final_index = self.print_solution(manager, routing, solution)
        return final_index


if __name__ == '__main__':
    """
    4. This method using Google solver to optimize the routes. The distance matrix is the same as RouteFinder.
    """
    google_routes_solver = GoogleSolver()
    travel_index = google_routes_solver.main()
    resort_data_x1 = np.zeros(total_points)
    resort_data_y1 = np.zeros(total_points)
    for i in range(total_points):
        resort_data_x1[i] = ctrl_vxs[travel_index[i]]
        resort_data_y1[i] = ctrl_vys[travel_index[i]]
    # print(resort_data_x1)
    # print(resort_data_y1)
    # np.save('resort_data_x1.npy', resort_data_x1)
    # np.save('resort_data_y1.npy', resort_data_y1)

    sum_dist_2 = 0
    for i in range(total_points - 1):
        sum_dist_2 += chebyshev_dist_real[travel_index[i]][travel_index[i + 1]]
    print("my_self_cal: ", np.round(sum_dist_2/1000, 4))
    plt.figure('google scan routes')
    # 设置坐标轴的取值范围;
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))

    # 设置坐标轴的label;
    plt.xlabel('X voltage')
    plt.ylabel('Y voltage')
    plt.title('Google Solver Sorted Scan routes: '+str(np.round(sum_dist_2/1000, 4)))
    # 设置x坐标轴刻度;
    plt.xticks(np.linspace(-5, 5, 11))
    plt.yticks(np.linspace(-5, 5, 11))
    plt.plot(resort_data_x1, resort_data_y1, '*-')
    plt.show()
