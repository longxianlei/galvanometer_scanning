"""Simple travelling salesman problem between cities."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from py2opt.routefinder import RouteFinder
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
np.random.seed(0)

total_points = 100

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

ctrl_vxs = np.round(np.random.uniform(low=-5.0, high=5.0, size=total_points), 4)
ctrl_vys = np.round(np.random.uniform(low=-5.0, high=5.0, size=total_points), 4)
# print(ctrl_vxs)
# print(ctrl_vys)

plt.figure(2)
# 设置坐标轴的取值范围;
plt.xlim((-5, 5))
plt.ylim((-5, 5))

# 设置坐标轴的label;
plt.xlabel('X voltage')
plt.ylabel('Y voltage')
plt.title('Sorted Scan distance')
# 设置x坐标轴刻度;
plt.xticks(np.linspace(-5, 5, 11))
plt.yticks(np.linspace(-5, 5, 11))
plt.plot(ctrl_vxs, ctrl_vys, '*-')


# np.save('original_x.npy', ctrl_vxs)
# np.save('original_y.npy', ctrl_vys)

real_data = np.array([ctrl_vxs, ctrl_vys]).transpose()
method_1 = 'euclidean'
method_2 = 'chebyshev'
chebyshev_dist_real = cdist(real_data, real_data, metric=method_2)
chebyshev_dist_real = np.round(chebyshev_dist_real*1000)
print(chebyshev_dist_real[0][12])
print(chebyshev_dist_real)
# cities_names = ['a', 'b', 'c', 'd', 'e']
ab_citi_names = []
for i in range(total_points):
    ab_citi_names.append(i)
print(ab_citi_names)
dist_mat = chebyshev_dist_real
route_finded = RouteFinder(dist_mat, ab_citi_names, iterations=1)
# start_counter = time.perf_counter()
best_distance, best_route = route_finded.solve()
# end_counter = time.perf_counter()
# print("The total time is (ms): ", (end_counter - start_counter) * 1000)
print(best_distance)
print(best_route)
sum_dist = 0
for i in range(total_points-1):
    sum_dist += chebyshev_dist_real[best_route[i]][best_route[i+1]]
print("my_self_cal: ", np.round(sum_dist/1000, 4))

resort_data_x = np.zeros(total_points)
resort_data_y = np.zeros(total_points)
for i in range(total_points):
    resort_data_x[i] = ctrl_vxs[best_route[i]]
    resort_data_y[i] = ctrl_vys[best_route[i]]
# print(resort_data_x)
# print(resort_data_y)

plt.figure(7)
# 设置坐标轴的取值范围;
plt.xlim((-5, 5))
plt.ylim((-5, 5))

# 设置坐标轴的label;
plt.xlabel('X voltage')
plt.ylabel('Y voltage')
plt.title('Sorted Scan distance: '+str(np.round(sum_dist/1000, 4)))
# 设置x坐标轴刻度;
plt.xticks(np.linspace(-5, 5, 11))
plt.yticks(np.linspace(-5, 5, 11))
plt.plot(resort_data_x, resort_data_y, '*-')


def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = chebyshev_dist_real
    # data['distance_matrix'] = [
    #     [0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972],
    #     [2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579],
    #     [713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260],
    #     [1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987],
    #     [1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371],
    #     [1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999],
    #     [2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701],
    #     [213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099],
    #     [2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600],
    #     [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162],
    #     [1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200],
    #     [2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504],
    #     [1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0],
    # ]  # yapf: disable
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data


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


def main():
    """Entry point of the program."""
    # Instantiate the data problem.
    data = create_data_model()

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
        final_index = print_solution(manager, routing, solution)
    return final_index


if __name__ == '__main__':
    travel_index = main()
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

    plt.figure(8)
    # 设置坐标轴的取值范围;
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))

    # 设置坐标轴的label;
    plt.xlabel('X voltage')
    plt.ylabel('Y voltage')
    plt.title('Sorted Scan: '+str(np.round(sum_dist_2/1000, 4)))
    # 设置x坐标轴刻度;
    plt.xticks(np.linspace(-5, 5, 11))
    plt.yticks(np.linspace(-5, 5, 11))
    plt.plot(resort_data_x1, resort_data_y1, '*-')
    plt.show()
