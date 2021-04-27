# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sample_points = 10
    ctrl_vxs = np.round(np.random.uniform(low=-5.0, high=5.0, size=sample_points), 4)
    ctrl_vys = np.round(np.random.uniform(low=-5.0, high=5.0, size=sample_points), 4)

    print(ctrl_vxs)
    print(ctrl_vys)

    add_all = ctrl_vxs + ctrl_vys
    index_max = np.argmax(add_all)
    index_min = np.argmin(add_all)
    print(add_all)
    print(index_max)
    print(index_min)
    print(type(index_min))


    a1 = np.array([0,1,1,1])
    a2 = np.array([1,0,0,1])
    a3 = a1+a2
    max_index = np.argmax(a3)
    min_index = np.argmin(a3)
    print(max_index)
    print(min_index)

    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
