import zmq
import time
import random
import os
import numpy as np
import matplotlib.pyplot as plt

context = zmq.Context()
# Socket to talk to server
print("Connecting to hello world server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")
# socket.connect("tcp://192.168.3.41:5000")


def send_ctrl_voltage(series_num, total_nums, index, ctrl_vx, ctrl_vy):
    r"""
    send the control voltage to the motor.
    series_num:序列号，第几次发送；1 to N;
    total_num: 当前part总共发送多少次；N;
    index:当前采样点的索引号； 0 ~ N-1
    ctrl_vx:当前采样点的x轴控制电压；
    ctrl_vy:当前采样点的y轴控制电压；
    send_fun(series_num, total_num, index, ctrl_vx, ctrl_vy)

    例子：
    send_fun(1, 200, 3, 4.23, -1.33)
    send_fun(3, 350, 145, -1.59, -0.633)
    """
    send_msg = str(series_num) + str(",") + str(total_nums) + str(",") + str(index) + str(",") + \
               str(ctrl_vx) + str(",") + str(ctrl_vy)
    socket.send_string(send_msg)
    received_msg = socket.recv_string()
    print(received_msg)
    received_flag = -1
    if received_msg == 'Received : Normal':
        received_flag = 0
    elif received_msg == 'Received : Finish':
        received_flag = 1
    elif received_msg == 'Received : Parse Error':
        received_flag = -1
    return received_flag


def random_int_list(start, stop, length):
    """
    generate the initial random uniform sampling control points.
    :param start: the min voltage;
    :param stop: the max voltage;
    :param length: how many sample points.
    :return: generated points.
    """
    random_list = []
    for temp in range(length):
        random_list.append(round(random.uniform(start, stop), 4))
    return random_list


def read_processed_targets(results_path):
    targets_list = []
    for root, dirs, files in os.walk(results_path):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                file_name = file[0:-4]
                # print(file_name)
                targets_list.append(file_name)
    detected_x = []
    detected_y = []
    for detect_txt in targets_list:
        temp_path_list = detect_txt.split('_')
        detected_x.append(float(temp_path_list[2]))
        detected_y.append(-float(temp_path_list[3]))
    return detected_x, detected_y


if __name__ == '__main__':
    experiments_date = '02_28_detect_result'
    total_rounds = 2
    total_points = 100
    is_send_msg = 1
    is_save_fig = 0
    # if not os.path.exists(r'C:/RapidEye/record/'+experiments_date):
    #     os.mkdir('C:/RapidEye/record/'+experiments_date)
    plot_vxs = []
    plot_vys = []
    process_times = []
    if is_send_msg:
        for series_index in range(total_rounds):
            start_counter = time.perf_counter()
            # ctrl_vxs = random_int_list(-5, 5, length=100)
            ctrl_vxs = np.round(np.random.uniform(low=-5.0, high=5.0, size=total_points), 4)
            ctrl_vys = np.round(np.random.uniform(low=-5.0, high=5.0, size=total_points), 4)
            plot_vxs.append(ctrl_vxs)
            plot_vys.append(ctrl_vys)
            send_count = 0
            for ctrl_pts in range(total_points):
                cnt_time = time.perf_counter()
                return_flag = send_ctrl_voltage(series_num=series_index + 1,
                                                total_nums=total_points,
                                                index=ctrl_pts,
                                                ctrl_vx=ctrl_vxs[ctrl_pts],
                                                ctrl_vy=ctrl_vys[ctrl_pts])
                print("send cnt: ", str(ctrl_pts), "; send time is (ms): ",
                      np.round((time.perf_counter()-cnt_time)*1000, 4))
                # print(return_flag)
                if return_flag == 0:
                    pass
                elif return_flag == 1:
                    break
                elif return_flag == -1:
                    print("The send message was parsed error!!!")
                    exit()
            end_counter = time.perf_counter()
            per_process_time = np.round((end_counter-start_counter)*1000/total_points, 4)
            print("The total time is (ms): ", (end_counter - start_counter)*1000)
            print("Total counts is: ", total_points)
            print("Each mq times is (ms): ", per_process_time)
            try:
                with open('C:/RapidEye/processing_time.txt', 'a') as f:
                    f.write('{}\n'.format(per_process_time))
                f.close()
            except:
                pass
        
    # print(plot_vxs)
    # print(plot_vxs[1])
    if is_save_fig:
        detected_points_x, detected_points_y = read_processed_targets('C:/RapidEye/record/'+experiments_date)
        # print(detected_points_x, detected_points_y)
        for i in range(total_rounds):
            plt.figure()
            # 设置坐标轴的取值范围;
            plt.xlim((-5, 5))
            plt.ylim((-5, 5))

            # 设置坐标轴的label;
            plt.xlabel('X voltage')
            plt.ylabel('Y voltage')

            # 设置x坐标轴刻度;
            plt.xticks(np.linspace(-5, 5, 11))
            plt.yticks(np.linspace(-5, 5, 11))

            if is_send_msg:
                plt.plot(plot_vxs[i], - plot_vys[i], '*')
            plt.plot(detected_points_x, detected_points_y, 'r.')
            plt.savefig('C:/RapidEye/record/'+experiments_date+'/'+'ttp'+str(total_points)+'_round_'+str(i)+'_'
                        + experiments_date+'.jpg')
            plt.close()
