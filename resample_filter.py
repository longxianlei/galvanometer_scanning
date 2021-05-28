import random
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
"""
This part is intended to solve the sampling based re_sample method. When there are objects in the sample image, 
we should consider sampling/pay more attention in the neighborhood of this sample. 
"""

total_initial_samples = 100
# ctrl_vxs = np.round(np.random.uniform(low=-5.0, high=5.0, size=total_samples), 4)
sample_center = np.array([2.34, 1.67])
sample_radius = np.array([3])
all_detections_objs_toy = {'centre': [[2.3, 1.45], [0.13, -2.1], [-3.33, 3.05]], 'confidence': [0.85, 0.24, 0.51]}

decay_rate = 0.9  # To reduce the total samples, as we run more steps, the more accurate objects places are detected.
# Thus, we need to reduce the amount of total samples points. As our model coverage step by step.

remain_fraction_initial = 0.2  # This is the remaining fraction of total samples. Besides assign each detected object
# proportion new samples, we need to also pay some attention to other undetected places by assigning some samples.

conf_threshold = 0.5  # We should compare the different confidence value. Select the best value across different
# scan time, routes and other critical factors, which have impacts on the final efficiency and performance.

radius = 1.0  # The radius should have also the same comparison between efficiency and scan time.
# Also, if the radius is larger, we may using more scanning time. Besides, we should consider the radius_confidence
# curve, e.g., radius = standard_radius * (1-obj_conf). To avoid the complexity, we can delete the radius changes.

epochs = 10  # The training or sampling epochs, whether the epochs remain constant or condition on the running process.
# We can craft this setting.


def assign_samples(total_samples, all_detections_objs, remain_fraction):
    total_weights = 0
    filtering_objs = []
    filtered_conf = []
    sampling_x_axis = []
    sampling_y_axis = []
    sampling_centre_list = []
    for obj_index, obj_conf in enumerate(all_detections_objs['confidence']):
        if obj_conf > conf_threshold:
            total_weights += obj_conf
            filtered_conf.append(obj_conf)
            filtering_objs.append(obj_index)
    normalized_conf = np.array(filtered_conf)/total_weights  # normalized the weights of filtered objects.

    # Assign each filtered objects with some new sampling points, which is proportion to the weights.
    num_totals = np.round(total_samples*(1-remain_fraction))
    print('total assigned points: ', num_totals)
    assigned_nums = np.int16(np.round(num_totals*normalized_conf))
    print("each assigned points: ", assigned_nums)
    # 01, Attention. assign the filtered objects within radius of samples. With radius constraints.
    # At this time, we can consider whether the radius is dependent on the confidence or a fixed value.
    for i in range(len(filtering_objs)):
        sampling_nums = assigned_nums[i]
        print(sampling_nums)
        sampling_centre_index = filtering_objs[i]
        sampling_centre = all_detections_objs['centre'][sampling_centre_index]
        sampling_centre_list.append(sampling_centre)
        sampling_x_radius = np.round(np.random.uniform(low=-radius, high=radius, size=sampling_nums), 4)
        sampling_x_axis.append(np.clip(sampling_x_radius+sampling_centre[0], -5, 5))

        # sampling_x_axis.append(min(max(sampling_x_radius + sampling_centre[0], -5), 5))
        sampling_y_radius = np.round(np.random.uniform(low=-radius, high=radius, size=sampling_nums), 4)
        sampling_y_axis.append(np.clip(sampling_y_radius + sampling_centre[1], -5, 5))
        # sampling_y_axis.append(min(max(sampling_y_radius + sampling_centre[1], -5), 5))

    # 02, Global information complement.
    # Assign the remaining sample points to the global information without constraints.
    # sampling_x_axis.append(np.round(np.random.uniform(low=-5, high=5, size=int(total_samples-num_totals)), 4))
    # sampling_y_axis.append(np.round(np.random.uniform(low=-5, high=5, size=int(total_samples-num_totals)), 4))
    global_comp_points = int(total_samples - np.sum(assigned_nums))
    # print(sampling_centre_list)

    # print(np.vstack(sampling_centre_list))
    return np.hstack(sampling_x_axis), np.hstack(sampling_y_axis), np.vstack(sampling_centre_list), \
           filtering_objs, global_comp_points


def assign_nearby_samples():
    pass


if __name__ == '__main__':

    # Part 1. Using the designed samples here to construct the basic model.
    all_detections_objs_toy = {'centre': [[4.8, 1.45], [0.13, -2.1], [-3.33, 3.05], [-1.45, 1.22], [3.13, -1.54]],
                                      'confidence': [0.85, 0.24, 0.51, 0.49, 0.76]}

    plt.figure('resampling points')
    # 设置坐标轴的取值范围;
    plt.xlim((-5.5, 5.5))
    plt.ylim((-5.5, 5.5))
    # 设置坐标轴的label;
    plt.xlabel('X voltage')
    plt.ylabel('Y voltage')
    plt.title('The distribution of resampling points')
    # plt.title('The distribution of original points')
    # 设置x坐标轴刻度;
    plt.xticks(np.linspace(-5, 5, 11))
    plt.yticks(np.linspace(-5, 5, 11))
    # plt.plot(np.array(all_detections_objs['centre'])[:, 0], np.array(all_detections_objs['centre'])[:, 1], '*',
    #          label='original_points')
    #
    # re_sample_x, re_sample_y, filtered_objs, global_points = assign_samples(total_samples=20,
    #                                                                         all_detections_objs=all_detections_objs,
    #                                                                         remain_fraction=0.1)
    # print('global_points: ', global_points)
    # global_points_x = np.round(np.random.uniform(low=-5, high=5, size=global_points), 4)
    # global_points_y = np.round(np.random.uniform(low=-5, high=5, size=global_points), 4)
    # plt.plot(global_points_x, global_points_y, 'o', label='global_points')
    #
    # plt.plot(re_sample_x, re_sample_y, '^', label='resampling points')
    # plt.legend(loc='best', )
    # plt.show()


    # Part2, Using the random generate 100 samples to build and refine the model.
    # # Load the original data. Using the 100 samples.
    # orig_x = np.reshape(np.load('../path_optimize/google_scan_x_start.npy'), (-1, 1))
    # orig_y = np.reshape(np.load('../path_optimize/google_scan_y_start.npy'), (-1, 1))
    orig_x = np.reshape(np.load('../path_optimize/original_x.npy'), (-1, 1))
    orig_y = np.reshape(np.load('../path_optimize/original_y.npy'), (-1, 1))
    num_point = len(orig_y)
    print(num_point)
    # print(orig_x.shape)
    # print(orig_y.shape)
    # np.reshape(orig_x, (-1, 1))
    # np.reshape(orig_y, (-1, 1))
    # print(np.reshape(orig_x, (-1, 1)))
    # print(np.reshape(orig_y, (-1, 1)))

    print(orig_x)
    data_total = np.concatenate((orig_x, orig_y), axis=1)
    # print(data_total)
    above_conf = int(num_point*0.1)
    below_conf = num_point - above_conf
    generate_conf1 = np.round(np.random.uniform(low=0.4, high=1, size=above_conf), 4)
    generate_conf2 = np.round(np.random.uniform(low=0.0, high=0.5, size=below_conf), 4)
    # generate_conf = np.round(np.random.normal(loc=0.0, scale=1.0, size=num_point), 4)
    generate_conf = np.concatenate((generate_conf1, generate_conf2), axis=0)
    all_detections_objs_original = {'centre': data_total, 'confidence': generate_conf}

    # all_detections_objs1 = {'centre': [[4.8, 1.45], [0.13, -2.1], [-3.33, 3.05], [-1.45, 1.22], [3.13, -1.54]],
    #                                       'confidence': [0.85, 0.24, 0.51, 0.49, 0.76]}

    plt.plot(np.array(all_detections_objs_original['centre'])[:, 0],
             np.array(all_detections_objs_original['centre'])[:, 1],
             '*', label='original_points')
    # plt.savefig('original_distribution.jpg')
    print(all_detections_objs_original)
    re_sample_x, re_sample_y, sampling_centers, \
        filtered_objs, global_points = assign_samples(total_samples=num_point,
                                                      all_detections_objs=all_detections_objs_original,
                                                      remain_fraction=0.1)
    print('num of resamples: ', len(re_sample_x))
    print(filtered_objs)
    global_points_x = np.round(np.random.uniform(low=-5, high=5, size=global_points), 4)
    print('num of global points: ', len(global_points_x))
    global_points_y = np.round(np.random.uniform(low=-5, high=5, size=global_points), 4)
    plt.plot(global_points_x, global_points_y, 'mo', label='global_points')
    plt.plot(re_sample_x, re_sample_y, 'g^', label='resampling_points')
    plt.plot(sampling_centers[:, 0], sampling_centers[:, 1], 'r*', label='object_points')
    plt.legend(loc='best')
    # plt.savefig('based_on_resampling_points_0.jpg')
    plt.show()
    plt.close()

    # # Construct the next iteration's data.
    # new_points_x = np.reshape(np.hstack((re_sample_x, global_points_x)), (-1, 1))
    # new_points_y = np.reshape(np.hstack((re_sample_y, global_points_y)), (-1, 1))
    # data_total = np.concatenate((new_points_x, new_points_y), axis=1)
    # above_conf = int(num_point*0.1)
    # below_conf = num_point - above_conf
    # generate_conf1 = np.round(np.random.uniform(low=0.4, high=1, size=above_conf), 4)
    # generate_conf2 = np.round(np.random.uniform(low=0.0, high=0.5, size=below_conf), 4)
    # # generate_conf = np.round(np.random.normal(loc=0.0, scale=1.0, size=num_point), 4)
    # generate_conf = np.concatenate((generate_conf1, generate_conf2), axis=0)
    # np.random.shuffle(generate_conf)
    # all_detections_objs_original = {'centre': data_total, 'confidence': generate_conf}

    # Part3, combine the samples decay and epochs here to build the whole model.
    # Whether the decay is linear decrease or poly decrease. E.g., 100 -> 90 -> 80 ... or 100 -> 90 -> 72 -> 53.
    for epoch in range(epochs):
        # np.power()
        num_point = int(num_point*(np.power(decay_rate, epoch)))
        print(num_point)

        # # The old version data construct. Which is wrong in some dimensions.
        # new_points_x = np.hstack((re_sample_x, global_points_x))
        # new_points_y = np.hstack((re_sample_y, global_points_y))
        # data_total = np.vstack((new_points_x, new_points_y))
        # above_conf = int(num_point*0.1)
        # below_conf = num_point - above_conf
        # generate_conf1 = np.round(np.random.uniform(low=0.4, high=1, size=above_conf), 4)
        # generate_conf2 = np.round(np.random.uniform(low=0.0, high=0.5, size=below_conf), 4)
        # # generate_conf = np.round(np.random.normal(loc=0.0, scale=1.0, size=num_point), 4)
        # generate_conf = np.concatenate((generate_conf1, generate_conf2), axis=0)
        # np.random.shuffle(generate_conf)
        # all_detections_objs_original = {'centre': data_total, 'confidence': generate_conf}

        # The new construct data method.
        new_points_x = np.reshape(np.hstack((re_sample_x, global_points_x)), (-1, 1))
        new_points_y = np.reshape(np.hstack((re_sample_y, global_points_y)), (-1, 1))
        data_total = np.concatenate((new_points_x, new_points_y), axis=1)
        above_conf = int(num_point * 0.1)
        print('above confidence nums: ', above_conf)
        below_conf = num_point - above_conf
        print('below confidence nums: ', below_conf)

        generate_conf1 = np.round(np.random.uniform(low=0.4, high=1, size=above_conf), 4)
        generate_conf2 = np.round(np.random.uniform(low=0.0, high=0.5, size=below_conf), 4)
        # generate_conf = np.round(np.random.normal(loc=0.0, scale=1.0, size=num_point), 4)
        generate_conf = np.concatenate((generate_conf1, generate_conf2), axis=0)
        np.random.shuffle(generate_conf)
        all_detections_objs_original = {'centre': data_total, 'confidence': generate_conf}

        # print(all_detections_objs_original)
        re_sample_x, re_sample_y, sampling_centers, \
            filtered_objs, global_points = assign_samples(total_samples=num_point,
                                                          all_detections_objs=all_detections_objs_original,
                                                          remain_fraction=0.1)

        global_points_x = np.round(np.random.uniform(low=-5, high=5, size=global_points), 4)
        global_points_y = np.round(np.random.uniform(low=-5, high=5, size=global_points), 4)

        plt.figure(epoch)
        # 设置坐标轴的取值范围;
        plt.xlim((-5.5, 5.5))
        plt.ylim((-5.5, 5.5))
        # 设置坐标轴的label;
        plt.xlabel('X voltage')
        plt.ylabel('Y voltage')
        # plt.title('The distribution of original points')
        # 设置x坐标轴刻度;
        plt.xticks(np.linspace(-5, 5, 11))
        plt.yticks(np.linspace(-5, 5, 11))
        plt.plot(global_points_x, global_points_y, 'mo', label='global_points')
        plt.plot(re_sample_x, re_sample_y, 'g^', label='resampling points')
        plt.plot(sampling_centers[:, 0], sampling_centers[:, 1], 'r*', label='original')
        plt.legend(loc='best')
        plt.title('The distribution of resampling points at epoch: ' + str(epoch))
        # plt.savefig('based_on_prev_attention_epoch_' + str(epoch+1)+'.jpg')
        # plt.savefig('radius_1_draw_attention_epoch_' + str(epoch + 1) + '.jpg')
        plt.show()
