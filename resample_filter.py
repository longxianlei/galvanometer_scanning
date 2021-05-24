import random
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
"""
This part is intended to solve the sampling based re_sample method. When there are objects in the sample image, 
we should consider sampling/pay more attention in the neighborhood of this sample. 
"""

total_samples = 100
# ctrl_vxs = np.round(np.random.uniform(low=-5.0, high=5.0, size=total_samples), 4)
sample_center = np.array([2.34, 1.67])
sample_radius = np.array([3])
all_detections_objs = {'centre': [[2.3, 1.45], [0.13, -2.1], [-3.33, 3.05]], 'confidence':[0.85, 0.24, 0.51]}

decay_rate = 0.9  # To reduce the total samples, as we run more steps, the more accurate objects places are detected.
# Thus, we need to reduce the amount of total samples points. As our model coverage step by step.

remain_fraction = 0.1  # This is the remaining fraction of total samples. Besides assign each detected object
# proportion new samples, we need to also pay some attention to other undetected places by assigning some samples.

conf_threshold = 0.5  # We should compare the different confidence value. Select the best value across different
# scan time, routes and other critical factors, which have impacts on the final efficiency and performance.

radius = 1.0  # The radius should have also the same comparison between efficiency and scan time.
# Also, if the radius is larger, we may using more scanning time. Besides, we should consider the radius_confidence
# curve, e.g., radius = standard_radius * (1-obj_conf). To avoid the complexity, we can delete the radius changes.


def assign_samples(total_samples, all_detections_objs, remain_fraction):
    total_weights = 0
    filtered_objs = []
    filtered_conf = []
    sampling_x_axis = []
    sampling_y_axis = []
    for obj_index, obj_conf in enumerate(all_detections_objs['confidence']):
        if obj_conf > conf_threshold:
            total_weights += obj_conf
            filtered_conf.append(obj_conf)
            filtered_objs.append(obj_index)
    normalized_conf = np.array(filtered_conf)/total_weights  # normalized the weights of filtered objects.

    # Assign each filtered objects with some new sampling points, which is proportion to the weights.
    num_totals = np.round(total_samples*(1-remain_fraction))
    print('total assigned points: ', num_totals)
    assigned_nums = np.int16(np.round(num_totals*normalized_conf))
    print("each assigned points: ", assigned_nums)
    # 01, Attention. assign the filtered objects within radius of samples. With radius constraints.
    # At this time, we can consider whether the radius is dependent on the confidence or a fixed value.
    for i in range(len(filtered_objs)):
        sampling_nums = assigned_nums[i]
        print(sampling_nums)
        sampling_centre_index = filtered_objs[i]
        sampling_centre = all_detections_objs['centre'][sampling_centre_index]
        sampling_x_radius = np.round(np.random.uniform(low=-radius, high=radius, size=sampling_nums), 4)

        # print(sampling_x_radius)
        # print(sampling_centre)
        # print(max((sampling_x_radius+sampling_centre[0]).all(),  5))
        #
        # batch_x_axis = sampling_x_radius + sampling_centre[0]

        # batch_clip = np.clip(sampling_x_radius + sampling_centre[0], -5, 5)
        # print(batch_clip)

        # print(batch_x_axis)
        # batch_x_axis[batch_x_axis > 5] = 5
        # batch_x_axis[batch_x_axis < -5] = -5
        # print(batch_x_axis)
        # print((sampling_x_radius+sampling_centre[0])[(sampling_x_radius+sampling_centre[0])>5]= 5))

        sampling_x_axis.append(np.clip(sampling_x_radius+sampling_centre[0], -5, 5))

        # sampling_x_axis.append(min(max(sampling_x_radius + sampling_centre[0], -5), 5))
        sampling_y_radius = np.round(np.random.uniform(low=-radius, high=radius, size=sampling_nums), 4)
        sampling_y_axis.append(np.clip(sampling_y_radius + sampling_centre[1], -5, 5))
        # sampling_y_axis.append(min(max(sampling_y_radius + sampling_centre[1], -5), 5))

    # 02, Global information complement.
    # Assign the remaining sample points to the global information without constraints.
    # sampling_x_axis.append(np.round(np.random.uniform(low=-5, high=5, size=int(total_samples-num_totals)), 4))
    # sampling_y_axis.append(np.round(np.random.uniform(low=-5, high=5, size=int(total_samples-num_totals)), 4))
    global_points = int(total_samples-np.sum(assigned_nums))
    return np.hstack(sampling_x_axis), np.hstack(sampling_y_axis), filtered_objs, global_points


def assign_nearby_samples():
    pass


# samples = np.array([3, 12, 6])
# total_weights = np.sum(samples)
# normalized_data = samples/total_weights
# new_samples = np.round(100*normalized_data)
# print(new_samples)

all_detections_objs = {'centre': [[4.8, 1.45], [0.13, -2.1], [-3.33, 3.05], [-1.45, 1.22], [3.13, -1.54]],
                                  'confidence': [0.85, 0.24, 0.51, 0.49, 0.76]}
# print(all_detections_objs['centre'][0][0])
# print(all_detections_objs['confidence'][0])


plt.figure('resampling points')
# 设置坐标轴的取值范围;
plt.xlim((-5.5, 5.5))
plt.ylim((-5.5, 5.5))
# 设置坐标轴的label;
plt.xlabel('X voltage')
plt.ylabel('Y voltage')
plt.title('The distribution of resampling points')
# 设置x坐标轴刻度;
plt.xticks(np.linspace(-5, 5, 11))
plt.yticks(np.linspace(-5, 5, 11))
plt.plot(np.array(all_detections_objs['centre'])[:, 0], np.array(all_detections_objs['centre'])[:, 1], '*',
         label='original_points')


# print(np.array(all_detections_objs['centre'])[:, 0])

re_sample_x, re_sample_y, filtered_objs, global_points = assign_samples(total_samples=20,
                                                                        all_detections_objs=all_detections_objs,
                                                                        remain_fraction=0.1)
print('global_points: ', global_points)
global_points_x = np.round(np.random.uniform(low=-5, high=5, size=global_points), 4)
global_points_y = np.round(np.random.uniform(low=-5, high=5, size=global_points), 4)
plt.plot(global_points_x, global_points_y, 'o', label='global_points')

plt.plot(re_sample_x, re_sample_y, '^', label='resampling points')
plt.legend(loc='best', )
plt.show()

