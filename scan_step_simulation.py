import cv2
import numpy as np

np.random.seed(10)


def compute_iou(object_box1, scan_box2):
    """
    compute the maximum IOU given the query object box.
    :param object_box1: query object box. [center_x, center_y, w, h]
    :param scan_box2: scanning box list. [N, 4]
    :return:
    """
    max_area = 0
    max_index = 0
    # print(len(scan_box2))
    for i in range(len(scan_box2)):
        rect1 = ((object_box1[0], object_box1[1]), (object_box1[2], object_box1[3]), 0)
        rect2 = ((scan_box2[i][0], scan_box2[i][1]), (scan_box2[i][2], scan_box2[i][3]), 0)
        inter_section = cv2.rotatedRectangleIntersection(rect1, rect2)
        if inter_section[0] != 0:
            inter_area = cv2.contourArea(inter_section[1])
            max_index = i if inter_area > max_area else max_index
            max_area = inter_area if inter_area > max_area else max_area
    # print(max_area, max_index)
    return max_area, max_index


def plot_scan_image(scan_image, step_type, is_save):
    """
    Simulation the data distribution, plot the scanning FOV and the step for every scan.
    :param scan_image:original white image.
    :param step_type:0 is non-overlap, 1 is full step, 2 is half the object.
    :param is_save: whether to save the plot image.
    :return:
    """
    white_img = scan_image
    step_len_h = scanning_height - int(step_height / step_type) if step_type > 0 else scanning_height
    step_len_w = scanning_width - int(step_width / step_type) if step_type > 0 else scanning_width
    """ plot the scanning FOV over the panorama image."""
    scan_count = 0
    scan_boxes = []

    for i in range(0, panorama_height, step_len_h):
        for j in range(0, panorama_height, step_len_w):
            # print(i)
            # temp_box = []
            scan_count += 1
            if (i / step_len_h) % 2 == 0:
                if (j / step_len_w) % 2 == 0:
                    # cv2.rectangle(white_img, (j, i), (j + scanning_width -5, i + scanning_height - 5), (255, 51, 255),
                    #               thickness=2)
                    cv2.drawMarker(white_img, (j + int(scanning_width / 2), i + int(scanning_height / 2)),
                                   (255, 51, 255),
                                   cv2.MARKER_TILTED_CROSS, thickness=2)
                else:
                    # cv2.rectangle(white_img, (j, i), (j + scanning_width -5, i + scanning_height - 5), (255, 51, 133),
                    #               thickness=2)
                    cv2.drawMarker(white_img, (j + int(scanning_width / 2), i + int(scanning_height / 2)),
                                   (255, 51, 133),
                                   cv2.MARKER_TILTED_CROSS, thickness=2)
            else:
                if (j / step_len_w) % 2 == 0:
                    # cv2.rectangle(white_img, (j, i), (j + scanning_width -5, i + scanning_height - 5), (26, 102, 255),
                    #               thickness=2)
                    cv2.drawMarker(white_img, (j + int(scanning_width / 2), i + int(scanning_height / 2)),
                                   (26, 102, 255),
                                   cv2.MARKER_TILTED_CROSS, thickness=2)
                else:
                    # cv2.rectangle(white_img, (j, i), (j + scanning_width - 5, i + scanning_height - 5), (204, 204, 0),
                    #               thickness=2)
                    cv2.drawMarker(white_img, (j + int(scanning_width / 2), i + int(scanning_height / 2)),
                                   (204, 204, 0),
                                   cv2.MARKER_TILTED_CROSS, thickness=2)
            scan_boxes.append([j + scanning_width / 2, i + scanning_height / 2, scanning_width, scanning_width])

    # print("scan boxes: ", scan_boxes)
    # print("num of scan boxes: ", len(scan_boxes))

    """ generate random samples position."""
    random_samples_pos = np.random.randint(int(max(object_width, object_height) / 2),
                                           panorama_width - int(max(object_width, object_height) / 2),
                                           size=(num_samples, 2))

    for i in range(len(random_samples_pos)):
        object_box = [random_samples_pos[i][0], random_samples_pos[i][1], object_width, object_height]
        cv2.circle(white_img, (random_samples_pos[i][0], random_samples_pos[i][1]),
                   radius=int(object_height / 2), color=(255, 51, 199), thickness=2)
        # compute the maximum IOU of the corresponding scanning window.
        max_area, belong_index = compute_iou(object_box, scan_boxes)
        print(belong_index)
        cv2.rectangle(white_img,
                      (int(scan_boxes[belong_index][0]) - int(scanning_width / 2),
                       int(scan_boxes[belong_index][1]) - int(scanning_height / 2)),
                      (int(scan_boxes[belong_index][0]) + int(scanning_width / 2),
                       int(scan_boxes[belong_index][1]) + int(scanning_height / 2)),
                      color=(0, 204, 0), thickness=3)

    # print(random_samples_pos)
    # print("total scanning : ", scan_count)
    cv2.imshow("plot image", white_img)
    if is_save:
        cv2.imwrite("scanning_img/" + str(step_type) + "_width_step_" + str(scan_count) + "new.jpg", white_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    scanning_width = 220
    scanning_height = 220
    object_width = 80
    object_height = 80
    panorama_width = 1080
    panorama_height = 1080
    num_samples = 40
    step_height = object_width
    step_width = object_height
    original_img = np.ones((panorama_width, panorama_height, 3), np.uint8)  # # b, g, r
    original_img = 255 * original_img
    isSave = False
    plot_scan_image(original_img, step_type=2, is_save=isSave)
