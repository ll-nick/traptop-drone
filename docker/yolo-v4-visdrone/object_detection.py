#!/usr/bin/python3
import argparse
import math
import os
import glob
import random
import time

import cv2
import numpy as np

import darknet


def pair(arg):
    xs = arg.split(',')
    if len(xs) != 2:
        raise ValueError('Number of arguments must be 2!')
    return np.array([int(x) for x in xs])


def parser():
    parser = argparse.ArgumentParser(description='YOLO Object Detection')
    parser.add_argument('config_file', type=str, help='Path to config file')
    parser.add_argument('data_file', type=str, help='Path to data file')
    parser.add_argument('weights', type=str, help='Path to weights file')
    parser.add_argument(
        'input', type=str, help='Image source. Can be a single image, a txt with paths to them, or a folder. Valid formats are jpg, jpeg or png.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='number of images to be processed at the same time')
    parser.add_argument('--thresh', type=float, default=0.4,
                        help='remove detections with lower confidence')
    parser.add_argument('--take', type=int, default=None,
                        help='Number of samples to process')
    parser.add_argument('--save_to', type=str,
                        default=None, help='Export path')
    parser.add_argument('--save_labels', action='store_true',
                        help='Save detections for each image. Requires --save_to arg to be set.')
    parser.add_argument('--save_image', action='store_true',
                        help='Save image with detections. Requires --save_to arg to be set.')
    parser.add_argument('--print_detections',
                        action='store_true', help='Print list of detections')
    parser.add_argument("--crop_size", type=pair, default=None,
                        help="Set an optional crop size.")
    return parser.parse_args()


def check_args(args):
    assert 0 < args.thresh < 1, 'Threshold should be within [0, 1)'
    if not os.path.exists(args.config_file):
        raise ValueError(
            f'Invalid config path {os.path.abspath(args.config_file)}.')
    if not os.path.exists(args.weights):
        raise ValueError(
            f'Invalid weight path {os.path.abspath(args.weights)}.')
    if not os.path.exists(args.data_file):
        raise ValueError(
            f'Invalid data file path {os.path.abspath(args.data_file)}.')
    if args.input and not os.path.exists(args.input):
        raise ValueError(f'Invalid image path {os.path.abspath(args.input)}.')
    if args.save_image or args.save_labels:
        assert args.save_to is not None


def draw_boxes(detections, image, colors):
    import cv2
    for label, confidence, bbox in detections:
        left, top, right, bottom = darknet.bbox2points(bbox)
        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 2)
        cv2.putText(image, f'{label} ({round(100 * float(confidence))}%)',
                    (left, top - 5), cv2.FONT_HERSHEY_PLAIN, 1, colors[label], 2)
    return image


def transform_detections(detections, crop):

    def transform_detection(detection, dx, dy, dw, dh):
        label, confidence, bbox = detection
        x, y, w, h = bbox
        x += dx
        y += dy
        w += dw
        h += dh
        bbox = x, y, w, h
        return label, confidence, bbox

    detections_transformed = []
    for detection in detections:
        detections_transformed.append(
            transform_detection(detection, crop[1], crop[0], 0, 0))
    return detections_transformed


def scale_detections(detections, scale):

    def scale_detection(detection, scale):
        label, confidence, bbox = detection
        x, y, w, h = bbox
        x *= scale[0]
        y *= scale[1]
        w *= scale[0]
        h *= scale[1]
        bbox = x, y, w, h
        return label, confidence, bbox

    detections_scaled = []
    for detection in detections:
        detections_scaled.append(scale_detection(detection, scale))
    return detections_scaled


def nms(detections, score_threshold=0.25, nms_threshold=0.25):
    bboxes = []
    scores = []
    for detection in detections:
        label, confidence, bbox = detection
        x, y, w, h = bbox
        bboxes.append(list(bbox))
        scores.append(confidence)
    indices = cv2.dnn.NMSBoxes(
        bboxes=bboxes, scores=scores, score_threshold=score_threshold, nms_threshold=nms_threshold)
    indices = np.array(indices).reshape(-1)
    if len(indices):
        detections = [detections[idx.astype(int)] for idx in indices]
    return detections


def filter_border_detections(detections, crop, crop_size, image_size):

    detections_filtered = []
    for detection in detections:
        _, _, bbox = detection
        x, y, w, h = bbox

        xmin = x == 0 and crop[1] != 0
        ymin = y == 0 and crop[0] != 0
        xmax = x == crop_size[1] and x + crop_size[1] != image_size[1]
        ymax = y == crop_size[0] and y + crop_size[0] != image_size[0]

        if not (xmin or ymin or xmax or ymax):
            detections_filtered.append(detection)

    return detections_filtered


def check_shapes(images, batch_size):
    shapes = [image.shape for image in images]
    if len(set(shapes)) > 1:
        raise ValueError("Images don't have same shape")
    if len(shapes) > batch_size:
        raise ValueError('Batch size higher than number of images')
    return shapes[0]


def load_images(images_path, take):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    filenames = []
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        filenames.append(images_path)
    elif input_path_extension == 'txt':
        with open(images_path, 'r') as f:
            filenames.extend(f.read().splitlines())
    else:
        filenames.extend(glob.glob(os.path.join(images_path, '*.jpg')))
        filenames.extend(glob.glob(os.path.join(images_path, '*.png')))
        filenames.extend(glob.glob(os.path.join(images_path, '*.jpeg')))
    filenames = sorted(filenames)
    if take:
        filenames = filenames[:take]

    return filenames


def prepare_batch(network, images, channels=3):
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    batch_size = len(images)
    darknet_image = darknet.make_image(width, height, channels * batch_size)
    images_resized = []
    for image in images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        images_resized.append(image_resized)
    image_resized = np.concatenate(images_resized, axis=-1)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    return darknet_image


def batch_detection(network, images, class_names, class_colors,
                    thresh=0.25, hier_thresh=.5, nms=.45, batch_size=4):
    height, width, _ = check_shapes(images, batch_size)
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_images = prepare_batch(network, images)
    batch_detections = darknet.network_predict_batch(network, darknet_images, batch_size, width,
                                                     height, thresh, hier_thresh, None, 0, 0)
    darknet.free_image(darknet_images)

    scale = [images[0].shape[1] / width, images[0].shape[0] / height]
    batch_predictions = []
    for idx in range(batch_size):
        num = batch_detections[idx].num
        detections = batch_detections[idx].dets
        if nms:
            darknet.do_nms_sort(detections, num, len(class_names), nms)
        detections = darknet.remove_negatives(detections, class_names, num)
        detections = scale_detections(detections, scale)
        batch_predictions.append(detections)
    darknet.free_batch_detections(batch_detections, batch_size)
    return batch_predictions


def convert_abs2rel(image, bbox):
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height


def save_labels(fn, image, detections, class_names):
    with open(fn, 'w') as f:
        for label, confidence, bbox in detections:
            x, y, w, h = convert_abs2rel(image, bbox)
            label = class_names.index(label)
            f.write(
                f'{label} {x:.4f} {y:.4f} {w:.4f} {h:.4f} {float(confidence):.4f}\n')


def print_detections(detections):
    histogram = dict((label, 0) for label, _, _ in detections)
    for label, confidence, bbox in detections:
        histogram[label] += 1

    logmsg = f'Got {len(detections)} detections: '
    for key, val in histogram.items():
        logmsg += f'{val} {key}, '

    logmsg = logmsg[:-1]
    print(logmsg[:-1])


def create_batches(inputs, batch_size):
    outputs = []
    for idx in range(0, len(inputs), batch_size):
        outputs.append(inputs[idx:idx+batch_size])
    last_batch = outputs[-1]
    last_input = last_batch[-1]

    # Replicate last element to fill up last batch
    for _ in range(batch_size - len(last_batch)):
        last_batch.append(last_input)
    outputs[-1] = last_batch
    return outputs


def split_image(image_size, crop_size, min_overlap_ratio=0.1):
    min_overlap = min_overlap_ratio * crop_size

    def divide_single_dim(total_length, crop_length, min_overlap):
        num_crops = math.ceil((total_length - min_overlap) /
                              (crop_length - min_overlap))
        overlap = (num_crops * crop_length - total_length) / (num_crops - 1)
        indices = []
        for crop_idx in range(num_crops):
            rounded_overlap_sum = round(crop_idx * overlap)
            start = max(0, crop_idx * crop_length - rounded_overlap_sum)
            end = min(total_length, start + crop_length)
            end = start + crop_length
            indices.append(start)
        return indices

    indices_x = divide_single_dim(image_size[0], crop_size[0], min_overlap[0])
    indices_y = divide_single_dim(image_size[1], crop_size[1], min_overlap[1])

    crops = []
    for idx_x in indices_x:
        for idx_y in indices_y:
            crops.append([idx_x, idx_y])
    return crops


def detect(args):
    if args.save_to and (args.save_image or args.save_labels):
        print(f'Saving output to {os.path.abspath(args.save_to)}')
        os.makedirs(args.save_to, exist_ok=True)

    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=args.batch_size
    )

    image_names = load_images(args.input, args.take)

    for count, image_name in enumerate(image_names):
        t_start = time.time()
        image = cv2.imread(image_name)
        image_size = np.array(image.shape[:2])
        crops = split_image(image_size=image_size, crop_size=args.crop_size)

        cropped_images = []
        for crop in crops:
            cropped_images.append(
                image[crop[0]:crop[0]+args.crop_size[0], crop[1]:crop[1]+args.crop_size[1]])

        batch_images_list = create_batches(cropped_images, args.batch_size)

        batch_detections = []
        for batch_images in batch_images_list:
            batch_detections.extend(batch_detection(network, batch_images, class_names,
                                                    class_colors, thresh=args.thresh, batch_size=args.batch_size))

        all_detections = []
        for detections, crop in zip(batch_detections, crops):
            detections = filter_border_detections(
                detections, crop, args.crop_size, image_size)
            all_detections.extend(transform_detections(detections, crop))
        detections = all_detections
        detections = nms(detections)

        image = draw_boxes(detections, image, class_colors)
        path_out = os.path.join(args.save_to,
                                os.path.basename(image_name)).split('.')[:-1][0]
        if args.save_image:
            cv2.imwrite(filename=path_out + '.png', img=image)
        if args.print_detections:
            print_detections(detections)
        if args.save_labels:
            save_labels(path_out + '.txt', image, detections, class_names)
        t_end = time.time()
        print(
            f'{count+1} / {len(image_names)} @ {int(1000 * (t_end - t_start))} ms')


if __name__ == '__main__':
    args = parser()
    check_args(args)
    random.seed(1)  # custom colors
    detect(args)
