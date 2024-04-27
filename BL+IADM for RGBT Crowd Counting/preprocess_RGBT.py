import cv2
import numpy as np
import os
from glob import glob
import xml.etree.ElementTree as ET

def generate_data(label_path, root_path):
    rgb_path = label_path.replace('GT', 'RGB').replace('R.xml', '.jpg').replace('labels', 'rgb')
    t_path = label_path.replace('GT', 'T').replace('xml', 'jpg').replace('labels', 'tir')
    rgb = cv2.imread(os.path.join(root_path, rgb_path))
    t = cv2.imread(os.path.join(root_path, t_path))
    im_h, im_w, _ = rgb.shape

    tree = ET.parse(os.path.join(root_path, label_path))
    root = tree.getroot()

    points = []
    for object in root.findall('object'):
        point = object.find('point')
        if point is not None:
            x = int(point.find('x').text)
            y = int(point.find('y').text)
            points.append([x, y])

    points = np.asarray(points)
    idx_mask = (points[:, 0] >= 0) & (points[:, 0] < im_w) & (points[:, 1] >= 0) & (points[:, 1] < im_h)
    points = points[idx_mask]
    return rgb, t, points

def process_and_save(data_list, phase, root_path, save_dir):
    sub_save_dir = os.path.join(save_dir, phase)
    if not os.path.exists(sub_save_dir):
        os.makedirs(sub_save_dir)
        os.makedirs(os.path.join(sub_save_dir, 'rgb'))
        os.makedirs(os.path.join(sub_save_dir, 'tir'))
        os.makedirs(os.path.join(sub_save_dir, 'labels'))

    for file_name in data_list:
        gt_path = os.path.join('labels', file_name + 'R.xml')
        rgb, t, points = generate_data(gt_path, root_path)

        rgb_save_path = os.path.join(sub_save_dir, 'rgb', file_name + '.jpg')
        t_save_path = os.path.join(sub_save_dir, 'tir', file_name + 'R.jpg')
        gd_save_path = os.path.join(sub_save_dir, 'labels', file_name + 'R.npy')

        cv2.imwrite(rgb_save_path, rgb)
        cv2.imwrite(t_save_path, t)
        np.save(gd_save_path, points)
        print(file_name)

if __name__ == '__main__':
    root_path = 'RGBT_game/train'  # dataset root path
    save_dir = 'dataset'  # new base directory for train and val splits

    # Read data from train.txt and val.txt
    with open(os.path.join(root_path, 'train.txt'), 'r') as file:
        train_data = [line.strip() for line in file]
    with open(os.path.join(root_path, 'val.txt'), 'r') as file:
        val_data = [line.strip() for line in file]

    # Process and save train data
    process_and_save(train_data, 'train', root_path, save_dir)

    # Process and save val data
    process_and_save(val_data, 'val', root_path, save_dir)



# import numpy as np
# import os
# from glob import glob
# import cv2
# import json
#
#
# def generate_data(label_path):
#     rgb_path = label_path.replace('GT', 'RGB').replace('json', 'jpg')
#     t_path = label_path.replace('GT', 'T').replace('json', 'jpg')
#     rgb = cv2.imread(rgb_path)[..., ::-1].copy()
#     t = cv2.imread(t_path)[..., ::-1].copy()
#     im_h, im_w, _ = rgb.shape
#     print('rgb and t shape', rgb.shape, t.shape)
#     with open(label_path, 'r') as f:
#         label_file = json.load(f)
#     points = np.asarray(label_file['points'])
#     # print('points', points.shape)
#     idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
#     points = points[idx_mask]
#     return rgb, t, points
#
#
# if __name__ == '__main__':
#
#     root_path = '/data2/cjq/RGBT-CC-V2'  # dataset root path
#     save_dir = '/data2/cjq/bayes-RGBT-CC-V2'
#     # save_dir = '/data2/cjq/RGBT-test'
#
#     for phase in ['train', 'val', 'test']:
#         sub_dir = os.path.join(root_path, phase)
#         sub_save_dir = os.path.join(save_dir, phase)
#         if not os.path.exists(sub_save_dir):
#             os.makedirs(sub_save_dir)
#         gt_list = glob(os.path.join(sub_dir, '*json'))
#         for gt_path in gt_list:
#             name = os.path.basename(gt_path)
#             # print('name', name)
#             rgb, t, points = generate_data(gt_path)
#             im_save_path = os.path.join(sub_save_dir, name)
#             rgb_save_path = im_save_path.replace('GT', 'RGB').replace('json', 'jpg')
#             t_save_path = im_save_path.replace('GT', 'T').replace('json', 'jpg')
#             cv2.imwrite(rgb_save_path, rgb)
#             cv2.imwrite(t_save_path, t)
#             gd_save_path = im_save_path.replace('json', 'npy')
#             np.save(gd_save_path, points)
