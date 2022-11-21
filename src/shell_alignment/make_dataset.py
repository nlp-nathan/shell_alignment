import argparse
import numpy as np
import os
import random
import shutil
import tensorflow as tf

from utils import get_keypoints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str)
    parser.add_argument("--keypoint_fp", type=str)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--val_unseen", type=int, default=2)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--test_unseen", type=int, default=1)
    parser.add_argument("--test_size", type=float, default=0.1)
    args = parser.parse_args()

    n_val_unseen = args.val_unseen
    n_test_unseen = args.test_unseen
    img_dir = args.img_dir
    xml_path = args.keypoint_fp
    output_dir = args.output_dir

    # Randomly select n number of unseen for val and test.
    turtle_dirs = np.array([os.path.join(img_dir, file) for file in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, file))])
    random_image_idx = random.sample(range(len(turtle_dirs)), k=n_val_unseen+n_test_unseen)

    val = turtle_dirs[random_image_idx][:n_val_unseen]
    test = turtle_dirs[random_image_idx][n_val_unseen:]
    turtle_dirs = np.delete(turtle_dirs, random_image_idx)
    print("val unseen individuals:", val)
    print("test unseen individuals:", test)

    image_path_dict = {"train": [], "val": [], "test": [],}
    data_dirs = {"train": turtle_dirs, "val": val, "test": test}

    for k, v in data_dirs.items():
        for turtle_dir in v.tolist():
            for base, directory, files in os.walk(turtle_dir):
                for file in files:
                    if os.path.splitext(file)[1] == '.jpg':
                        image_path_dict[k].append(os.path.join(base, file))

    image_path_dict['train'] = np.array(image_path_dict['train'])

    n_total_image_paths = len(image_path_dict['val']) + len(image_path_dict['test']) + len(image_path_dict['train'])
    n_more_val_images = int(n_total_image_paths*args.val_size)-len(image_path_dict['val'])
    n_more_test_images = int(n_total_image_paths*args.test_size)-len(image_path_dict['test'])

    if n_more_val_images > 0:
        random_image_idx = random.sample(range(len(image_path_dict['train'])), k=n_more_val_images)
        val = np.append(val, image_path_dict['train'][random_image_idx])
        image_path_dict['train'] = np.delete(image_path_dict['train'], random_image_idx)
    
    if n_more_test_images > 0:
        random_image_idx = random.sample(range(len(image_path_dict['train'])), k=n_more_test_images)
        test = np.append(test, image_path_dict['train'][random_image_idx])
        image_path_dict['train'] = np.delete(image_path_dict['train'], random_image_idx)

    data_dirs = {"train": image_path_dict['train'], "val": val, "test": test,}
    for k, v in data_dirs.items():
        for item in v.tolist():
            dir_name = f"{os.path.sep}".join(splitall(item)[1:])
            if os.path.isdir(item):
                shutil.copytree(item, os.path.join(output_dir, "raw", k, dir_name))
            else:
                base_dir = os.path.join(output_dir, "raw", k, "".join(os.path.split(dir_name)[0]))
                if not os.path.exists(base_dir):
                    os.makedirs(base_dir)
                shutil.copy2(item, os.path.join(output_dir, "raw", k, dir_name))

    # Create TFRecords
    train_dir = os.path.join(output_dir, "raw", 'train')
    val_dir = os.path.join(output_dir, "raw", 'val')
    test_dir = os.path.join(output_dir, "raw", 'test')

    # Get image names and shell keypoints.
    image_paths = {
        train_dir: [],
        val_dir: [],
        test_dir: [],
    }

    keypoints = {
        train_dir: [],
        val_dir: [],
        test_dir: [],
    }

    for img_dir, v in image_paths.items():
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                if os.path.splitext(file)[1] == '.jpg':
                    image_paths[img_dir].append(os.path.join(root, file))
                    keypoints[img_dir].append(get_keypoints(xml_path, file))

    print()
    print(f'length of training images: {len(image_paths[train_dir])}')
    print(f'length of trainging keypoints: {len(keypoints[train_dir])}')
    print()
    print(f'length of training images: {len(image_paths[val_dir])}')
    print(f'length of trainging keypoints: {len(keypoints[val_dir])}')
    print()

    print(f'length of training images: {len(image_paths[test_dir])}')
    print(f'length of trainging keypoints: {len(keypoints[test_dir])}')

    train_flat_keypoints = [list(np.concatenate([corners, centers]).flatten()) for corners, centers in keypoints[train_dir]]
    val_flat_keypoints = [list(np.concatenate([corners, centers]).flatten()) for corners, centers in keypoints[val_dir]]
    test_flat_keypoints = [list(np.concatenate([corners, centers]).flatten()) for corners, centers in keypoints[test_dir]]

    train_tfdatasets_dir = (os.path.join(output_dir, 'records', 'train'))
    val_tfdatasets_dir = (os.path.join(output_dir, 'records', 'val'))
    test_tfdatasets_dir = (os.path.join(output_dir, 'records', 'test'))

    for tfdatasets_dir in [train_tfdatasets_dir, val_tfdatasets_dir, test_tfdatasets_dir]:
        if not os.path.exists(tfdatasets_dir):
            os.makedirs(tfdatasets_dir)

    num_samples = 44
    create_records(image_paths[train_dir], train_flat_keypoints, train_tfdatasets_dir, num_samples)
    create_records(image_paths[val_dir], val_flat_keypoints, val_tfdatasets_dir, num_samples)
    create_records(image_paths[test_dir], test_flat_keypoints, test_tfdatasets_dir, num_samples)

def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

def create_example(image, path, keypoints):
    feature = {
        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(image).numpy()])),
        "path": tf.train.Feature(bytes_list=tf.train.BytesList(value=[path.encode()])),
        "keypoints": tf.train.Feature(float_list=tf.train.FloatList(value=keypoints))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def create_records(image_paths, flat_keypoints, tfdatasets_dir, num_samples):
    num_tfrecords = -1* (-len(image_paths) // num_samples)
    records = []
    for tfrec_num in range(num_tfrecords):
        samples = image_paths[tfrec_num*num_samples : num_samples+num_samples*tfrec_num]
        targets = flat_keypoints[tfrec_num*num_samples : num_samples+num_samples*tfrec_num]

        file_name = tfdatasets_dir + "/file_%.2i-%i.tfrec" % (tfrec_num, len(samples))
        records.append(file_name)
        with tf.io.TFRecordWriter(file_name) as writer:
            for sample, target in zip(samples, targets):
                image_path = sample
                image = tf.io.decode_jpeg(tf.io.read_file(image_path))
                example = create_example(image, image_path, target)
                writer.write(example.SerializeToString())
                
    return records

if __name__ == "__main__":
    main()