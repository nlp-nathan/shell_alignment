import cv2
import numpy as np
import tensorflow as tf
import xmltodict


TEMPLATE = np.float32([
    (0.725, 0.05), (0.95, 0.20),
    (0.725, 0.35), (0.95, 0.50),
    (0.725, 0.65), (0.95, 0.80),
    (0.725, 0.95), (0.275, 0.95),
    (0.05, 0.80),  (0.275, 0.65),
    (0.05, 0.50),  (0.275, 0.35),  
    (0.05, 0.20), (0.275, 0.05),
    (0.50, 0.20),
    (0.50, 0.50),
    (0.50, 0.80)])

def get_keypoints(xml_path, image_name):
    """
    Parses keypoints from xml for a specific image.

    Arguments:
        xml_path: path to the xml created by CVAT.
        image_name: name of the image annotated by CVAT.

    Returns:
        tuple: ([corners that make the scutes perimeter], [the centers of the three scutes]).
    """

    with open(xml_path, "r") as f:
        xml_dict = xmltodict.parse(f.read())
        centers = []
        corners = []
        for img_annotation in xml_dict["annotations"]["image"]:
            if image_name == img_annotation["@name"]:
                if len(img_annotation["points"]) > 1:
                    for label in img_annotation["points"]:
                        if label["@label"] == "centers":
                            for point in label["@points"].split(";"):
                                centers.append(
                                    tuple(
                                        round(float(coord))
                                        for coord in point.split(",")
                                    )
                                )
                        if label["@label"] == "corners":
                            for point in label["@points"].split(";"):
                                corners.append(
                                    tuple(
                                        round(float(coord))
                                        for coord in point.split(",")
                                    )
                                )

    return corners, centers


def align_scutes(image, corners, centers, TEMPLATE, size=(500, 500)):
    """
    Parses keypoints from xml for a specific image.

    Arguments:
        image: np.array of image with scutes to be aligned.
        corners: list of tuples of x, y coordinates of the keypoints
            that make up the perimeter of the three central scutes.
        centers: list of tuples of x, y coordinates of the center
            keypoint of each of the three central scutes.
        TEMPLATE: the template the scutes will be warped to.
        size: the dimensions in pixels of the output image.

    Returns:
        np.array: Image of aligned center scutes.
    """

    # Create the template.
    width = size[0]
    height = size[1]
    template = np.zeros((height, width, 3))

    # Keypoint indexes that make up a triangle.
    scute1 = [0, 1, 2, 11, 12, 13]
    scute2 = [2, 3, 4, 9, 10, 11]
    scute3 = [4, 5, 6, 7, 8, 9]
    scute_idxs = [scute1, scute2, scute3]

    # Create triangles for both original and template.
    img_triangles = []
    temp_triangles = []
    for i in range(len(scute_idxs)):
        left = -1
        for corner_idx in scute_idxs[i]:
            x1, y1 = corners[scute_idxs[i][left]]
            x2, y2 = corners[corner_idx]
            x3, y3 = centers[i]
            img_triangles.append([x1, y1, x2, y2, x3, y3])

            x1, y1 = TEMPLATE[scute_idxs[i][left]]
            x2, y2 = TEMPLATE[corner_idx]
            x3, y3 = TEMPLATE[-3 + i]
            temp_triangles.append([x1, y1, x2, y2, x3, y3])

            left += 1

    aligned = np.zeros(template.shape, np.uint8)
    image_copy = image.copy()

    temp_triangles = np.array(temp_triangles)
    temp_triangles[:, [1, 3, 5]] *= height
    temp_triangles[:, [0, 2, 4]] *= width
    scaled_triangles = temp_triangles.astype(np.int32)

    for img_triangle, scaled_triangle in zip(img_triangles, scaled_triangles):
        # Get the points for the original triangle.
        tr1_pt1 = img_triangle[0], img_triangle[1]
        tr1_pt2 = img_triangle[2], img_triangle[3]
        tr1_pt3 = img_triangle[4], img_triangle[5]
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

        # Get a bouding box around the original traingle.
        rect1 = cv2.boundingRect(triangle1)
        (x, y, w, h) = rect1
        cropped_triangle1 = image_copy[y : y + h, x : x + w]

        # Fill the outside with black and the inside with white.
        cropped_tr1_mask = np.zeros((h, w), np.int8)
        points1 = np.array(
            [
                [tr1_pt1[0] - x, tr1_pt1[1] - y],
                [tr1_pt2[0] - x, tr1_pt2[1] - y],
                [tr1_pt3[0] - x, tr1_pt3[1] - y],
            ]
        )
        cv2.fillConvexPoly(cropped_tr1_mask, points1, 255)

        # Use the mask to cut around the original triangle in its bounding box.
        cropped_triangle1 = cv2.bitwise_and(
            cropped_triangle1, cropped_triangle1, mask=cropped_tr1_mask
        )

        # Get the points for the template triangle.
        tr2_pt1 = scaled_triangle[0], scaled_triangle[1]
        tr2_pt2 = scaled_triangle[2], scaled_triangle[3]
        tr2_pt3 = scaled_triangle[4], scaled_triangle[5]
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

        # Get a bouding box around the template traingle.
        rect2 = cv2.boundingRect(triangle2)
        (x, y, w, h) = rect2
        points2 = np.array(
            [
                [tr2_pt1[0] - x, tr2_pt1[1] - y],
                [tr2_pt2[0] - x, tr2_pt2[1] - y],
                [tr2_pt3[0] - x, tr2_pt3[1] - y],
            ]
        )

        # Affine Transform
        points1 = np.float32(points1)
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points1, points2)
        warped_triangle = cv2.warpAffine(
            cropped_triangle1, M, (w, h), flags=cv2.INTER_NEAREST
        )

        # Place triangles to match template.
        triangle_area = aligned[y : y + h, x : x + w]
        triangle_area_gray = cv2.cvtColor(triangle_area, cv2.COLOR_BGR2GRAY)

        # Remove the lines
        _, mask_triangles_designed = cv2.threshold(
            triangle_area_gray, 1, 255, cv2.THRESH_BINARY_INV
        )
        warped_triangle = cv2.bitwise_and(
            warped_triangle, warped_triangle, mask=mask_triangles_designed
        )

        triangle_area = cv2.add(triangle_area, warped_triangle)
        aligned[y : y + h, x : x + w] = triangle_area

    # Turn black background to white
    black_pixels = np.where(
        (aligned[:, :, 0] == 0) & (aligned[:, :, 1] == 0) & (aligned[:, :, 2] == 0)
    )

    # set those pixels to white
    aligned[black_pixels] = [255, 255, 255]

    return aligned


def parse_tfrecord_fn(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
        "keypoints": tf.io.VarLenFeature(tf.float32),

    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
    example["keypoints"] = tf.sparse.to_dense(example["keypoints"])
    return example["image"], example["keypoints"]