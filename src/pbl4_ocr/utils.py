from __future__ import print_function
import math
import os

import cv2
import numpy as np
from PIL import Image, JpegImagePlugin
from scipy import ndimage
from skimage import io
from urllib.request import urlretrieve


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def loadImage(img_file):
    img = io.imread(img_file)  # RGB order
    if img.shape[0] == 2:
        img = img[0]
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = np.array(img)

    return img


def four_point_transform(image, rect):
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def group_text_box(
    polys,
    slope_ths=0.1,
    ycenter_ths=0.5,
    height_ths=0.5,
    width_ths=1.0,
    add_margin=0.05,
    sort_output=True,
):
    # poly top-left, top-right, low-right, low-left
    horizontal_list, free_list, combined_list, merged_list = [], [], [], []

    for poly in polys:
        slope_up = (poly[3] - poly[1]) / np.maximum(10, (poly[2] - poly[0]))
        slope_down = (poly[5] - poly[7]) / np.maximum(10, (poly[4] - poly[6]))
        if max(abs(slope_up), abs(slope_down)) < slope_ths:
            x_max = max([poly[0], poly[2], poly[4], poly[6]])
            x_min = min([poly[0], poly[2], poly[4], poly[6]])
            y_max = max([poly[1], poly[3], poly[5], poly[7]])
            y_min = min([poly[1], poly[3], poly[5], poly[7]])
            horizontal_list.append(
                [x_min, x_max, y_min, y_max, 0.5 * (y_min + y_max), y_max - y_min]
            )
        else:
            height = np.linalg.norm([poly[6] - poly[0], poly[7] - poly[1]])
            width = np.linalg.norm([poly[2] - poly[0], poly[3] - poly[1]])

            margin = int(1.44 * add_margin * min(width, height))

            theta13 = abs(
                np.arctan((poly[1] - poly[5]) / np.maximum(10, (poly[0] - poly[4])))
            )
            theta24 = abs(
                np.arctan((poly[3] - poly[7]) / np.maximum(10, (poly[2] - poly[6])))
            )
            # do I need to clip minimum, maximum value here?
            x1 = poly[0] - np.cos(theta13) * margin
            y1 = poly[1] - np.sin(theta13) * margin
            x2 = poly[2] + np.cos(theta24) * margin
            y2 = poly[3] - np.sin(theta24) * margin
            x3 = poly[4] + np.cos(theta13) * margin
            y3 = poly[5] + np.sin(theta13) * margin
            x4 = poly[6] - np.cos(theta24) * margin
            y4 = poly[7] + np.sin(theta24) * margin

            free_list.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    if sort_output:
        horizontal_list = sorted(horizontal_list, key=lambda item: item[4])

    # combine box
    new_box = []
    for poly in horizontal_list:

        if len(new_box) == 0:
            b_height = [poly[5]]
            b_ycenter = [poly[4]]
            new_box.append(poly)
        else:
            # comparable height and comparable y_center level up to ths*height
            if abs(np.mean(b_ycenter) - poly[4]) < ycenter_ths * np.mean(b_height):
                b_height.append(poly[5])
                b_ycenter.append(poly[4])
                new_box.append(poly)
            else:
                b_height = [poly[5]]
                b_ycenter = [poly[4]]
                combined_list.append(new_box)
                new_box = [poly]
    combined_list.append(new_box)

    # merge list use sort again
    for boxes in combined_list:
        if len(boxes) == 1:  # one box per line
            box = boxes[0]
            margin = int(add_margin * min(box[1] - box[0], box[5]))
            merged_list.append(
                [box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin]
            )
        else:  # multiple boxes per line
            boxes = sorted(boxes, key=lambda item: item[0])

            merged_box, new_box = [], []
            for box in boxes:
                if len(new_box) == 0:
                    b_height = [box[5]]
                    x_max = box[1]
                    new_box.append(box)
                else:
                    if (
                        abs(np.mean(b_height) - box[5]) < height_ths * np.mean(b_height)
                    ) and (
                        (box[0] - x_max) < width_ths * (box[3] - box[2])
                    ):  # merge boxes
                        b_height.append(box[5])
                        x_max = box[1]
                        new_box.append(box)
                    else:
                        b_height = [box[5]]
                        x_max = box[1]
                        merged_box.append(new_box)
                        new_box = [box]
            if len(new_box) > 0:
                merged_box.append(new_box)

            for mbox in merged_box:
                if len(mbox) != 1:  # adjacent box in same line
                    # do I need to add margin here?
                    x_min = min(mbox, key=lambda x: x[0])[0]
                    x_max = max(mbox, key=lambda x: x[1])[1]
                    y_min = min(mbox, key=lambda x: x[2])[2]
                    y_max = max(mbox, key=lambda x: x[3])[3]

                    box_width = x_max - x_min
                    box_height = y_max - y_min
                    margin = int(add_margin * (min(box_width, box_height)))

                    merged_list.append(
                        [x_min - margin, x_max + margin, y_min - margin, y_max + margin]
                    )
                else:  # non adjacent box in same line
                    box = mbox[0]

                    box_width = box[1] - box[0]
                    box_height = box[3] - box[2]
                    margin = int(add_margin * (min(box_width, box_height)))

                    merged_list.append(
                        [
                            box[0] - margin,
                            box[1] + margin,
                            box[2] - margin,
                            box[3] + margin,
                        ]
                    )
    # may need to check if box is really in image
    return merged_list, free_list


def calculate_ratio(width, height):
    """
    Calculate aspect ratio for normal use case (w>h) and vertical text (h>w)
    """
    ratio = width / height
    if ratio < 1.0:
        ratio = 1.0 / ratio
    return ratio


def compute_ratio_and_resize(img, width, height, model_height):
    """
    Calculate ratio and resize correctly for both horizontal text
    and vertical case
    """
    ratio = width / height
    if ratio < 1.0:
        ratio = calculate_ratio(width, height)
        img = cv2.resize(
            img,
            (model_height, int(model_height * ratio)),
            interpolation=Image.ANTIALIAS,
        )
    else:
        img = cv2.resize(
            img,
            (int(model_height * ratio), model_height),
            interpolation=Image.ANTIALIAS,
        )
    return img, ratio


def get_image_list(horizontal_list, free_list, img, model_height=64, sort_output=True):
    image_list = []
    maximum_y, maximum_x = img.shape

    max_ratio_hori, max_ratio_free = 1, 1
    for box in free_list:
        rect = np.array(box, dtype="float32")
        transformed_img = four_point_transform(img, rect)
        ratio = calculate_ratio(transformed_img.shape[1], transformed_img.shape[0])
        new_width = int(model_height * ratio)
        if new_width == 0:
            pass
        else:
            crop_img, ratio = compute_ratio_and_resize(
                transformed_img,
                transformed_img.shape[1],
                transformed_img.shape[0],
                model_height,
            )
            image_list.append(
                (box, crop_img)
            )  # box = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            max_ratio_free = max(ratio, max_ratio_free)

    max_ratio_free = math.ceil(max_ratio_free)

    for box in horizontal_list:
        x_min = max(0, box[0])
        x_max = min(box[1], maximum_x)
        y_min = max(0, box[2])
        y_max = min(box[3], maximum_y)
        crop_img = img[y_min:y_max, x_min:x_max]
        width = x_max - x_min
        height = y_max - y_min
        ratio = calculate_ratio(width, height)
        new_width = int(model_height * ratio)
        if new_width == 0:
            pass
        else:
            crop_img, ratio = compute_ratio_and_resize(
                crop_img, width, height, model_height
            )
            image_list.append(
                (
                    [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
                    crop_img,
                )
            )
            max_ratio_hori = max(ratio, max_ratio_hori)

    max_ratio_hori = math.ceil(max_ratio_hori)
    max_ratio = max(max_ratio_hori, max_ratio_free)
    max_width = math.ceil(max_ratio) * model_height

    if sort_output:
        image_list = sorted(
            image_list, key=lambda item: item[0][0][1]
        )  # sort by vertical position
    return image_list, max_width


def diff(input_list):
    return max(input_list) - min(input_list)


def get_paragraph(raw_result, x_ths=1, y_ths=0.5, mode="ltr"):
    # create basic attributes
    box_group = []
    for box in raw_result:
        all_x = [int(coord[0]) for coord in box[0]]
        all_y = [int(coord[1]) for coord in box[0]]
        min_x = min(all_x)
        max_x = max(all_x)
        min_y = min(all_y)
        max_y = max(all_y)
        height = max_y - min_y
        box_group.append(
            [box[1], min_x, max_x, min_y, max_y, height, 0.5 * (min_y + max_y), 0]
        )  # last element indicates group
    # cluster boxes into paragraph
    current_group = 1
    while len([box for box in box_group if box[7] == 0]) > 0:
        box_group0 = [box for box in box_group if box[7] == 0]  # group0 = non-group
        # new group
        if len([box for box in box_group if box[7] == current_group]) == 0:
            box_group0[0][7] = current_group  # assign first box to form new group
        # try to add group
        else:
            current_box_group = [box for box in box_group if box[7] == current_group]
            mean_height = np.mean([box[5] for box in current_box_group])
            min_gx = min([box[1] for box in current_box_group]) - x_ths * mean_height
            max_gx = max([box[2] for box in current_box_group]) + x_ths * mean_height
            min_gy = min([box[3] for box in current_box_group]) - y_ths * mean_height
            max_gy = max([box[4] for box in current_box_group]) + y_ths * mean_height
            add_box = False
            for box in box_group0:
                same_horizontal_level = (min_gx <= box[1] <= max_gx) or (
                    min_gx <= box[2] <= max_gx
                )
                same_vertical_level = (min_gy <= box[3] <= max_gy) or (
                    min_gy <= box[4] <= max_gy
                )
                if same_horizontal_level and same_vertical_level:
                    box[7] = current_group
                    add_box = True
                    break
            # cannot add more box, go to next group
            if add_box is False:
                current_group += 1
    # arrage order in paragraph
    result = []
    for i in set(box[7] for box in box_group):
        current_box_group = [box for box in box_group if box[7] == i]
        mean_height = np.mean([box[5] for box in current_box_group])
        min_gx = min([box[1] for box in current_box_group])
        max_gx = max([box[2] for box in current_box_group])
        min_gy = min([box[3] for box in current_box_group])
        max_gy = max([box[4] for box in current_box_group])

        text = ""
        while len(current_box_group) > 0:
            highest = min([box[6] for box in current_box_group])
            candidates = [
                box for box in current_box_group if box[6] < highest + 0.4 * mean_height
            ]
            # get the far left
            if mode == "ltr":
                most_left = min([box[1] for box in candidates])
                for box in candidates:
                    if box[1] == most_left:
                        best_box = box
            elif mode == "rtl":
                most_right = max([box[2] for box in candidates])
                for box in candidates:
                    if box[2] == most_right:
                        best_box = box
            text += " " + best_box[0]
            current_box_group.remove(best_box)

        result.append(
            [
                [
                    [min_gx, min_gy],
                    [max_gx, min_gy],
                    [max_gx, max_gy],
                    [min_gx, max_gy],
                ],
                text[1:],
            ]
        )

    return result


def printProgressBar(prefix="", suffix="", decimals=1, length=100, fill="█"):
    """
    Call in a loop to create terminal progress bar
    @params:
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """

    def progress_hook(count, blockSize, totalSize):
        progress = count * blockSize / totalSize
        percent = ("{0:." + str(decimals) + "f}").format(progress * 100)
        filledLength = int(length * progress)
        bar = fill * filledLength + "-" * (length - filledLength)
        print(f"\r{prefix} |{bar}| {percent}% {suffix}", end="")

    return progress_hook


def reformat_input(image):
    if type(image) == str:
        if image.startswith("http://") or image.startswith("https://"):
            tmp, _ = urlretrieve(
                image,
                reporthook=printProgressBar(
                    prefix="Progress:", suffix="Complete", length=50
                ),
            )
            img_cv_grey = cv2.imread(tmp, cv2.IMREAD_GRAYSCALE)
            os.remove(tmp)
        else:
            img_cv_grey = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            image = os.path.expanduser(image)
        img = loadImage(image)  # can accept URL
    elif type(image) == bytes:
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    elif type(image) == np.ndarray:
        if len(image.shape) == 2:  # grayscale
            img_cv_grey = image
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            img_cv_grey = np.squeeze(image)
            img = cv2.cvtColor(img_cv_grey, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 3:  # BGRscale
            img = image
            img_cv_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBAscale
            img = image[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif type(image) == JpegImagePlugin.JpegImageFile:
        image_array = np.array(image)
        img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(
            "Invalid input type."
            "Supporting format = string(file path or url), bytes, numpy array"
        )

    return img, img_cv_grey


def reformat_input_batched(image, n_width=None, n_height=None):
    """
    reformats an image or list of images or a 4D numpy image array &
    returns a list of corresponding img, img_cv_grey nd.arrays
    image:
        [file path, numpy-array, byte stream object,
        list of file paths, list of numpy-array, 4D numpy array,
        list of byte stream objects]
    """
    if (isinstance(image, np.ndarray) and len(image.shape) == 4) or isinstance(
        image, list
    ):
        # process image batches if image is list of image np arr, paths, bytes
        img, img_cv_grey = [], []
        for single_img in image:
            clr, gry = reformat_input(single_img)
            if n_width is not None and n_height is not None:
                clr = cv2.resize(clr, (n_width, n_height))
                gry = cv2.resize(gry, (n_width, n_height))
            img.append(clr)
            img_cv_grey.append(gry)
        img, img_cv_grey = np.array(img), np.array(img_cv_grey)
        # ragged tensors created when all input imgs are not of the same size
        if len(img.shape) == 1 and len(img_cv_grey.shape) == 1:
            raise ValueError(
                "The input image array contains images of different sizes. "
                "Please resize all images to same shape"
                " or pass n_width, n_height to auto-resize"
            )
    else:
        img, img_cv_grey = reformat_input(image)
    return img, img_cv_grey


def make_rotated_img_list(rotationInfo, img_list):

    result_img_list = img_list[:]

    # add rotated images to original image_list
    max_ratio = 1

    for angle in rotationInfo:
        for img_info in img_list:
            rotated = ndimage.rotate(img_info[1], angle, reshape=True)
            height, width = rotated.shape
            ratio = calculate_ratio(width, height)
            max_ratio = max(max_ratio, ratio)
            result_img_list.append((img_info[0], rotated))
    return result_img_list


def set_result_with_confidence(results):
    """Select highest confidence augmentation for TTA
    Given a list of lists of results (outer list has one list per augmentation,
    inner lists index the images being recognized), choose the best result
    according to confidence level.
    Each "result" is of the form (box coords, text, confidence)
    A final_result is returned which contains one result for each image
    """
    final_result = []
    for col_ix in range(len(results[0])):
        # Take the row_ix associated with the max confidence
        best_row = max(
            [(row_ix, results[row_ix][col_ix][2]) for row_ix in range(len(results))],
            key=lambda x: x[1],
        )[0]
        final_result.append(results[best_row][col_ix])

    return final_result
