import cv2
import numpy as np
import pycocotools.mask as pycoco_mask_utils

from ..utils.image import get_crop_matrix, get_pad_matrix, get_resize_matrix


def rle_to_mask(segmentation):
    h, w = segmentation["size"]
    rle = pycoco_mask_utils.frPyObjects(segmentation, h, w)
    return pycoco_mask_utils.decode(rle)  # (H, W) uint8, values 0 or 1


def calculate_square_padding(w, h):
    if w > h:
        pad_top = (w - h) // 2
        pad_bottom = w - h - pad_top
        return 0, pad_top, 0, pad_bottom
    else:
        pad_left = (h - w) // 2
        pad_right = h - w - pad_left
        return pad_left, 0, pad_right, 0


def get_bbox(img):
    # input convention: (h, w, 3) uint8
    assert img.ndim == 3 and img.shape[2] == 3
    # return smallest bounding box containing all non-black pixels
    y, x = np.where(img.sum(-1) > 0)
    if len(x) == 0:
        return None  # image is entirely black
    x1, y1 = x.min(), y.min()
    x2, y2 = x.max() + 1, y.max() + 1
    w, h = x2 - x1, y2 - y1
    return x1, y1, w, h


def process_rgb_and_depth(rgb, depth, bbox, output_size, mask=None):
    # mask the rgb and depth if mask is not None
    if mask is not None:
        rgb = np.where(mask[..., None], rgb, 0)
        depth = np.where(mask, depth, 0)

    x, y, w, h = bbox
    padding = calculate_square_padding(w, h)

    # 1. crop using the bbox
    # 2. pad the cropped image to make it square
    # 3. resize the padded image to the output size
    # Note: square padding is wrong if output size is not square
    rgb = rgb[y : y + h, x : x + w]
    rgb = np.pad(
        rgb,
        (
            (padding[1], padding[3]),
            (padding[0], padding[2]),
            (0, 0),
        ),
    )
    src_size = rgb.shape[:2]
    # cv2 resize expects (W, H) convention
    rgb = cv2.resize(rgb, output_size[::-1], interpolation=cv2.INTER_LINEAR)

    depth = depth[y : y + h, x : x + w]
    depth = np.pad(
        depth,
        (
            (padding[1], padding[3]),
            (padding[0], padding[2]),
        ),
    )
    depth = cv2.resize(depth, output_size[::-1], interpolation=cv2.INTER_NEAREST)

    # compute the transformation matrix
    T_crop = get_crop_matrix(bbox)
    T_pad = get_pad_matrix(padding)
    T_resize = get_resize_matrix(src_size, output_size)

    M = T_resize @ T_pad @ T_crop

    return rgb, depth, M
