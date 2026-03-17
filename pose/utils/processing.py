import pycocotools.mask as pycoco_mask_utils


def rle_to_mask(segmentation):
    """Convert RLE to mask."""
    h, w = segmentation["size"]
    rle = pycoco_mask_utils.frPyObjects(segmentation, h, w)
    return pycoco_mask_utils.decode(rle)  # (H, W) uint8, values 0 or 1
