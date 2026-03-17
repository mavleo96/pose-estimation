def calculate_square_padding(w, h):
    if w > h:
        pad_top = (w - h) // 2
        pad_bottom = w - h - pad_top
        return 0, pad_top, 0, pad_bottom
    else:
        pad_left = (h - w) // 2
        pad_right = h - w - pad_left
        return pad_left, 0, pad_right, 0
