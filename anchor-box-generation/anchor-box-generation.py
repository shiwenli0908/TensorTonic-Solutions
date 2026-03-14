def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    anchors = []

    stride = image_size / feature_size

    for i in range(feature_size):    # Rows
        for j in range(feature_size):    # Cols
            # Grid cell center in image coordination
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride

            for scale in scales:
                for ratio in aspect_ratios:
                    # Ratio = w / h
                    w = scale * (ratio ** 0.5)
                    h = scale / (ratio ** 0.5)

                    x1 = cx - 0.5 * w
                    y1 = cy - 0.5 * h
                    x2 = cx + 0.5 * w
                    y2 = cy + 0.5 * h

                    anchors.append([x1, y1, x2, y2])

    return anchors
                    

    