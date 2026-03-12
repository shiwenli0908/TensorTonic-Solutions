def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    gray = []

    for i in range(len(image)):    # Iterate rows
        row = []
        for j in range(len(image[i])):    # Iterate cols

            R = image[i][j][0]
            G = image[i][j][1]
            B = image[i][j][2]

            gray_value = 0.299 * R + 0.587 * G + 0.114 * B

            row.append(gray_value)
        gray.append(row)

    return gray