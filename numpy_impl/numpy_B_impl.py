import numpy as np


def compute_weight_matrix(init_coordinate):
    [y_start, x_start] = init_coordinate
    y_length_mat = np.ones((3, 3))
    x_length_mat = np.ones((3, 3))
    x_length_mat[:, 0] = 1 - x_start
    x_length_mat[:, 2] = x_start
    y_length_mat[0, :] = 1 - y_start
    y_length_mat[2, :] = y_start
    return y_length_mat * x_length_mat


# K is the support of the filter (2K+1)x(2K+1)
def conv(image, kernel):
    [H, W] = image.shape
    K = (kernel.shape[0] - 1) // 2
    padded_image = np.zeros((H + 1, W + 1))
    padded_image[:H, :W] = image
    out = np.zeros(padded_image.shape, dtype='float64')
    for shift_x in range(-K, K + 1):
        for shift_y in range(-K, K + 1):
            shifted_image = np.roll(padded_image, (shift_y, shift_x), axis=(0, 1))
            out += shifted_image * kernel[K - shift_y, K - shift_x] * 0.25
    out = out[1:H:2, 1:W:2]
    return out


def transposed_conv(image, kernel):
    [H, W] = image.shape
    K = (kernel.shape[0] - 1) // 2
    padded_image = np.zeros((H * 2, W * 2))
    out = np.zeros(padded_image.shape)
    padded_image[1::2, 1::2] = image
    rot_kernel = np.rot90(np.rot90(kernel, axes=(0, 1)), axes=(0, 1))
    for shift_x in range(-K, K + 1):
        for shift_y in range(-K, K + 1):
            shifted_image = np.roll(padded_image, (shift_y, shift_x), axis=(0, 1))
            if shift_y < 0:
                shifted_image[shift_y:, :] = 0
            else:
                shifted_image[:shift_y, :] = 0
            if shift_x < 0:
                shifted_image[:, shift_x:] = 0
            else:
                shifted_image[:, :shift_x] = 0
            out += shifted_image * rot_kernel[K - shift_y, K - shift_x] * 0.25
    return out


def B_numpy(image, init_point_list):
    out = []
    for i, init_point in enumerate(init_point_list):
        weight_matrix = compute_weight_matrix(init_point)
        out.append(conv(image, weight_matrix))
    out = np.stack(np.asarray(out), axis=0)
    return out


def B_T_numpy(image, init_point_list):
    out = []
    for i, init_point in enumerate(init_point_list):
        weight_matrix = compute_weight_matrix(init_point)
        out.append(transposed_conv(image[i], weight_matrix))
    out = np.sum(np.asarray(out), axis=0)
    return out


if __name__ == '__main__':
    pass
