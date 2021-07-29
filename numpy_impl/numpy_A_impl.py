import numpy as np
import astra

# assuming 256*256 input
x_N = 256
y_N = x_N // 2
num_detector = 256


def A_i(x, y_shift, x_shift):
    # print(f'A x shift: {x_shift}')
    # print(f'A y shift: {y_shift}')
    proj_geom = astra.create_proj_geom('parallel', 1.0, num_detector, np.linspace(0, np.pi, 180))
    vol_geom = astra.create_vol_geom(y_N, y_N, -y_N + x_shift, y_N + x_shift, -y_N - y_shift, y_N - y_shift)
    projector = astra.create_projector("strip", proj_geom, vol_geom)
    astra_id, result = astra.create_sino(x, projector)
    astra.data2d.delete(astra_id)
    return result


def A_i_adj(x, y_shift, x_shift):
    # print(f'A adj x shift: {x_shift}')
    # print(f'A adj shift: {y_shift}')
    proj_geom = astra.create_proj_geom('parallel', 1.0, num_detector, np.linspace(0, np.pi, 180))

    vol_geom = astra.create_vol_geom(y_N, y_N, -y_N + x_shift, y_N + x_shift, -y_N - y_shift, y_N - y_shift)

    projector = astra.create_projector("strip", proj_geom, vol_geom)
    astra_id, result = astra.create_backprojection(x, projector)
    astra.data2d.delete(astra_id)
    return result


def A_batched(x, mat_list):
    x_list = []
    for i in range(mat_list.shape[0]):
        x_list.append(A_i(x[i], mat_list[i, 0], mat_list[i, 1]))
    return np.stack(x_list, axis=0)


def A_adj_batched(x, mat_list):
    x_list = []
    for i in range(mat_list.shape[0]):
        x_list.append(A_i_adj(x[i], mat_list[i, 0], mat_list[i, 1]))
    return np.stack(x_list, axis=0)


if __name__ == '__main__':
    pass
