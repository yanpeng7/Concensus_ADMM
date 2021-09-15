import numpy as np
import astra
import tomo.tomo as tomo

# assuming 256*256 input
# x_N = 256
# y_N = x_N // 2
# num_detector = 256

class OperatorA:
    def __init__(self,
                 input_size: int,
                 num_detector: int,
                 num_angles: int,
                 init_point_list: np.ndarray
                 ):
        self.x_N = input_size
        self.y_N = self.x_N // 2
        self.num_detector = num_detector
        self.num_angles = num_angles
        self.init_points = init_point_list

    def A_i(self, x, y_shift, x_shift):
        proj_geom = astra.create_proj_geom('parallel', 1.0, self.num_detector, np.linspace(0, np.pi, self.num_angles))
        vol_geom = astra.create_vol_geom(self.y_N, self.y_N, -self.y_N + x_shift, self.y_N + x_shift,
                                         -self.y_N - y_shift, self.y_N - y_shift)
        projector = astra.create_projector("line", proj_geom, vol_geom)
        astra_id, result = astra.create_sino(x, projector)
        astra.data2d.delete(astra_id)
        return result

    def A_i_v2(self, x):
        proj_geom = astra.create_proj_geom('parallel', 1.0, self.num_detector, np.linspace(0, np.pi, self.num_angles))
        vol_geom = astra.create_vol_geom(self.y_N, self.y_N, -self.y_N, self.y_N, -self.y_N, self.y_N)
        projector = astra.create_projector("line", proj_geom, vol_geom)
        astra_id, result = astra.create_sino(x, projector)
        astra.data2d.delete(astra_id)
        return result

    # astra bug, shift not needed
    def A_i_adj(self, x, y_shift, x_shift):
        proj_geom = astra.create_proj_geom('parallel', 1.0, self.num_detector, np.linspace(0, np.pi, self.num_angles))
        vol_geom = astra.create_vol_geom(self.y_N, self.y_N, -self.y_N, self.y_N, -self.y_N, self.y_N)
        projector = astra.create_projector("line", proj_geom, vol_geom)
        astra_id, result = astra.create_backprojection(x, projector)
        astra.data2d.delete(astra_id)
        return result

    # v2 version using tomo package implemented by Dr. Doga Gursoy
    def A_i_adj_v2(self, x, y_shift, x_shift):
        REGPAR = np.array(1, dtype='float32')
        theta = np.linspace(0, np.pi, self.num_angles, endpoint=False, dtype='float32')
        result = tomo.backproj(x, theta, sx=-y_shift, sy=-x_shift, downscale=2, reg_pars=REGPAR)[0]
        return result

    def fmult(self, x):
        x_list = []
        for i in range(self.init_points.shape[0]):
            x_list.append(self.A_i(x[i], self.init_points[i, 0], self.init_points[i, 1]))
        return np.stack(x_list, axis=0)

    def fmult_v2(self, x):
        x_list = []
        for i in range(self.init_points.shape[0]):
            x_list.append(self.A_i_v2(x[i]))
        return np.stack(x_list, axis=0)

    def adj(self, x):
        x_list = []
        for i in range(self.init_points.shape[0]):
            x_list.append(self.A_i_adj(x[i], self.init_points[i, 0], self.init_points[i, 1]))
        return np.stack(x_list, axis=0)

    def adj_v2(self, x):
        x_list = []
        for i in range(self.init_points.shape[0]):
            x_list.append(self.A_i_adj_v2(x[i], self.init_points[i, 0], self.init_points[i, 1]))
        return np.stack(x_list, axis=0)

