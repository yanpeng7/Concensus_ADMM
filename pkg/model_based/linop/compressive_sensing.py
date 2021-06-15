from ..linop import LinearOperator
import torch


class CompressiveSensing(LinearOperator):
    def __init__(self,
                 groundtruth_dim,
                 measurement_dim,
                 pre_saved_file_path=None,
                 ):

        self.groundtruth_dim, self.measurement_dim = groundtruth_dim, measurement_dim

        if pre_saved_file_path:
            try:
                self.A = torch.load(pre_saved_file_path)
                print("SUCCEED loading pre_saved matrix from: [%s]." % pre_saved_file_path)
            except:
                print("FAIL loading pre_saved matrix from: [%s]." % pre_saved_file_path)

                self.A = torch.randn(measurement_dim**2, groundtruth_dim**2)
                torch.save(self.A, pre_saved_file_path)
                print("Have created one then saved it at the same path.")

        else:
            self.A = torch.randn(measurement_dim**2, groundtruth_dim**2)

    def fmult(self, x):
        ret = torch.matmul(self.A, x.flatten())
        return ret.reshape([self.measurement_dim, self.measurement_dim])

    def ftran(self, x):
        ret = torch.matmul(self.A.transpose(0, 1), x.flatten())
        return ret.reshape([self.groundtruth_dim, self.groundtruth_dim])

    def cuda(self):
        self.A = self.A.cuda()

        return self
