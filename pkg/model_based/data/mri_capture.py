import torch
import os
from tqdm import tqdm
import h5py
import numpy as np
from ...utility.torch_complex_op import new_torch_complex


Kd = 960
Nd = 640


index2file_map = {
    1:  'Subj01',
    2:  'Subj02',
    3:  'Subj03',
    4:  'Subj04',
    5:  'Subj05',
    6:  'Subj06',
    7:  'Subj07',
    8:  'Subj08',
    9:  'Subj09',
    10: 'Subj10',
    37: 'Y90_PMR37',
    40: 'Y90_PMR40',
    44: 'Y90_PMR44',
    45: 'Y90_PMR45',
    50: 'Y90_PMR50',
    52: 'Y90_PMR52',
    54: 'Y90_PMR54',
    55: 'Y90_PMR55',
}


default_index2slice_map = {
    1:  tuple(range(1, 96 + 1)),
    2:  tuple(range(1, 96 + 1)),
    3:  tuple(range(1, 96 + 1)),
    4:  tuple(range(1, 96 + 1)),
    5:  tuple(range(1, 96 + 1)),
    6:  tuple(range(1, 96 + 1)),
    7:  tuple(range(1, 96 + 1)),
    8:  tuple(range(1, 96 + 1)),
    9:  tuple(range(1, 96 + 1)),
    10: tuple(range(1, 96 + 1)),
    37: tuple(range(1, 96 + 1)),
    40: tuple(range(1, 120 + 1)),
    44: tuple(range(1, 96 + 1)),
    45: tuple(range(1, 112 + 1)),
    50: tuple(range(1, 112 + 1)),
    52: tuple(range(1, 96 + 1)),
    54: tuple(range(1, 104 + 1)),
    55: tuple(range(1, 96 + 1)),
}


def load_forward_models_files(
        file_index,
        slice_indexes,
        num_lines,
        root_path='/export1/project/Dataset/HA/Source_Files/',
        num_phases=10,
):

    ret = []
    for s_index in tqdm(slice_indexes, desc='[HAsCAPTURE load_FM_files... INDEX %d]' % file_index):
        file_path = root_path + index2file_map[file_index] + '/FM_SLIM_nLines_v2%d_nPhases%d/S%d.h5' % (
            num_lines, num_phases, s_index)
        if os.path.exists(file_path):
            print(file_path)
            ret.append(file_path)

        else:
            raise FileNotFoundError('FM File Index: [%.2d, %s] nLines: [%.2d] Slice Index: [%.2d] file_path: [%s]'
                                    % (file_index, index2file_map[file_index], num_lines, s_index, file_path))

    return ret


def load_tgv_reconstruction_files(
        file_index,
        slice_indexes,
        num_lines,
        root_path='/export1/project/Dataset/HA/Source_Files/',
        num_phases=10,
):

    ret = []
    for s_index in tqdm(slice_indexes, desc='[HAsCAPTURE load_CS_files... INDEX %d]' % file_index):
        file_path = root_path + index2file_map[file_index] + '/Rec_nLines_v2%d_nPhases%d/S%d_CS.h5' % (
            num_lines, num_phases, s_index)
        if os.path.exists(file_path):
            print(file_path)
            ret.append(file_path)

        else:
            raise FileNotFoundError('mcnufft File Index: [%.2d, %s] nLines: [%.2d] Slice Index: [%.2d]file_path: [%s]'
                                    % (file_index, index2file_map[file_index], num_lines, s_index, file_path))

    return ret


def load_mcnufft_reconstruction_files(
        file_index,
        slice_indexes,
        num_lines,
        root_path='/export1/project/Dataset/HA/Source_Files/',
        num_phases=10,
):

    ret = []
    for s_index in tqdm(slice_indexes, desc='[HAsCAPTURE load_MCNUFFT_files... INDEX %d]' % file_index):
        file_path = root_path + index2file_map[file_index] + '/Rec_nLines_v2%d_nPhases%d/S%d_MCNUFFT.h5' % (
            num_lines, num_phases, s_index)
        if os.path.exists(file_path):
            print(file_path)
            ret.append(file_path)

        else:
            raise FileNotFoundError('tgv File Index: [%.2d, %s] nLines: [%.2d] Slice Index: [%.2d]file_path: [%s]'
                                    % (file_index, index2file_map[file_index], num_lines, s_index, file_path))

    return ret


class CAPTURE:
    def __init__(
            self,
            file_indexes,
            num_lines,
            index2slice_map: dict = None,
            is_pre_load_to_memory: bool = False,
            is_ret_forward_model: bool = True,
            is_ret_mcnufft: bool = False,
            is_ret_tgv: bool = False,
    ):

        assert (is_ret_forward_model or is_ret_mcnufft or is_ret_tgv)

        if index2slice_map is None:
            index2slice_map = {}

        self.is_ret_forward_model = is_ret_forward_model
        self.is_ret_mcnufft = is_ret_mcnufft
        self.is_ret_tgv = is_ret_tgv

        self.forward_model_files = []
        self.mcnufft_files = []
        self.tgv_files = []

        self.__index_map = []
        for f_index, index in enumerate(file_indexes):
            if index in index2slice_map:
                slice_indexes = index2slice_map[index]
            else:
                slice_indexes = default_index2slice_map[index]

            if self.is_ret_forward_model:
                self.forward_model_files.append(
                    load_forward_models_files(
                        file_index=index, num_lines=num_lines, slice_indexes=slice_indexes)
                )

            if self.is_ret_mcnufft:
                self.mcnufft_files.append(
                    load_mcnufft_reconstruction_files(
                        file_index=index, num_lines=num_lines, slice_indexes=slice_indexes)
                )

            if self.is_ret_tgv:
                self.tgv_files.append(
                    load_tgv_reconstruction_files(
                        file_index=index, num_lines=num_lines, slice_indexes=slice_indexes)
                )

            for s_index in range(slice_indexes.__len__()):
                self.__index_map.append([f_index, s_index])

        self.is_pre_load_to_memory = is_pre_load_to_memory
        self.pre_load_data = []
        if self.is_pre_load_to_memory:
            for item in tqdm(range(len(self)), desc='Pre Loading Data to RAM'):
                self.pre_load_data.append(self.__getitemHelper__(item))

    def __len__(self):
        return self.__index_map.__len__()

    def __getitemHelper__(self, item):
        f_index, s_index = self.__index_map[item]

        ret = tuple()

        if self.is_ret_forward_model:
            h5_file = self.forward_model_files[f_index]

            h5_file = h5_file[s_index]
            h5_file = h5py.File(h5_file, 'r', swmr=True)

            y = new_torch_complex(h5_file['y_real'][:], h5_file['y_imag'][:])
            b1 = new_torch_complex(h5_file['b1_real'][:], h5_file['b1_imag'][:])
            w = torch.from_numpy(h5_file['w'][:])
            ktraj = new_torch_complex(h5_file['ktraj_real'][:], h5_file['ktraj_imag'][:]) * 2 * np.pi

            y = y.permute([0, 3, 4, 1, 2])
            ktraj = ktraj.permute([0, 3, 1, 2])

            ret = ret + (y, b1, w, ktraj)

            h5_file.close()

        if self.is_ret_mcnufft:
            h5_file = self.mcnufft_files[f_index]

            h5_file = h5_file[s_index]
            h5_file = h5py.File(h5_file, 'r', swmr=True)

            recon_mcnufft = new_torch_complex(
                h5_file['recon_mcnufft_real'][:], h5_file['recon_mcnufft_imag'][:]
            ).permute([3, 0, 1, 2]).to(torch.float32)

            ret = ret + (recon_mcnufft,)

            h5_file.close()

        if self.is_ret_tgv:
            h5_file = self.tgv_files[f_index]

            h5_file = h5_file[s_index]
            h5_file = h5py.File(h5_file, 'r', swmr=True)

            recon_tgv = new_torch_complex(
                h5_file['recon_CS_real'][:], h5_file['recon_CS_imag'][:]
            ).permute([3, 0, 1, 2]).to(torch.float32)

            ret = ret + (recon_tgv,)

            h5_file.close()

        return ret

    def __getitem__(self, item):
        if self.is_pre_load_to_memory:
            return self.pre_load_data[item]

        else:
            return self.__getitemHelper__(item)
