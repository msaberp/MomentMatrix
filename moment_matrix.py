import torch
import torch.nn as nn
import math


class MomentMatrix(nn.Module):
    def __init__(self, k1, k2, device):
        super(MomentMatrix, self).__init__()
        self.factorial_array = []
        for i in range(max(k1, k2)):
            self.factorial_array.append(math.factorial(i))
        self.factorial_array = torch.Tensor(self.factorial_array)
        self.t = k1
        self.k = k2
        if self.t == 1:
            self.t_arr = torch.Tensor([1])
            self.t_ind_arr = torch.Tensor([1])
        else:
            self.t_arr = torch.arange(self.t)
            self.t_ind_arr = self.t_arr - self.t // 2
        self.k_arr = torch.arange(self.k)
        self.k_ind_arr = self.k_arr - self.k // 2

        self.coef_mat = self.create_coef_matrix().to(device)

    def create_coef_matrix(self):
        factorial_coef = (
            (1 / self.factorial_array[: self.t]).reshape(self.t, 1, 1)
            * (1 / self.factorial_array[: self.k]).reshape(1, self.k, 1)
            * (1 / self.factorial_array[: self.k]).reshape(1, 1, self.k)
        )
        mat_coef_t = self.t_ind_arr.reshape(1, self.t) ** self.t_arr.reshape(self.t, 1)
        mat_coef_k = self.k_ind_arr.reshape(1, self.k) ** self.k_arr.reshape(self.k, 1)
        mat_coef_tkk = (
            mat_coef_t.reshape(-1, 1, 1)
            * mat_coef_k.reshape(1, -1, 1)
            * mat_coef_k.reshape(1, 1, -1)
        )
        mat_coef_tkk = mat_coef_tkk.reshape(
            self.t, self.t, self.k, self.k, self.k, self.k
        )  # ixk1xjxk2xlxk3
        return factorial_coef.reshape(self.t, 1, self.k, 1, self.k, 1) * mat_coef_tkk

    def forward(self, kernel):
        b, c, _, _, _ = kernel.shape
        mul_mat = kernel.reshape(b, c, 1, self.t, 1, self.k, 1, self.k) * self.coef_mat
        return torch.einsum("bcijklmn->bcikm", mul_mat)
