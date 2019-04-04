# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.autograd import Function
from torch.nn import Module, Parameter
from spconv import SparseConvTensor


class NetworkInNetwork(Module):
    def __init__(self, nIn, nOut, bias=False):
        Module.__init__(self)
        self.nIn = nIn
        self.nOut = nOut
        std = (2.0 / nIn)**0.5
        self.weight = Parameter(torch.Tensor(
            nIn, nOut).normal_(
            0,
            std))
        if bias:
            self.bias = Parameter(torch.Tensor(nOut).zero_())

    def forward(self, input):
        # assert input.features.nelement() == 0 or input.features.size(1) == self.nIn, (self.nIn, input.features.shape)
        output = SparseConvTensor( 
                input.features, input.indices,
                input.spatial_shape, input.batch_size) 
        return output

    def __repr__(self):
        s = 'NetworkInNetwork' + str(self.nIn) + '->' + str(self.nOut)
        return s

    def input_spatial_size(self, out_size):
        return out_size
