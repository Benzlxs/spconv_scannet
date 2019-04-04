# Copyright Xuesong LI, UNSW, (benzlee08@gmail.com)
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import spconv
from torch import nn


def UNet(reps, nPlanes, residual_blocks=True, downsample=[3, 2], leakiness=0):
    """
    U-Net style network with VGG or ResNet-style blocks.
    For voxel level prediction:
    import sparseconvnet as scn
    import torch.nn
    class Model(nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.sparseModel = scn.Sequential().add(
               scn.SubmanifoldConvolution(3, nInputFeatures, 64, 3, False)).add(
               scn.UNet(3, 2, [64, 128, 192, 256], residual_blocks=True, downsample=[2, 2]))
            self.linear = nn.Linear(64, nClasses)
        def forward(self,x):
            x=self.sparseModel(x).features
            x=self.linear(x)
            return x
    """
    #def block(m, a, b):
    #    if residual_blocks: #ResNet style blocks
    #        m.add(spconv.ConcatTable()
    #              .add(spconv.Identity() if a == b else spconv.NetworkInNetwork(a, b, False))
    #              .add(spconv.SparseSequential()
    #                .add( SubMConv3d )

    #                .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
    #                .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False))
    #                .add(scn.BatchNormLeakyReLU(b,leakiness=leakiness))
    #                .add(scn.SubmanifoldConvolution(dimension, b, b, 3, False)))
    #         ).add(scn.AddTable())
    #    else: #VGG style blocks
    #        m.add(scn.Sequential()
    #             .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
    #             .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False)))




    def U(nPlanes): #Recursive function
        m = spconv.SparseSequential()
        if len(nPlanes) == 1:
            for _ in range(reps):
                m.add( spconv.SparseBasicBlock(nPlanes[0], nPlanes[0],3, indice_key="subm{}".format(len(nPlanes))))
        else:
            m = spconv.SparseSequential()
            for _ in range(reps):
                m.add( spconv.SparseBasicBlock(nPlanes[0], nPlanes[0],3, indice_key="subm{}".format(len(nPlanes))))
            m.add(
                spconv.ConcatTable().add(
                    spconv.Identity()).add(
                    spconv.SparseSequential().add(
                        spconv.SparseConv3d(nPlanes[0], nPlanes[1], downsample[0], stride = downsample[1], bias=False, indice_key ="conv{}".format(len(nPlanes)))).add(
                            nn.BatchNorm1d(nPlanes[1], eps=1e-3, momentum=0.01)).add(nn.ReLU()
                                ).add(
                        U(nPlanes[1:])).add(
                        spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0],downsample[0], bias=False, indice_key ="conv{}".format(len(nPlanes)))).add(
                            nn.BatchNorm1d(nPlanes[0], eps=1e-3, momentum=0.01)).add(nn.ReLU())))
            m.add(spconv.JoinTable())
            for i in range(reps):
                m.add( spconv.SubMConv3d( nPlanes[0] * 2, nPlanes[0], 3, bias=False, indice_key="end_pp{}".format(len(nPlanes)))).add(
                        nn.BatchNorm1d(nPlanes[0], eps=1e-3, momentum=0.01)).add(nn.ReLU())
                m.add( spconv.SparseBasicBlock(nPlanes[0], nPlanes[0],3, indice_key="end_pp{}".format(len(nPlanes))))
        return m
    m = U(nPlanes)
    return m

