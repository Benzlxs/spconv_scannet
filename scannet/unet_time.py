# Xuesong LI, UNSW, benzlee08@gmail.com
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Options
m = 16 # 16 or 32
residual_blocks=True #True or False
block_reps = 2 #Conv block repetition factor: 1 or 2
dense_shape = [512, 512, 512] # [batch, h, w, l, channels]
features_cc = 3 # rgb or xyz
num_layers = 7

import torch, data, iou
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sparseconvnet as scn
import spconv
import time
import os, sys, glob
import math
import numpy as np

use_cuda = torch.cuda.is_available()
exp_name='unet_scale20_m16_rep1_notResidualBlocks'
exp_name='unet_scale20_m%d_rep%d_%sResidualBlocks'%(m, block_reps, '' if residual_blocks else 'not')
exp_name='unet_scale50_m%d_rep%d_%sResidualBlocks'%(m, block_reps, '' if residual_blocks else 'not')

def UNet_vgg(reps, nPlanes, residual_blocks=True, downsample=[3, 2], leakiness=0):

    def U(nPlanes): #Recursive function
        m = spconv.SparseSequential()
        if len(nPlanes) == 1:
            for _ in range(reps):
                #m.add( spconv.SparseBasicBlock(nPlanes[0], nPlanes[0],3, indice_key="subm{}".format(len(nPlanes))))
                m.add(spconv.SubMConv3d(nPlanes[0], nPlanes[0],3, bias=False, indice_key="subm{}".format(len(nPlanes)))).add(
                nn.BatchNorm1d(nPlanes[0], eps=1e-3, momentum=0.01)).add(nn.ReLU())
        else:
            m = spconv.SparseSequential()
            for _ in range(reps):
                # m.add( spconv.SparseBasicBlock(nPlanes[0], nPlanes[0],3, indice_key="subm{}".format(len(nPlanes))))
                m.add(spconv.SubMConv3d(nPlanes[0], nPlanes[0],3, bias=False, indice_key="subm{}".format(len(nPlanes)))).add(
                nn.BatchNorm1d(nPlanes[0], eps=1e-3, momentum=0.01)).add(nn.ReLU())
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
                m.add( spconv.SubMConv3d( nPlanes[0] * (2 if i == 0 else 1), nPlanes[0], 3, bias=False, indice_key="subm{}".format(len(nPlanes)))).add(
                        nn.BatchNorm1d(nPlanes[0], eps=1e-3, momentum=0.01)).add(nn.ReLU())
                # m.add( spconv.SparseBasicBlock(nPlanes[0], nPlanes[0],3, indice_key="end_pp{}".format(len(nPlanes))))
        return m
    m = U(nPlanes)
    return m


class Model(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.input0 = scn.InputLayer(data.dimension, data.full_scale, mode=4)
        self.sparseModel = spconv.SparseSequential(
           spconv.SubMConv3d( 3, m, 3, bias=False, indice_key="start_")).add(
           nn.BatchNorm1d(m, eps=1e-3, momentum=0.01)).add(nn.ReLU()).add(
               UNet_vgg(block_reps,  [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m ], residual_blocks)) # [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m]
        self.out0 = scn.OutputLayer(data.dimension)
        self.linear = nn.Linear(m, 20)

    #def forward(self,features, coors, batch_size):
    def forward(self,x):
        t1 = time.time()
        in0 = self.input0(x)
        # coors = coors.int()[:,[3,2,1,0]] # ordering is [batch, z, y, z]
        coors = in0.get_spatial_locations().int()[:,[3,2,1,0]]
        coors.cuda()
        t2 = time.time()
        print("in time:{}".format(t2-t1))
        ret  = spconv.SparseConvTensor(in0.features, coors, dense_shape, data.batch_size)
        x=self.sparseModel(ret)
        t3 = time.time()
        print("middle time:{}".format(t3-t2))
        temp0 = scn.SparseConvNetTensor(x.features, in0.metadata)
        x = self.out0(temp0)
        t4 = time.time()
        print("out time:{}".format(t4-t3))
        x=self.linear(x)
        return x

class Model_bk(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.input00 = scn.InputLayer(data.dimension, data.full_scale, mode=4)
        #self.sparseModel = spconv.SparseSequential(
        #   spconv.SubMConv3d( 3, m, 3, bias=False, indice_key="start_")).add(
        #   nn.BatchNorm1d(m, eps=1e-3, momentum=0.01)).add(nn.ReLU()).add(
        #       spconv.UNet(block_reps,  [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m ], residual_blocks)) # [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m]
        self.input0 = spconv.SparseSequential(
            spconv.SubMConv3d(3, m, 3, bias=False, indice_key="subm0"),
            nn.BatchNorm1d(m, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(m, m, 3, bias=False, indice_key="subm0"),
            nn.BatchNorm1d(m, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(m, m, 3, bias=False, indice_key="subm0"),
            nn.BatchNorm1d(m, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            )
        self.input_net = []
        self.upsample_net = []
        self.fuse_net = []
        for k in range(2,num_layers+1):
            # input network
            temp_model = None
            temp_model = spconv.SparseSequential(
              spconv.SparseConv3d((k-1)*m, k*m, 3, stride = 2, bias=False, indice_key="conv{}".format(k-1)),
              nn.BatchNorm1d( k*m, eps=1e-3, momentum=0.01),
              nn.ReLU(),
              spconv.SubMConv3d( k*m, k*m, 3, bias=False, indice_key="subm{}".format(k-1)),
              nn.BatchNorm1d( k*m, eps=1e-3, momentum=0.01),
              nn.ReLU(),
              spconv.SubMConv3d( k*m, k*m, 3, bias=False, indice_key="subm{}".format(k-1)),
              nn.BatchNorm1d( k*m, eps=1e-3, momentum=0.01),
              nn.ReLU(),
            )
            self.input_net.append(temp_model)
            # upsample network
            temp_model = None
            temp_model = spconv.SparseSequential(
              spconv.SparseInverseConv3d(k*m, (k-1)*m, 3, bias=False, indice_key="conv{}".format(k-1)),
              nn.BatchNorm1d( (k-1)*m, eps=1e-3, momentum=0.01),
              nn.ReLU(),
            )
            self.upsample_net.append(temp_model)
            # fuse network
            temp_model = None
            temp_model = spconv.SparseSequential(
              spconv.SubMConv3d( (k-1)*m*2, (k-1)*m, 3, bias=False, indice_key="subm{}".format(k-2)),
              nn.BatchNorm1d( (k-1)*m, eps=1e-3, momentum=0.01),
              nn.ReLU(),
              spconv.SubMConv3d( (k-1)*m, (k-1)*m, 3, bias=False, indice_key="subm{}".format(k-2)),
              nn.BatchNorm1d( (k-1)*m, eps=1e-3, momentum=0.01),
              nn.ReLU(),
            )
            self.fuse_net.append(temp_model)



        self.out0 = scn.OutputLayer(data.dimension)
        self.linear = nn.Linear(m, 20)


    def concate_inputs(self, input1, input2):
         output = spconv.SparseConvTensor(
                torch.cat([input1.features, input2.features],1) ,input1.indices, input1.spatial_shape, input1.batch_size )
         return output
    #def forward(self,features, coors, batch_size):
    def forward(self,x):
        t1 = time.time()
        in0 = self.input00(x)
        # coors = coors.int()[:,[3,2,1,0]] # ordering is [batch, z, y, z]
        coors = in0.get_spatial_locations().int()[:,[3,2,1,0]]
        t2 = time.time()
        print("in time:{}".format(t2-t1))
        ret  = spconv.SparseConvTensor(in0.features, coors, dense_shape, data.batch_size)
        # x=self.sparseModel(ret)
        x=[]
        ret = self.input0(ret)
        x.append(ret)
        # forwarding
        for i in range(num_layers-1):
            ret = self.input_net[i](ret)
            x.append(ret)
        for i in range(num_layers-1, 0, -1):
            ret = self.upsample_net[i](x[i])
            ret = self.concate_inputs(ret,x[i-1])
            ret = self.fuse_net[i-1](ret)
        t3 = time.time()
        print("middle time:{}".format(t3-t2))
        temp0 = scn.SparseConvNetTensor(ret.features, in0.metadata)
        ret = self.out0(temp0)
        t4 = time.time()
        print("out time:{}".format(t4-t3))
        ret=self.linear(ret)
        return ret

unet=Model()
if use_cuda:
    unet=unet.cuda()

training_epochs=512
training_epoch=spconv.checkpoint_restore(unet,exp_name,'unet',use_cuda)
optimizer = optim.Adam(unet.parameters())
print('#classifer parameters', sum([x.nelement() for x in unet.parameters()]))

for epoch in range(training_epoch, training_epochs+1):
    unet.train()
    stats = {}
    # scn.forward_pass_multiplyAdd_count=0
    # scn.forward_pass_hidden_states=0
    start = time.time()
    train_loss=0
    for i,batch in enumerate(data.train_data_loader):
        optimizer.zero_grad()
        if use_cuda:
            batch['x'][1]=batch['x'][1].cuda()
            batch['y']=batch['y'].cuda()
        print("fetching data {}".format(time.time()-start))
        predictions=unet(batch['x'])
        #predictions=unet(batch['x'][1], batch['x'][0], data.batch_size)
        loss = torch.nn.functional.cross_entropy(predictions,batch['y'])
        train_loss+=loss.item()
        loss.backward()
        optimizer.step()
    print(epoch,'Train loss',train_loss/(i+1), 'time=',time.time() - start,'s')
    spconv.checkpoint_save(unet,exp_name,'unet',epoch, use_cuda)

    #if scn.is_power2(epoch):
    if epoch %10 == 0:
        with torch.no_grad():
            unet.eval()
            store=torch.zeros(data.valOffsets[-1],20)
            # scn.forward_pass_multiplyAdd_count=0
            # scn.forward_pass_hidden_states=0
            start = time.time()
            for rep in range(1,1+data.val_reps):
                for i,batch in enumerate(data.val_data_loader):
                    if use_cuda:
                        batch['x'][1]=batch['x'][1].cuda()
                        batch['y']=batch['y'].cuda()
                    predictions=unet(batch['x'])
                    store.index_add_(0,batch['point_ids'],predictions.cpu())
                # print(epoch,rep,'Val MegaMulAdd=',scn.forward_pass_multiplyAdd_count/len(data.val)/1e6, 'MegaHidden',scn.forward_pass_hidden_states/len(data.val)/1e6,'time=',time.time() - start,'s')
                print(epoch,rep,'Val MegaMulAdd=' ,'time=',time.time() - start,'s')
                iou.evaluate(store.max(1)[1].numpy(),data.valLabels)
