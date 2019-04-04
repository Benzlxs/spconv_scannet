# Options
m = 16 # 16 or 32
residual_blocks=False #True or False
block_reps = 2 #Conv block repetition factor: 1 or 2
dense_shape = [512, 512, 512] # [batch, h, w, l, channels]
features_cc = 3 # rgb or xyz


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
                m.add( spconv.SubMConv3d( nPlanes[0] * (2 if i == 0 else 1), nPlanes[0], 3, bias=False, indice_key="end_pp{}".format(len(nPlanes)))).add(
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
        in0 = self.input0(x)
        # coors = coors.int()[:,[3,2,1,0]] # ordering is [batch, z, y, z]
        coors = in0.get_spatial_locations().int()[:,[3,2,1,0]]
        coord.cuda()
        ret  = spconv.SparseConvTensor(in0.features, coors, dense_shape, data.batch_size)
        x=self.sparseModel(ret)
        temp0 = scn.SparseConvNetTensor(x.features, in0.metadata)
        x = self.out0(temp0)
        x=self.linear(x)
        return x

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
