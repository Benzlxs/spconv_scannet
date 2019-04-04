# Copyright



import torch, glob, os



def checkpoint_restore(model,exp_name,name2,use_cuda=True,epoch=0):
    if use_cuda:
        model.cpu()
    if epoch>0:
        f=exp_name+'-%09d-'%epoch+name2+'.pth'
        assert os.path.isfile(f)
        print('Restore from ' + f)
        model.load_state_dict(torch.load(f))
    else:
        f=sorted(glob.glob(exp_name+'-*-'+name2+'.pth'))
        if len(f)>0:
            f=f[-1]
            print('Restore from ' + f)
            model.load_state_dict(torch.load(f))
            epoch=int(f[len(exp_name)+1:-len(name2)-5])
    if use_cuda:
        model.cuda()
    return epoch+1


def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)

def checkpoint_save(model,exp_name,name2,epoch, use_cuda=True):
    f=exp_name+'-%09d-'%epoch+name2+'.pth'
    model.cpu()
    torch.save(model.state_dict(),f)
    if use_cuda:
        model.cuda()
    #remove previous checkpoints unless they are a power of 2 to save disk space
    epoch=epoch-1
    f=exp_name+'-%09d-'%epoch+name2+'.pth'
    if os.path.isfile(f):
        if not is_power2(epoch):
            os.remove(f)
