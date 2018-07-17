import torch
from torch import nn , optim
import train_utils as tu
import os
import errno
def main():
    
    inargs = tu.args()
    print("Image Directory = {}\n".format(inargs.indir),
          "Architecture = {}\n".format(inargs.arch),
          "Hidden Units = {}\n".format(inargs.layers),
          "Device = {}\n".format(inargs.device),
          "Learn Rate = {}\n".format(inargs.learn_rate),
          "Epoch = {}\n".format(inargs.epoch),
          "Out Directory = {}\n".format(inargs.out)
          )
    
    if not os.path.exists(inargs.indir):
        raise FileNotFoundError(errno.ENOENT,inargs.indir)
            
    if os.path.exists(inargs.out):
        raise FileExistsError(errno.EEXIST,inargs.out)    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() and inargs.device=='gpu' else "cpu")
    
    data, data_loader = tu.load(inargs.indir)
    model = tu.create_model(inargs.arch,inargs.layers)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr=inargs.learn_rate)
    
    tu.trainer(model,data_loader['trainloader'],criterion,optimizer,data_loader['validloader'],inargs.epoch,device)
    
    state = {
        'epoch': inargs.epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'mapping': data['train_data'].class_to_idx,
        'layers' : inargs.layers,
        'arch'   : inargs.arch
        }
    
    tu.save_checkpoint(state,inargs.out) 
    
    
if __name__ == "__main__":
    main()