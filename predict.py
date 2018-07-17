import torch
from train_utils import create_model
import predict_utils as pu
import os
import errno


def main() :

    inargs = pu.args()

    device = torch.device("cuda:0" if torch.cuda.is_available() and inargs.device=='gpu' else "cpu")

    print("Flower Image = {}\n".format(inargs.img),
          "Checkpoint = {}\n".format(inargs.checkpoint),
          "Map File = {}\n".format(inargs.map),
          "Topk = {}\n".format(inargs.topk),
          "Device = {}\n".format(device)
         )

    if not os.path.exists(inargs.checkpoint):
        raise FileNotFoundError(errno.ENOENT,inargs.checkpoint)
    if not os.path.exists(inargs.img):
        raise FileNotFoundError(errno.ENOENT, inargs.img)

    model = pu.load_checkpoint(inargs.checkpoint)
    probs,labels,classes,map = pu.predict(inargs.img,model,inargs.map,device,inargs.topk)

    for i in range(inargs.topk):
        print("{}- Flower Name: {}, Probability:{:.2f}".format(i+1,labels[i],probs[i]))

    pu.display(inargs.img, probs,classes,map)

if __name__ == "__main__":
    main()
