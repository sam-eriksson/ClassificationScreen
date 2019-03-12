from __future__ import print_function, division
from skimage import  transform
import matplotlib.pyplot as plt
from torch.utils.data import  DataLoader
from torchvision import transforms
from dataset.screendataset import ScreenDataset as ScreenDataset
from dataset.transforms import Rescale
from dataset.transforms import ToTensor
from train.trainer import Trainer


def show_landmarks(screendataset):
    plt.imshow(screendataset.__getitem__(0)['image'])


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--max', type=int, default=100000)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    print("load=",args.load, "lr=", args.lr, "max=", args.max, "epochs=", args.epochs)
    trainer = Trainer(args.load, args.lr, args.epochs, args.max)
    trainer.train_model()
        