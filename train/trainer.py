import torch.optim as optim
from network.nntest import Net
from dataset.screendataset import ScreenDataset
from torch.utils.data import  DataLoader
from dataset.transforms import Rescale
from dataset.transforms import RandomCrop
from dataset.transforms import ToTensor
from torchvision import transforms, utils
import torch.nn as nn
import numpy
import torch
import time
#device
import torch.device
import torch.cuda

from network.fa_resnet import fa_resnet34

class Trainer():
    
    def __init__(self, pretrained=False, lr=0.01, num_epochs=2, max=0):
        self.batchsize = 8
        self.num_epochs=num_epochs
        self.save_path="/Users/sameriksson/temp/model/model.pt"
        #self.net = Net().double()
        if pretrained:
            self.loadModel()
        else:
            self.net = fa_resnet34(pretrained=False, num_classes=4).double()
        print(lr,pretrained)
        #self.criterion = nn.functional.binary_cross_entropy
        self.optimizer = optim.SGD(self.net.parameters(), lr, momentum=0.9)
        transform=transforms.Compose([Rescale(240),ToTensor()])
        self.scrtest = ScreenDataset("/Users/sameriksson/temp/screensgen/test/", transform, max)
        self.scrtrain = ScreenDataset("/Users/sameriksson/temp/screensgen/train/", transform, max)
        
        self.dataloaders = {'train': DataLoader(self.scrtrain, batch_size=self.batchsize,
                        shuffle=True, num_workers=1), 'test': DataLoader(self.scrtest, batch_size=self.batchsize,
                        shuffle=True, num_workers=1)}
        
        #device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        
    def train(self):
        for epoch in range(2):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.dataloaders['train'], 0):
                images = data[0]
                labels = data[1]
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(images)
                loss = nn.functional.binary_cross_entropy(outputs, labels)
                print("loss:",loss)
                loss.backward()
                self.optimizer.step()
                print("Number of images processed:", (i+1)*8)
                if (i+1)/100 == round((i+1)/100):
                    self.saveModel()

        print('Finished Training')
        self.saveModel()
        
    def train_model(self):
        is_inception=False
        since = time.time()
    
        val_acc_history = []
        best_acc = 0.0
        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)
    
            # Each epoch has a training and validation phase
            for phase in ['train', 'test']:
                if phase == 'train':
                    self.net.train()  # Set model to training mode
                    #device
                    self.net.to(self.device)
                else:
                    self.net.eval()   # Set model to evaluate mode
                    #device
                    self.net.to(self.device)
    
                running_loss = 0.0
                running_corrects = 0
    
                # Iterate over data.
                i=0
                for inputs, labels in self.dataloaders[phase]:
                    #device
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    # zero the parameter gradients
                    self.optimizer.zero_grad()
    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if is_inception and phase == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = self.net(inputs)
                            loss1 = nn.functional.binary_cross_entropy(outputs, labels)
                            loss2 = nn.functional.binary_cross_entropy(aux_outputs, labels)
                            loss = loss1 + 0.4*loss2
                        else:
                            outputs = self.net(inputs)
                            loss = nn.functional.binary_cross_entropy(outputs, labels)
    
                        _, preds = torch.max(outputs, 1)
                        _, labela  = torch.max(labels, 1)
    
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
    
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labela)
                    i=i+self.batchsize
                    print("images:", i , " loss:", loss)
    
                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(self.dataloaders[phase].dataset)
    
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    
                # deep copy the model
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    #best_model_wts = copy.deepcopy(model.state_dict())
                    self.saveModel()
                if phase == 'test':
                    val_acc_history.append(epoch_acc)
    
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
       
    def saveModel(self):
        torch.save(self.net.state_dict(), self.save_path)
    
    def loadModel(self):
        self.net = fa_resnet34(pretrained=False, num_classes=4)
        self.net.load_state_dict(torch.load( self.save_path))
        self.net.eval()
        self.net.double()
    
