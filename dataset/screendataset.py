from torch.utils.data import Dataset, DataLoader
from os import listdir
from os.path import isfile, join
from os import path
from skimage import io
from skimage.color import rgba2rgb
from skimage import data
import json
import numpy as np

from PIL import Image

class ScreenDataset(Dataset):
    MENUGROUP = 1
    IMAGE=2
    IMAGEBUTTON=3
    MENUITM=4
    
    def __init__(self, directory, transform=None, max=0):
        self.max = max
        self.directory = directory
        self.transform = transform
        self.listofjson = self.listOfJsonFiles()

    def __len__(self):
        length = len(self.listofjson)
        if (self.max>length): 
            return len(self.listofjson)
        else:
            return self.max

    #print(trn_ds2.__getitem__(1)[1])
    #(array([ 38.,  87., 118., 127.,  48.,  42., 203., 186.], dtype=float32), array([14, 12]))
    def __getitem__(self, idx):
        jsonfilename = self.listofjson[idx]
        imagefilename = jsonfilename[0:len(jsonfilename)-5] + ".png"
        img_name = path.join(self.directory,
                             imagefilename )
        imagergba = io.imread(img_name)
        #convert rgba to rgb
        image = rgba2rgb(imagergba)
        #
        c,t = self.loadClassAndPosition(jsonfilename)
        ilc = image,[c,t]
        #
        if self.transform:
            ilc = self.transform(ilc)
        #return ilc
        return ilc[0], ilc[1][1]
    
    def listOfJsonFiles(self):        
        listoffiles= [f for f in listdir(self.directory) if isfile(join(self.directory, f))]
        jsonlist = list(filter(lambda item: item.endswith(".json") , listoffiles))
        jsonlist.sort(key = lambda x: int( x.split('.')[0]) )
        return jsonlist
    
    def loadJsonFile(self, jsonfilename):
        filename = jsonfilename    
        s = self.load(filename)
        return s
    
    def load(self, filename):
        #print ("load")
        
        with open(self.directory+ filename, 'r') as myfile:
            st= myfile.read()
        s = json.loads(st)
        return s
    
    def loadClassAndPosition(self, jsonfilename):
        returnvalue = []
        t = []
        c = []
        dict = self.loadJsonFile(jsonfilename)
        composites = dict['SCREEN']
        for item in composites:
            scrtype = 0
            two = item['COMPOSITE']
            if 'LABEL' in two: 
                label = two['LABEL']
                word = label['WORD']
                text = word['VALUE']
            if 'IMAGE' in two:
                scritem = two['IMAGE']
                scrtype = ScreenDataset.IMAGE
            elif 'IMAGEBUTTON' in two:
                scritem = two['IMAGEBUTTON']
                scrtype = ScreenDataset.IMAGEBUTTON
            elif 'MENU' in two:
                scritem = two['MENU']
                scrtype = ScreenDataset.MENU
            elif 'MENUGROUP' in two :
                scritem = two['MENUGROUP']
                scrtype = ScreenDataset.MENUGROUP
             
            coordinates = scritem['COORDINATES']
            x= coordinates['X']
            y= coordinates['Y']
            x1= coordinates['X1']
            y1= coordinates['Y1']
            x2= coordinates['X2']
            y2= coordinates['Y2']
            x3= coordinates['X3']
            y3= coordinates['Y3']
            t.append(scrtype)
            c.append([x,y,x2,y2])
        return np.asarray(c),self.convert(t)
    
    def convert(self, labels):
        list=[0.,0.,0.,0.]
        for label in labels:
            if label==1:
                list[0]=1.
            elif label==2:
                list[1]=1.  
            elif label==3:
                list[2]=1.
            elif label==4:
                list[3]=1.               
        return np.asarray(list)