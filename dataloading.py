"""
Loading datasets and pretrained feature extractor models
"""

import random
import numpy as np
import torch.hub
import torchvision
import torchvision.transforms as pth_transforms
import dino.dino_loading as dinoloader
from ibot import vit_base
from torchvision.datasets import FakeData



# Paths to the datasets on disk 
# Note: You need to download the CUB-200-2011 manually (https://data.caltech.edu/records/65de6-vp158). Also, please try to manually download the other datasets in case the torchvision downloading fails.
data_path={
    'cub':None,
    'flowers':None,
    'food':None,
    'pcam':None
}

# Transformation applied to all images (based on DINO) given mean and standard deviation
def transform_base(image_mean,image_sd):
    return pth_transforms.Compose([
    pth_transforms.Resize(224, interpolation=3),
    pth_transforms.CenterCrop(224),
    pth_transforms.ToTensor(),
    pth_transforms.Normalize(image_mean,image_sd),
    ])

# Mean and std of ImageNet RGB images
imagenet_mean_sd=[(0.485, 0.456, 0.406),(0.229, 0.224, 0.225)]


# Modifying TorchVision dataset classes to retrieve only selected classes of data

class Cub200(torchvision.datasets.ImageFolder):

    def __init__(self,classes,test_split,transform,shuff_inds,path=data_path['cub'],ratio_train=0.75):

        super().__init__(root=path,transform=transform)

        self.samples=[self.samples[i] for i in shuff_inds]

        ntr=int(len(self.samples)*ratio_train)
        if not test_split:
            self.samples=self.samples[0:ntr]
        else:
            self.samples=self.samples[ntr:]

        news=[]
        for s in self.samples:
            if s[1] in classes:
                news.append(s)
        self.samples=news
        self.imgs=self.samples
        print('CUB data loaded, #samples:',len(self.samples))

class Flowers(torchvision.datasets.Flowers102):

    def __init__(self,classes,split,transform,path=data_path['flowers']):

        super().__init__(root=path,split=split,transform=transform,download=True)

        labelsnw=[]
        imgfilesnw=[]

        for i in range(len(self._labels)):
            if self._labels[i] in classes:
                imgfilesnw.append(self._image_files[i])
                labelsnw.append(self._labels[i])
        
        inds=list(range(len(imgfilesnw)))
        random.shuffle(inds)

        self._image_files=[]
        self._labels=[]
        for i in range(len(imgfilesnw)):
            self._image_files.append(imgfilesnw[inds[i]])
            self._labels.append(labelsnw[inds[i]])

        print('flowers loaded:',len(self._image_files))

class Food(torchvision.datasets.Food101):

    def __init__(self,classes,split,transform,path=data_path['food']):

        super().__init__(root=path,split=split,transform=transform,download=True)

        labelsnw=[]
        imgfilesnw=[]

        for i in range(len(self._labels)):
            if self._labels[i] in classes:
                imgfilesnw.append(self._image_files[i])
                labelsnw.append(self._labels[i])
        
        inds=list(range(len(imgfilesnw)))
        random.shuffle(inds)

        self._image_files=[]
        self._labels=[]
        
        for i in range(len(imgfilesnw)):
            self._image_files.append(imgfilesnw[inds[i]])
            self._labels.append(labelsnw[inds[i]])

        print('foods loaded:',len(self._image_files))

class Pcam(torchvision.datasets.PCAM):

    def __init__(self, class_ind,split = "train",transform= None,target_transform = None,download=True,path=data_path['pcam']):
          
        super().__init__(path,split,transform,target_transform,download)

        self.class_ind=class_ind

        targets_file = self._FILES[self._split]["targets"][0]
        with self.h5py.File(self._base_folder / targets_file) as targets_data:
            y=targets_data['y'][:, 0, 0, 0]
        
        self.data_inds=np.argwhere(y==class_ind)

        print("pcam loaded, class:",class_ind)
    
    def __len__(self):
        return self.data_inds.shape[0]
    
    def __getitem__(self, idx: int):

        x,y= super().__getitem__(self.data_inds[idx,0])

        return x,y


def dataset_and_model(dataset_name:str,classes_in:list,classes_out:list,model_name:str,finetuned_model:str,rand_seed,has_out_train=False):
    """
    Returns a dictionary consisting of in-distribution and OOD train/test data and a pytorch model (either an ImageNet DINO/iBOT pretrained model or a DINO fine-tuned model from checkpoint)
    """

    random.seed(rand_seed)

    if model_name == 'ibot_vitb16':
      model_path = 'checkpoint_ibot.pth'
      state_dict = torch.load(model_path)['state_dict']
      state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
      model = vit_base(patch_size=16, return_all_tokens=True).cuda()
      model.load_state_dict(state_dict, strict=False)
      model.eval()
      for p in model.parameters():
          p.requires_grad = False
    else:
      model=torch.hub.load('dino',model_name,source='local')
    
    print("Backbone model initialized:",model_name)

    if finetuned_model is not None:
        model=dinoloader.load_full_checkpoint(finetuned_model,model,False).backbone
        print("Fine tuned model loaded from",finetuned_model)

    if dataset_name=='cub':

        if finetuned_model is None:
            norm_mean=imagenet_mean_sd[0]
            norm_sd=imagenet_mean_sd[1]
        else:
            norm_mean=(0.48, 0.50, 0.44)
            norm_sd=(0.23,0.23,0.26)

        cubsaminds=list(range(11788))
        random.shuffle(cubsaminds)

        if len(classes_out) == 0:
            in_train=Cub200(classes=range(200),test_split=False,shuff_inds=cubsaminds,transform=transform_base(norm_mean,norm_sd))
            in_test=Cub200(classes=range(200),test_split=True,shuff_inds=cubsaminds,transform=transform_base(norm_mean,norm_sd))
        else:
            in_train=Cub200(classes=classes_in,test_split=False,shuff_inds=cubsaminds,transform=transform_base(norm_mean,norm_sd))
            in_test=Cub200(classes=classes_in,test_split=True,shuff_inds=cubsaminds,transform=transform_base(norm_mean,norm_sd))
            if has_out_train:
                out_train=Cub200(classes=classes_out,test_split=False,shuff_inds=cubsaminds,transform=transform_base(norm_mean,norm_sd))
            out_test=Cub200(classes=classes_out,test_split=True,shuff_inds=cubsaminds,transform=transform_base(norm_mean,norm_sd))
        
    elif dataset_name=='flowers':

        if finetuned_model is None:
            norm_mean=imagenet_mean_sd[0]
            norm_sd=imagenet_mean_sd[1]
        else:
            norm_mean=(0.44, 0.38, 0.28)
            norm_sd=(0.29, 0.24, 0.26)
        
        # Train and test partitions are swapped for Flowers
        if len(classes_out) == 0:
            in_train=Flowers(classes=range(102),split='test',transform=transform_base(norm_mean,norm_sd))
            in_test=Flowers(classes=range(102),split='train',transform=transform_base(norm_mean,norm_sd))
        else:
            in_train=Flowers(classes=classes_in,split='test',transform=transform_base(norm_mean,norm_sd))
            in_test=Flowers(classes=classes_in,split='train',transform=transform_base(norm_mean,norm_sd))
            if has_out_train:
                out_train=Flowers(classes=classes_out,split='test',transform=transform_base(norm_mean,norm_sd))
            out_test=Flowers(classes=classes_out,split='train',transform=transform_base(norm_mean,norm_sd))

    elif dataset_name=='food':

        if finetuned_model is None:
            norm_mean=imagenet_mean_sd[0]
            norm_sd=imagenet_mean_sd[1]
        else:
            norm_mean=(0.54, 0.44, 0.35)
            norm_sd=(0.27, 0.27, 0.28)
        
        if len(classes_out) == 0:
            in_train=Food(classes=range(101),split='train',transform=transform_base(norm_mean,norm_sd))
            in_test=Food(classes=range(101),split='test',transform=transform_base(norm_mean,norm_sd))
        else:
            in_train=Food(classes=classes_in,split='train',transform=transform_base(norm_mean,norm_sd))
            in_test=Food(classes=classes_in,split='test',transform=transform_base(norm_mean,norm_sd))
            if has_out_train:
                out_train=Food(classes=classes_out,split='train',transform=transform_base(norm_mean,norm_sd))
            out_test=Food(classes=classes_out,split='test',transform=transform_base(norm_mean,norm_sd))

    elif dataset_name=='pcam':

        if finetuned_model is None:
            norm_mean=imagenet_mean_sd[0]
            norm_sd=imagenet_mean_sd[1]
        else:
            norm_mean=(0.70, 0.57, 0.70)
            norm_sd=(0.25, 0.30, 0.23)

        in_train=Pcam(class_ind=0,split='train',transform=transform_base(norm_mean,norm_sd))
        in_test=Pcam(class_ind=0,split='test',transform=transform_base(norm_mean,norm_sd))
        if has_out_train:
            out_train=Pcam(class_ind=1,split='train',transform=transform_base(norm_mean,norm_sd))
        out_test=Pcam(class_ind=1,split='test',transform=transform_base(norm_mean,norm_sd))

        if classes_in is not None and classes_in[0]>0:
            raise Exception("For PCAM, the normal data should be class 0")

    elif dataset_name=='noise':
        norm_mean=(0, 0, 0)
        norm_sd=(1.0, 1.0, 1.0)
        in_train=FakeData(3000,num_classes=1,transform=transform_base(norm_mean,norm_sd))
        in_test=FakeData(1000,num_classes=1,transform=transform_base(norm_mean,norm_sd))
        out_test=None


    if not has_out_train:
        data={'in_train':in_train,'in_test':in_test,'out_test':out_test}
    else:
        data={'in_train':in_train,'in_test':in_test,'out_train':out_train,'out_test':out_test}

    return data,model
    
            

