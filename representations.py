##############################################################
# Extracting representation of images using a pretrained model
##############################################################

#Note: please download the datasets if necessary, and set their paths in 'datasets.py'


import argparse
import datasets
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import pickle
import json
from os.path import join as joinpath
from os import environ
import datetime
import platform
import random


argsp=argparse.ArgumentParser()
argsp.add_argument("dataset",type=str,help="Name of dataset from which representations are obtained. One of {'cub','flowers','food','pcam'}")
argsp.add_argument("save_name",type=str,help="Name to save the output in workspace directory (as .pkl and .json files)")
argsp.add_argument("--classes_in",type=str,required=True,
                   help="Classes which will used as in-distribtuion data. Possible formats: 1. 'i1:i2' like a python range 2. 'rand_i1:i2' shuffles the class numbers before applying range 3. comma separated")
argsp.add_argument("--classes_out",type=str,required=False,default=None,help="Classes which will used as OOD data. Same format as classes_in")
argsp.add_argument("--n_samples_in",type=str,required=False,default='auto',
                   help="Number of images processed from in-distribution data. Options: 1. 'auto' uses all training IN data, and makes the size of test sets equal for IN and OOD data 2. 'n1,n2' uses n1 for training and n2 for test")
argsp.add_argument("--n_samples_out",type=str,required=False,default='auto', help="Number of images processed from OOD data. Same options as n_samples_in")
argsp.add_argument("--finetuned_path",type=str,required=False,default=None, help="Path to the pytorch model file, only when using a feature extractor model fine tuned on IN data")
argsp.add_argument("--include_train_out",action='store_true', help="If OOD training data should be included")
argsp.add_argument("--model_name",type=str,required=False,default='dino_vitb16', help="Name of the pretrained feature extractor model ('dino_vitb16','dino_resnet50','ibot_vitb16')")
argsp.add_argument("--vit_rep_type",type=str,required=False,default='cls',help="Token type when using vision transformer")
argsp.add_argument("--batch_size",type=int,required=False,default=32)
argsp.add_argument("--randseed",type=int,required=False,default=0, help="Random seed")
argsp.add_argument("--cuda",action='store_true')
argsp.add_argument("--comment",type=str,required=False,default='')

args=argsp.parse_args()

assert(args.dataset in {'cub','flowers','food','pcam'})

workpath=environ.get("ood_ws","workspace")
nclass={'cub':200,'flowers':102,'food':101,'pcam':1} # total number of classes
random.seed(args.randseed)
permlist=list(range(0,nclass[args.dataset])) # a random list for shuffling
random.shuffle(permlist)

def parsenums(arg:str):
    
    if arg.startswith("rand_"):
        arg=arg.replace("rand_","")
        inds=[int(k) for k in arg.split(':')]
        return permlist[inds[0]:inds[1]]
    elif ':' in arg:
        inds=[int(k) for k in arg.split(':')]
        if len(inds)==2:
            return range(inds[0],inds[1])
        else:
            return range(inds[0],inds[1],inds[2])
    else:
        inds=[int(k) for k in arg.split(',')]
        return inds

# Setting the list of IN and OOD classes
classes_in=parsenums(args.classes_in)
if args.classes_out is not None:
    classes_out=parsenums(args.classes_out)
    only_in=False
else:
    classes_out=[]
    only_in=True

if args.model_name=='dino_vitb16':
    dim_tok=768
    n_head=12
elif args.model_name=='dino_resnet50':
    dim_tok=2048
    n_head=-1
elif args.model_name=='ibot_vitb16':
    dim_tok=768
    n_head=12 

# Loading data and feature extractor model
data,model=datasets.get_dataset_model(args.dataset,classes_in,classes_out,model_name=args.model_name,finetuned_model=args.finetuned_path,rand_seed=args.randseed,has_out_train=args.include_train_out)

# Setting the number of samples in each set
n_samples=dict()
if args.n_samples_in=='auto' and not only_in:
    numtest=min(len(data['out_test']),len(data['in_test']))
    n_samples['in_test']=numtest
    n_samples['out_test']=numtest
    n_samples['in_train']=len(data['in_train'])
    if args.include_train_out:
        n_samples['out_train']=len(data['out_train'])
elif args.n_samples_in=='auto':
    n_samples['in_train']=len(data['in_train'])
    n_samples['in_test']=len(data['in_test'])
else:
    num_in=parsenums(args.n_samples_in)
    n_samples['in_train']=num_in[0]
    n_samples['in_test']=num_in[1]
    if not only_in:
        num_out=parsenums(args.n_samples_out)
        n_samples['out_train']=num_out[0]
        n_samples['out_test']=num_out[1]
if only_in:
    n_samples['out_train']=0
    n_samples['out_test']=0

print("using number of samples:",n_samples['in_train'],"-",n_samples['in_test'],"-",n_samples['out_test'])

# Preparing the model for test time
model.eval()
if args.cuda:
    model=model.cuda()

# Main dictionaries to save outputs
reps=dict()
labels=dict()
counts=dict()

# Main processing
for d in data.keys():

    if n_samples[d]==0:
        continue

    print("Processing:",d)

    if args.vit_rep_type=='cls':
        reps[d]=np.zeros([n_samples[d],dim_tok])
    #if args.vit_rep_type=='heads':
    #    reps[d]=np.zeros([n_samples[d],n_head,dim_tok//n_head])

    labels[d] = np.zeros([n_samples[d]],int)

    dataloader=DataLoader(data[d],args.batch_size,num_workers=2)
    i=0
    counts[d]=0

    with torch.no_grad():
        for x,y in dataloader:

            l=x.shape[0]
            if args.cuda:
                x=x.cuda()

            if i+l>n_samples[d]:
                l=n_samples[d]-i
                x=x[0:l,:]
                y=y[0:l]
            
            if args.vit_rep_type=='cls' or args.model_name=='dino_resnet50':
                
                z=model(x)
                if args.model_name=='ibot_vitb16':
                  z = z[:,0]
                reps[d][i:i+l,:]=z.cpu().numpy()
                labels[d][i:i+l]=y.numpy()
            else:
                raise Exception("not implemented yet.")

            i+=l
            print(i,"/",n_samples[d])
            counts[d]=i

            if i>=n_samples[d]:
                break

fname=args.save_name

print("writing to file...")

# Saving data representations and labels to disk
with open(joinpath(workpath,fname+'.pkl'),'wb') as f:
    fdic={'reps':reps,'labels':labels,'meta':str(vars(args))}
    #if calbackstats:
    #    fdic['back_stats_norm']=(meanback_norm,varback_norm)
    pickle.dump(fdic,f)

# Saving some meta info as JSON file
with open(joinpath(workpath,fname+'.json'),'w') as f:
    mdic={'dataset':args.dataset,'model':args.model_name,'class_in':list(classes_in),'class_out':list(classes_out),'n_class_in':len(classes_in),'n_class_out':len(classes_out),
          'n_in_train':counts.get('in_train'),'n_in_test':counts.get('in_test'),'n_out_test':counts.get('out_test'),'n_out_train':counts.get('out_train'),
          'in_train_shape':reps['in_train'].shape,'uses_finetuned':(args.finetuned_path is not None),'computer':platform.node(),'time':str(datetime.datetime.now()),'comment':args.comment,
          'args':vars(args)
          }
    json.dump(mdic,f)


print("Representations and labels saved successfully to", fname+".pkl", "with dataset size",str(counts), ",",len(classes_in),"IN and",len(classes_out),"OUT classes")
            
if len(set(classes_in).intersection(set(classes_out)))>0:
    print("Warning: overlap between IN and OUT classes")
if np.max(labels['in_train'])<np.max(classes_in):
    print("Warning: no training samples from some IN classes")
