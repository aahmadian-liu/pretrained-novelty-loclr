"""
Additional plotting and statistical tools (e.g., for intrinsic dimensionality)
"""

# "scikit-dimension" library is required for local intrinsic dimensionality estimation

from skdim.id import TLE
import numpy as np
import matplotlib.pyplot as plot
import utils 
from sklearn.preprocessing import StandardScaler
import argparse
from os.path import join as joinpath
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from os import environ
from sklearn.linear_model import LogisticRegression


argsp=argparse.ArgumentParser()
argsp.add_argument("command",type=str)
argsp.add_argument("data",type=str)
argsp.add_argument("--save_name",type=str,default="")
argsp.add_argument("--captions",type=str,default="")
argsp.add_argument("--k_neighbor",type=int,default=0)

args=argsp.parse_args()
workpath=environ.get("ood_ws","workspace")

l2normal=True
standard=True

# Histogram of Local Intrinsic Dimensionality (using TLE method) on IN data
def plot_id_tle(inputs):

    plot.figure()
    maxsize=10000
    caps=str.split(args.captions,",")

    for i,path in enumerate(str.split(inputs,",")):

        data=utils.load_data_reps(path=joinpath(workpath,path+".pkl"),l2normal=l2normal)['in_train']
        if data.shape[0]>maxsize:
            data=data[0:maxsize,:]
        if standard:
            sst=StandardScaler().fit(data)
            data=sst.transform(data)
            print("standardized")

        print("Estimating local ID using TLE...")
        estim=TLE()
        id=estim.fit_transform_pw(data,n_neighbors=50)

        print(path,":")
        print("ID mean:", id.mean())
        print("ID std:",id.std())

        plot.subplot(2,2,i+1)
        plot.hist(id,label=caps[i],range=(0,100))
        plot.legend()
        
    plot.savefig("output/"+args.save_name+".png")
    print("Plot saved.")

# Histogram of coefficients of a basic logistic regression model
def logisticreg(input):

    data=utils.load_data_reps(path=joinpath(workpath,input+".pkl"),l2normal=l2normal)
    data_l=utils.load_data_class(path=joinpath(workpath,input+".pkl"))

    z_train=data['in_train']
    z_test=data['in_test']
    l_train=data_l['in_train']
    l_test=data_l['in_test']
    
    # Training/test data of IN and OOD are merged (a binary classification in case of PCAM)
    z_train=np.concatenate([z_train,data['out_train']])
    z_test=np.concatenate([z_test,data['out_test']])
    l_train=np.concatenate([l_train,data_l['out_train']])
    l_test=np.concatenate([l_test,data_l['out_test']])
    if standard:
        sst=StandardScaler().fit(z_train)
        z_train=sst.transform(z_train)
        z_test=sst.transform(z_test)
        print("standardized")

    print("fitting lgr...")
    lgr=LogisticRegression(penalty='none')
    lgr.fit(z_train,l_train)

    acc=lgr.score(z_test,l_test)
    print("Test accuracy:",acc)

    cofs=np.abs(lgr.coef_[0,:])
    cofs=np.sort(cofs)
    plot.figure()
    plot.rcParams['font.size']=20
    plot.hist(cofs,bins=20)
    plot.savefig("output/"+args.save_name+"_hist.png",dpi=600)
    plot.figure()
    plot.bar(np.arange(0,768),cofs)
    plot.savefig("output/"+args.save_name+"_bar.png",dpi=600)           
    print("Plot saved.")

# Ordinary KNN classification on IN data
def knnclassify(input):

    data=utils.load_data_reps(path=joinpath(workpath,input+".pkl"),l2normal=l2normal)
    data_l=utils.load_data_class(path=joinpath(workpath,input+".pkl"))

    z_train=data['in_train']
    z_test=data['in_test']
    l_train=data_l['in_train']
    l_test=data_l['in_test']

    if standard:
        sst=StandardScaler().fit(z_train)
        z_train=sst.transform(z_train)
        z_test=sst.transform(z_test)
        print("standardized")

    print("fitting knn...")
    knn=KNeighborsClassifier(args.k_neighbor)
    knn.fit(z_train,l_train)

    acc=knn.score(z_test,l_test)
    print("Test accuracy:",acc)



if args.command=="local_id":
    assert(len(args.save_name)>0 and len(args.captions)>0)
    plot_id_tle(args.data)
if args.command=="knn_classify":
    assert(args.k_neighbor>0)
    knnclassify(args.data)
if args.command=="log_cof":
    assert(len(args.save_name)>0)
    logisticreg(args.data)