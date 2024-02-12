##############################################################
# Some auxiliary plotting and summary tools (e.g., for intrinsic dimensionality)
##############################################################


import numpy as np
import matplotlib.pyplot as plot
import utils 
from sklearn.preprocessing import StandardScaler
import argparse
from os.path import join as joinpath
from skdim.id import TLE
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


def plot_id_tle(inputs):

    plot.figure()
    maxsize=10000
    caps=str.split(args.captions,",")

    for i,path in enumerate(str.split(inputs,",")):

        data=utils.load_data(path=joinpath(workpath,path+".pkl"),l2normal=l2normal)['in_train']
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

def plot_nndis(inputs):

    plot.figure()
    maxsize=10000
    caps=str.split(args.captions,",")

    for i,path in enumerate(str.split(inputs,",")):

        data=utils.load_data(path=joinpath(workpath,path+".pkl"),l2normal=l2normal)['in_train']
        if data.shape[0]>maxsize:
            data=data[0:maxsize,:]
        if standard:
            sst=StandardScaler().fit(data)
            data=sst.transform(data)
            print("standardized")

        print("obtaining distance to nearest neighbor...")
        nn=NearestNeighbors(n_neighbors=2)
        nn.fit(data)
        dis,_=nn.kneighbors(data)
        dis=dis[:,1]

        print(path,":")
        print("dis to nn mean:", dis.mean())
        print("dis to nn std:",dis.std())

        plot.hist(dis,label=caps[i],alpha=0.7)
        
    plot.legend()  
    plot.savefig("output/"+args.save_name+".png")
    print("Plot saved.")

def plot_marg_density(input):

    data=utils.load_data(path=joinpath(workpath,input+".pkl"),l2normal=l2normal)
    data=data['in_train']
    data=data[0:30000,:]

    sst=StandardScaler().fit(data)
    data=sst.transform(data)

    neval=200
    ndims=data.shape[1]
    dens=np.zeros([neval,ndims])

    for i in range(ndims):
        kd=KernelDensity(bandwidth=0.2,kernel='gaussian')
        kd=kd.fit(data[:,i:i+1])
        u=np.flip(np.linspace(-1.5,1.5,neval))
        dens[:,i]=kd.score_samples(u[:,None])
    
    plot.matshow(dens,cmap='hot')
    plot.yticks([0,50,100,150,199],['1.5','0.75','0','-0.75','-1.5'])
    plot.savefig("output/"+args.save_name+".png",dpi=600)
    print("Plot saved.")

def knnclassify(input):

    data=utils.load_data(path=joinpath(workpath,input+".pkl"),l2normal=l2normal)
    data_l=utils.load_data(path=joinpath(workpath,input+".pkl"),field='class')
    incout= 'out_train' in data.keys()

    z_train=data['in_train']
    z_test=data['in_test']
    l_train=data_l['in_train']
    l_test=data_l['in_test']
    print(data.keys())
    if incout:
        z_train=np.concatenate([z_train,data['out_train']])
        z_test=np.concatenate([z_test,data['out_test']])
        l_train=np.concatenate([l_train,data_l['out_train']])
        l_test=np.concatenate([l_test,data_l['out_test']])
        print("concateneted OOD data")
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

def writecsv(input):

    data=utils.load_data(path=joinpath(workpath,input+".pkl"),l2normal=l2normal)

    z_in_train=data['in_train']
    z_in_test=data['in_test']
    z_out_test=data['out_test']

    if standard:
        sst=StandardScaler().fit(z_in_train)
        z_in_train=sst.transform(z_in_train)
        z_out_test=sst.transform(z_out_test)
        z_in_test=sst.transform(z_in_test)
        print("standardized")

    np.savetxt("output/d_in_train.csv",z_in_train, delimiter=",",fmt='%1.6f')
    np.savetxt("output/d_in_test.csv",z_in_test, delimiter=",",fmt='%1.6f')
    np.savetxt("output/d_out_test.csv",z_out_test, delimiter=",",fmt='%1.6f')
    print("csv files saved.")

def logisticreg(input):

    data=utils.load_data(path=joinpath(workpath,input+".pkl"),l2normal=l2normal)
    data_l=utils.load_data(path=joinpath(workpath,input+".pkl"),field='class')

    z_train=data['in_train']
    z_test=data['in_test']
    l_train=data_l['in_train']
    l_test=data_l['in_test']
    
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


if args.command=="local_id":
    assert(len(args.save_name)>0 and len(args.captions)>0)
    plot_id_tle(args.data)
if args.command=="marg_density":
    assert(len(args.save_name)>0)
    plot_marg_density(args.data)
if args.command=="knn_classify":
    assert(args.k_neighbor>0)
    knnclassify(args.data)
if args.command=="write_csv":
    writecsv(args.data)
if args.command=="nndis":
    assert(len(args.save_name)>0 and len(args.captions)>0)
    plot_nndis(args.data)
if args.command=="logreg":
    assert(len(args.save_name)>0)
    logisticreg(args.data)