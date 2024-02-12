##############################################################
# Local KNFST novelty detection (Bodesheim et al, 2015)
##############################################################


import numpy as np
import utils
import knfst
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import StandardScaler
import faiss
import argparse
from os.path import join as joinpath
from os import environ


argsp=argparse.ArgumentParser()
argsp.add_argument("data",type=str)
argsp.add_argument("--K",type=int,default=100)
argsp.add_argument("--kernel",type=str,default='linear')
argsp.add_argument("--rbf_gamma",type=str,default=0.02)
argsp.add_argument("--data_second",type=str,default=None)
argsp.add_argument("--standardize",action='store_true')
argsp.add_argument("--not_l2_normal",action='store_true')
argsp.add_argument("--output_file",type=str,default=None)
argsp.add_argument("--repeats",type=int,default=0)
argsp.add_argument("--truncate_data",type=int,default=None)
argsp.add_argument("--cuda",action='store_true')
argsp.add_argument("--comment",type=str,default='')

workpath=environ.get("ood_ws","workspace")   


def local_knfst(x,data_train,knn_model,k):

    print('local knfst')
    scs=[] # scores for each input data
    
    _,nbh_inds=knn_model.search(x,k) # nearest neighbors of each input

    print("knn search done.")

    # for each input data point
    for i in range(x.shape[0]):

        inds_loc=nbh_inds[i,:]
        d_loc=data_train[inds_loc,:] # the data in current neighborhood

        kmat_tr=kernel(d_loc,d_loc)
        pr,tr=knfst.learn_oneClassNovelty_knfst(kmat_tr)

        kmat_ts=kernel(d_loc,x[i:i+1,:])
        d=knfst.test_oneClassNovelty_knfst(pr,tr,kmat_ts)[0]
        d=np.real(d)
        
        scs.append(d)
    
    return scs

def run(data_path,data_second_path): 

    l2normal=not args.not_l2_normal
    standard=args.standardize
    faisscuda=args.cuda

    print("loading data...")
    if args.data_second is None:
        data=utils.load_data(path=joinpath(workpath,data_path+".pkl"),l2normal=l2normal)
    else:
        data=utils.load_data(path=joinpath(workpath,data_path+".pkl"),path_out=joinpath(workpath,data_second_path+".pkl"),l2normal=l2normal)

    z_in=data['in_train'].astype('float32')
    z_in_test=data['in_test'].astype('float32')
    z_out_test=data['out_test'].astype('float32')

    ndim=z_in.shape[1]

    if args.truncate_data is not None:
        z_in=z_in[0:args.truncate_data,:]
        print('training size', args.truncate_data)

    if standard:
        sst=StandardScaler().fit(z_in)
        z_in=sst.transform(z_in)
        z_in_test=sst.transform(z_in_test)
        z_out_test=sst.transform(z_out_test)
        print('standardized')

    k_neighbor=args.K
    print('initializing faiss...','k=',k_neighbor,'d=',ndim)
    index= faiss.IndexFlatL2(ndim)
    index.add(z_in)

    if faisscuda:
        print('faiss gpu...')
        gpures = faiss.StandardGpuResources()
        index=faiss.index_cpu_to_gpu(gpures, 0, index)
    
    print("kernel",args.kernel)
    if args.kernel=="rbf":
        print("gamma",args.rbf_gamma)
    print('computing scores...')

    s_test_in=local_knfst(z_in_test,z_in,index,k_neighbor)

    s_test_out=local_knfst(z_out_test,z_in,index,k_neighbor)

    onehots = np.array([1] * len(s_test_out) + [0] * len(s_test_in))
    scores = np.concatenate([np.array(s_test_out), np.array(s_test_in)], axis=0)

    res=utils.roc(onehots,scores)
    print("AUROC:",res['auroc'])
    print("FPR@95TPR:",res['fpr95tpr'])
    return res


if __name__=='__main__':

    args=argsp.parse_args()

    if args.kernel=='linear':
        kernel= lambda x,y : pairwise_kernels(x,y,metric='linear')
    elif args.kernel=='rbf':
        kernel= lambda x,y : pairwise_kernels(x,y,metric='rbf',gamma=args.rbf_gamma) 

    if args.repeats==0:
        res=run(args.data,args.data_second)
        logger=utils.logwriter("Local KNFST",args.output_file)
        logger.write(str(args.__dict__))
        logger.write("AUROC: {:.2f}%  FPR@95TPR: {:.2f}%".format(res['auroc']*100,res['fpr95tpr']*100))
    else:
        utils.repeated_run(run,"Local KNFST",args)
    print(" ")