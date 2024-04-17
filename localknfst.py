"""
 Local KNFST novelty detection (Bodesheim et al, 2015)
"""

import numpy as np
import utils
import knfst
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import StandardScaler
import faiss
import argparse
from os.path import join as joinpath
from os import environ
from tqdm import tqdm


argsp=argparse.ArgumentParser()
argsp.add_argument("data",type=str,help="name of the data file (without pkl extension) which contains the representations in the 'workspace' directory")
argsp.add_argument("--K",type=int,default=100,help="number of the nearest neighbors of input to retrieve for fitting a local model")
argsp.add_argument("--kernel",type=str,default='linear',help="kernel function (as defined in scikit-learn) for computing pairwise similarities")
argsp.add_argument("--rbf_gamma",type=str,default=0.02,help="gamma parameter of RBF kernel, only used when kernel='rbf'")
argsp.add_argument("--data_second",type=str,default=None,help="the data representations file of the dataset assumed as OOD (only for far-OOD experiments)")
argsp.add_argument("--standardize",action='store_true',help="if enabled, the representation vectors will be standardized using mean and standard deviation")
argsp.add_argument("--not_l2_normal",action='store_true',help="if enabled, the representation vectors will not be normalized in l^2 norm")
argsp.add_argument("--output_file",type=str,default=None,help="name for a text file in the 'output' directory to write the results")
argsp.add_argument("--repeats",type=int,default=0,help="the number of times the method will be repeated using different normal/novel splits (the data files should be stored by appending rX to their name, where X is the number of split)")
argsp.add_argument("--cuda",action='store_true',help="to use GPU for nearest neighbor search")
argsp.add_argument("--comment",type=str,default='')

workpath=environ.get("ood_ws","workspace")   


def local_knfst(x,data_in,knn_model,k):
    """
    Computes novelty scores given a set of query and training data/representations using the Local KNFST method.

    x: the input (query) data matrix, one data point per row
    data_in: the training (in-distribution/normal) data matrix, one data point per row
    knn_model: a k-nearest neighbor model (Faiss object) that performs search on 'data_in'
    k: number of nearest neighbors involved in the local model

    Returns: a list consisting of the scores for each row of 'x'
    """
    
    print('running Local KNFST...')
    scores=[] # scores for each input data
    
    _,nbh_inds=knn_model.search(x,k) # nearest neighbors of each input

    print("knn search done. computing scores...")

    # for each input data point
    for i in tqdm(range(x.shape[0])):

        inds_loc=nbh_inds[i,:]
        d_loc=data_in[inds_loc,:] # the training data in current neighborhood

        kmat_tr=kernel(d_loc,d_loc) # local training kernel matrix
        pr,tr=knfst.learn_oneClassNovelty_knfst(kmat_tr)

        kmat_ts=kernel(d_loc,x[i:i+1,:]) # local test kernel matrix
        d=knfst.test_oneClassNovelty_knfst(pr,tr,kmat_ts)[0]
        d=np.real(d)
        
        scores.append(d)
    
    return scores

def run(data_path,data_second_path): 
    """The main function for running the method"""
    
    l2normal=not args.not_l2_normal
    standard=args.standardize
    faisscuda=args.cuda

    print("loading data...")
    if args.data_second is None:
        data=utils.load_data_reps(path=joinpath(workpath,data_path+".pkl"),l2normal=l2normal)
    else:
        data=utils.load_data_reps(path=joinpath(workpath,data_path+".pkl"),path_out=joinpath(workpath,data_second_path+".pkl"),l2normal=l2normal)

    z_in=data['in_train'].astype('float32')
    z_in_test=data['in_test'].astype('float32')
    z_out_test=data['out_test'].astype('float32')

    ndim=z_in.shape[1]

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
        print('running faiss on GPU.')
        gpures = faiss.StandardGpuResources()
        index=faiss.index_cpu_to_gpu(gpures, 0, index)
    
    print("kernel:",args.kernel)
    if args.kernel=="rbf":
        print("gamma:",args.rbf_gamma)

    print('computing scores on IND test data...')    
    s_test_in=local_knfst(z_in_test,z_in,index,k_neighbor)
    print('computing scores on OOD test data...')   
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