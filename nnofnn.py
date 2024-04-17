"""
 Nearest Neighbors of Neighbors novelty detection (based on [Nizan and Tal, 2024]) 
"""

import numpy as np
import utils
from sklearn.preprocessing import StandardScaler
import faiss
import argparse
from os.path import join as joinpath
from os import environ
from tqdm import tqdm


# parameters
argsp=argparse.ArgumentParser()
argsp.add_argument("data",type=str,help="name of the data file (without pkl extension) which contains the representations in the 'workspace' directory")
argsp.add_argument("K_prime",type=int,help="hyperparameter specifying the number of nearest neighbors of the nearest neighbor")
argsp.add_argument("--K",type=int,default=1,help="number of nearest neighbors of input to consider; currently only 1 is supported")
argsp.add_argument("--S",type=int,default=1,help="number of dimensions in each partition of the PCA covariance matrix; currently only diagonal covariance (S=1) supported")
argsp.add_argument("--data_second",type=str,default=None,help="the data representations file of the dataset assumed as OOD (only for far-OOD experiments)")
argsp.add_argument("--standardize",action='store_true',help="if enabled, the representation vectors will be standardized using mean and standard deviation")
argsp.add_argument("--not_l2_normal",action='store_true',help="if enabled, the representation vectors will not be normalized in l^2 norm")
argsp.add_argument("--output_file",type=str,default=None,help="name for a text file in the 'output' directory to write the results")
argsp.add_argument("--repeats",type=int,default=0,help="the number of times the method will be repeated using different normal/novel splits (the data files should be stored by appending rX to their name, where X is the number of split)")
argsp.add_argument("--cuda",action='store_true',help="to use GPU for nearest neighbor search")
argsp.add_argument("--comment",type=str,default='')

workpath=environ.get("ood_ws","workspace")

def knnn_k1_s1(x,data_in,knn_model,k_prime):
    """Computes novelty scores given a set of query and training data/representations using the k-NNN method assuming K=1 and S=1
    
    Args:
    x: the input (query) data matrix, one data point per row
    data_in: the training (in-distribution/normal) data matrix, one data point per row
    knn_model: a k-nearest neighbor model (Faiss object) that performs search on 'data_in'
    k_prime: the number of nearest neighbors of nearest neighbor to consider for local PCA (called #neighbors-neighbors in original paper)

    Returns: a list consisting of the scores for each row of 'x'
    """

    print('running knnn (k=1,s=1)...')
    scs=[] # scores for each input data
    
    _,nn_inds=knn_model.search(x,1) # nearest neighbor of each input
    # nearest neighbors of neighbor
    _,nbh_inds=knn_model.search(data_in[nn_inds[:,0],:],k_prime)

    print("knn search done. computing scores...")

    # for each input data point
    for i in tqdm(range(x.shape[0])):

        inds_loc=nbh_inds[i,:]
        d_loc=data_in[inds_loc,:] # the data in current neighborhood

        nn=data_in[nn_inds[i,0],:] # nearest neighbor of current input

        center=nn
           
        cov_loc=np.std(d_loc,axis=0)
        
        # inverse of local covariance   
        eps=1e-5
        cov_loc_inv=1.0/(cov_loc+eps)
        # the main score value
        d=(np.abs(x[i,:]-center)*cov_loc_inv).sum()

        scs.append(d)
    

    return scs

def run(data_path,data_second_path): 
    """The main function for running the method"""

    l2normal=not args.not_l2_normal
    standard=args.standardize
    faisscuda=args.cuda

    if args.K!=1 or args.S!=1:
        raise(Exception("This implementation of k-NNN is only for K=1 and S=1"))

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

    k_neighbor=args.K_prime
    print('initializing faiss...','k=',k_neighbor,'d=',ndim)
    index= faiss.IndexFlatL2(ndim)
    index.add(z_in)

    if faisscuda:
        print('faiss running on GPU....')
        gpures = faiss.StandardGpuResources()
        index=faiss.index_cpu_to_gpu(gpures, 0, index)

    print('computing scores on IND test data...')
    s_test_in=knnn_k1_s1(z_in_test,z_in,index,k_neighbor)
    print('computing scores on OOD test data...')
    s_test_out=knnn_k1_s1(z_out_test,z_in,index,k_neighbor)

    onehots = np.array([1] * len(s_test_out) + [0] * len(s_test_in))
    scores = np.concatenate([np.array(s_test_out), np.array(s_test_in)], axis=0)

    res=utils.roc(onehots,scores)
    print("AUROC:",res['auroc'])
    print("FPR@95TPR:",res['fpr95tpr'])
    return res


if __name__=='__main__':
    args=argsp.parse_args()
    if args.repeats==0:
        res=run(args.data,args.data_second)
        logger=utils.logwriter("k-NNN (K=1,S=1)",args.output_file)
        logger.write(str(args.__dict__))
        logger.write("AUROC: {:.2f}%  FPR@95TPR: {:.2f}%".format(res['auroc']*100,res['fpr95tpr']*100))
    else:
        utils.repeated_run(run,"k-NNN (K=1,S=1)",args)
    print(" ")