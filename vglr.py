
###################################################################
# Implementation of the paper 'Unsupervised Novelty Detection in Pretrained Representation Space with Locally Adapted Likelihood Ratio' (AISTATS 2024)
# (C) A.Ahmadian, Y.Ding, G.Eilertsen, F.Lindsten
# MIT License
###################################################################

"""
The proposed Voronoi-Gaussian with Likelihood Ratio (VGLR) novelty detection 
"""


import numpy as np
import utils
from sklearn.preprocessing import StandardScaler
from numpy.linalg import inv,det
import faiss
import argparse
from os.path import join as joinpath
from os import environ
from tqdm import tqdm


# Parameters
argsp=argparse.ArgumentParser()
argsp.add_argument("data",type=str, help="name of the data file (without pkl extension) which contains the representations in the 'workspace' directory")
argsp.add_argument("K_prime",type=int, help="the k' hyperparameter (number of neighbors of nearest neighbor)")
argsp.add_argument("--data_second",type=str,default=None, help="the data representations file of the dataset assumed as OOD (only for far-OOD experiments)")
argsp.add_argument("--Lambda",type=float,default=1.0, help="covariance regularization hyperparameter (always 1 in our experiments)")
argsp.add_argument("--flags",default="ratio", 
                   help="the flags for algorithm options. 'ratio': use likelihood ratio with background model, 'noloc':no local adaptation")
argsp.add_argument("--not_standardize",action='store_true', help="if enabled, the representation vectors will not be standardized using mean and standard deviation")
argsp.add_argument("--not_l2_normal",action='store_true', help="if enabled, the representation vectors will not be normalized in l^2 norm")
argsp.add_argument("--background_model",type=str,default=None, help="used to load the background mean and covariance from file (e.g., ImageNet statistics) instead of computing them on training data")
argsp.add_argument("--output_file",type=str,default=None,help="name for a text file in the 'output' directory to write the results")
argsp.add_argument("--repeats",type=int,default=0, help="the number of times the method will be repeated using different normal/novel splits (the data files should be stored by appending rX to their name, where X is the number of split)")
argsp.add_argument("--cuda",action='store_true', help="to use GPU for nearest neighbor search")
argsp.add_argument("--comment",type=str,default='')

workpath=environ.get("ood_ws","workspace")

def vglr(x, data_in, knn_model, k_prime, prior_cov, c_lambda, k=1, back_mean_cov=None):
    """
    Computes novelty scores given a set of query and training data/representations using VG/VGLR method.

    Args:
    x: the input (query) data matrix, one data point per row
    data_in: the training (in-distribution/normal) data matrix, one data point per row
    knn_model: a k-nearest neighbor model (Faiss object) that performs search on 'data_in'
    k_prime: the number of nearest neighbors of nearest neighbor to consider for local covariance estimation
    prior_cov: a prior covariance matrix for regularizing the local covariance (not used in the paper)
    c_lambda: a coefficient for regularizing the local covariance (always equal to 1 in the paper)
    k: which neighbor to retrieve for fitting the local model (always the nearest neighbor in the paper, k=1)
    back_mean_cov: a tuple containing the mean and covariance of the assumed background model for likelihood ratio score (if set to None, likelihood ratio will not be applied)
    
    Returns: a list consisting of the scores for each row of 'x'
    """

    print('running VGLR...')
    scores=[] # novelty scores for each input data

    if len(prior_cov.shape)==2 and prior_cov.shape[1]>1:
        matglob=True # when global covariance matrix is not diagonal
    else:
        matglob=False

    if back_mean_cov is not None:
        if len(back_mean_cov[1].shape)==1:
            invbackcov=np.diag(1.0/back_mean_cov[1]) # inverse of diagonal background covariance
        else:
            invbackcov=inv(back_mean_cov[1]) # inverse of full background covariance
        backmean=back_mean_cov[0] 
        ratio=True # likelihood ratio mode enabled
    else:
        ratio=False
 
    # finding nearest (or k'th) neighbor of each input
    _,nn_inds=knn_model.search(x,k)
    nn_inds=nn_inds[:,-1]

    # finding nearest neighbors of inputs' nearest neighbors, used for obtaining local covariance matrices
    _,nbh_inds=knn_model.search(data_in[nn_inds[:],:],k_prime)
    
    print("knn search done. computing scores...")

    # for each input data point
    for i in tqdm(range(x.shape[0])):

        inds_loc=nbh_inds[i,:]
        d_loc=data_in[inds_loc,:] # the data in current neighborhood

        nn=data_in[nn_inds[i],:] # nearest neighbor of current input

        # use the nearest neighbor as center (mean of Gaussian distribution)
        center=nn
        
        # obtain the local diagonal matrix as a moment around the center
        cov_loc_0=np.mean((d_loc-center)**2,axis=0)

        if matglob:
            cov_loc_0=np.diag(cov_loc_0)

        cov_loc = c_lambda*cov_loc_0 + (1-c_lambda)*prior_cov  # regularization of the local covariance
    
        eps=1e-5
        
        # the main score value
        if not matglob:
            cov_loc_inv=1.0/(cov_loc+eps)
            d=(cov_loc_inv*((x[i,:]-center)**2)).sum()
        else:
            cov_loc_inv=inv(cov_loc)
            d=(x[i,:]-center) @ cov_loc_inv @ (x[i,:]-center).T

        # adding log-det of Gaussian density function
        if not matglob:
            d+=np.log(cov_loc+eps).sum()
        else:
            d+=np.log(det(cov_loc)+eps)

        if ratio:
            db=(x[i:i+1,:]-backmean) @ invbackcov @ (x[i:i+1,:]-backmean).T
            db=db[0]
            d=d-db # subtracting the score given by background Gaussian model

        scores.append(d)
    
    print('VGLR scores computed.')

    return scores


def run(data_path,data_second_path):
    """The main function for running the method"""

    l2normal=not args.not_l2_normal
    standard=not args.not_standardize
    faisscuda=args.cuda

    # loading representation vectors of IND and OOD data from file
    print("loading data...")
    if args.data_second is None:
        data=utils.load_data_reps(path=joinpath(workpath,data_path+".pkl"),l2normal=l2normal)
    else:
        data=utils.load_data_reps(path=joinpath(workpath,data_path+".pkl"),path_out=joinpath(workpath,data_second_path+".pkl"),l2normal=l2normal)

    z_in=data['in_train'].astype('float32')
    z_in_test=data['in_test'].astype('float32')
    z_out_test=data['out_test'].astype('float32')

    ndim=z_in.shape[1]

    if args.background_model is not None:
        # using external (ImageNet) background model
        with open(joinpath(workpath,args.background_model), 'rb') as f:
            imagenet_mean = np.load(f, allow_pickle=True)
            imagenet_cov = np.load(f, allow_pickle=True)
            imagenet_mean=imagenet_mean[None,:]
            imagenet_cov=imagenet_cov
            if standard:
                # applying standardization according to IN data
                imagenet_mean=imagenet_mean-np.mean(z_in,axis=0)
                stds=np.std(z_in,axis=0)
                for i in range(ndim):
                    for j in range(ndim):
                        imagenet_cov[i,j]=imagenet_cov[i,j]/(stds[i]*stds[j])
        ext_back_model=True
    else:
        ext_back_model=False

    if standard:
        sst=StandardScaler().fit(z_in)
        z_in=sst.transform(z_in)
        z_in_test=sst.transform(z_in_test)
        z_out_test=sst.transform(z_out_test)
        print('standardized')

    # mean and diagonal covariance of training (IN) data for background model
    mean_in_all=np.mean(z_in,axis=0)
    var_in_all=np.var(z_in,axis=0)
    if 'ratio' in args.flags:
        if not ext_back_model:
            back_mean_cov=(mean_in_all[None,:].copy(),var_in_all.copy())
        else:
            back_mean_cov=(imagenet_mean,imagenet_cov)
    else:
        back_mean_cov=None

    k_neighbor=args.K_prime
    print('initializing faiss...','k=',k_neighbor,'d=',ndim)
    faiss_index= faiss.IndexFlatL2(ndim)
    faiss_index.add(z_in)

    if faisscuda:
        print('faiss running on GPU.')
        gpures = faiss.StandardGpuResources()
        faiss_index=faiss.index_cpu_to_gpu(gpures, 0, faiss_index)
    
    Lambda=args.Lambda
    if 'noloc' in args.flags:
        var_in_all=np.ones(ndim)
        Lambda=0
        print("no local distance adaptation.")


    compute = lambda test_data,train_data : vglr(test_data,train_data,
                                           faiss_index,k_neighbor,var_in_all,Lambda,back_mean_cov=back_mean_cov) 
    

    print('computing scores on IND test data...')
    s_test_in=compute(z_in_test,z_in)
    print('computing scores on OOD test data...')
    s_test_out=compute(z_out_test,z_in)

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
        logger=utils.logwriter("VGLR",args.output_file)
        logger.write(str(args.__dict__))
        logger.write("AUROC: {:.2f}%  FPR@95TPR: {:.2f}%".format(res['auroc']*100,res['fpr95tpr']*100))
    else:
        utils.repeated_run(run,"VGLR",args)
    print(" ")