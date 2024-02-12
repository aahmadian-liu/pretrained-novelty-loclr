##############################################################
# The proposed Voronoi-Gaussian novelty detection 
##############################################################

import numpy as np
import utils
from sklearn.preprocessing import StandardScaler
from numpy.linalg import inv,det
import torch
import faiss
import argparse
from os.path import join as joinpath
from os import environ


argsp=argparse.ArgumentParser()
argsp.add_argument("data",type=str, help="name of the data file (without extension) which contains the represenations in the 'workspace' directory")
argsp.add_argument("K_prime",type=int, help="the k' hyperparameter (number of neighbors of neighbor)")
argsp.add_argument("--data_second",type=str,default=None, help="the data represenations file of the dataset that is assumed as novel (only for far-OOD experiments)")
argsp.add_argument("--Lambda",type=float,default=1.0, help="covariance regularization hyperparameter (always 1 in our experiments)")
argsp.add_argument("--flags",default="covdet,ratio", 
                   help="the flags for algorithm options. 'covdet': use covariance determinant term in the Gaussian distribution, 'ratio': use likelihood ratio with background model, 'noloc':no local adaptation")
argsp.add_argument("--not_standardize",action='store_true', help="if enabled, the representation vectors will not be standardized using mean and standard deviation")
argsp.add_argument("--not_l2_normal",action='store_true', help="if enabled, the representation vectors will not be normalized in l-2 norm")
argsp.add_argument("--background_model",type=str,default=None, help="used to load the background mean and covariance from file (e.g., Imagenet statistics) instead of computing them on training data")
argsp.add_argument("--output_file",type=str,default=None, help="text file name in the 'output' directory to write the results")
argsp.add_argument("--repeats",type=int,default=0, help="the number of times the method will be repeated using different normal/novel splits (the data files should be stored by appending rX to their name, where X is the number of split)")
argsp.add_argument("--cuda",action='store_true', help="to use GPU for nearest neighbor search")
argsp.add_argument("--comment",type=str,default='')


def vglr(x,data_in,knn_model,k_prime,globcov,lamda,k=1,nn_of_nn=True,dis_to_mean=False,nn_moment=True,back_mean_cov=None,use_cov_det=True):

    print('running VGLR...')
    scores=[] # novelty scores for each input data

    if len(globcov.shape)==2 and globcov.shape[1]>1:
        print('mat glob cov')
        matglob=True # when global covariance matrix is not diagonal
    else:
        matglob=False

    if dis_to_mean:
        print("using mean as center")
    if not nn_moment:
        print("using local variance (around mean)")

    if back_mean_cov is not None:
        print('with back term')
        if len(back_mean_cov[1].shape)==1:
            invbackcov=np.diag(1.0/back_mean_cov[1]) # inverse of diagonal background covariance
        else:
            invbackcov=inv(back_mean_cov[1]) # inverse of full background covariance
        backmean=back_mean_cov[0] 
        ratio=True # likelihood ratio mode
    else:
        ratio=False
 
 
    _,nn_inds=knn_model.search(x,k) # nearest (or kth) neighbor of each input
    nn_inds=nn_inds[:,-1]

    if nn_of_nn: # estimate local covariance on nearest neighbors of input's nearest neighbor
        _,nbh_inds=knn_model.search(data_in[nn_inds[:],:],k_prime)
    else:
        nbh_inds=nn_inds
    print("knn search done.")

    # for each input data point
    for i in range(x.shape[0]):

        inds_loc=nbh_inds[i,:]
        d_loc=data_in[inds_loc,:] # the data in current neighborhood

        nn=data_in[nn_inds[i],:] # nearest neighbor of current input

        if dis_to_mean: # use either the nearest neighbor or mean of local data as center
            center=np.mean(d_loc,axis=0)
        else:
            center=nn
        
        if nn_moment: # obtain the local diagonal matrix either as moment around nearest neighbor or ordinary variance
            cov_loc_0=np.mean((d_loc-nn)**2,axis=0)
        else:
            cov_loc_0=np.var(d_loc,axis=0)

        if matglob:
            cov_loc_0=np.diag(cov_loc_0)

        cov_loc = lamda*cov_loc_0 + (1-lamda)*globcov  # final local covariance matrix by regularization
    
        eps=1e-5
        
        # the main score value
        if not matglob:
            cov_loc_inv=1.0/(cov_loc+eps)
            d=(cov_loc_inv*((x[i,:]-center)**2)).sum()
        else:
            cov_loc_inv=inv(cov_loc)
            d=(x[i,:]-center) @ cov_loc_inv @ (x[i,:]-center).T

        if use_cov_det: # include log-det of Gaussian density
            if not matglob:
                d+=np.log(cov_loc+eps).sum()
            else:
                d+=np.log(det(cov_loc)+eps)

        if ratio:
            db=(x[i:i+1,:]-backmean) @ invbackcov @ (x[i:i+1,:]-backmean).T
            db=db[0]
            d=d-db # subtracting the score given by background Gaussian model

        scores.append(d)
    

    return scores


# The main function

def run(data_path,data_second_path):

    l2normal=not args.not_l2_normal
    standard=not args.not_standardize
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

    if args.background_model is not None:
        with open(joinpath(workpath,args.background_model), 'rb') as f:
            imagenet_mean = np.load(f, allow_pickle=True)
            imagenet_cov = np.load(f, allow_pickle=True)
            imagenet_mean=imagenet_mean[None,:]
            imagenet_cov=imagenet_cov
            if standard:
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

    mean_in_all=np.mean(z_in,axis=0)
    var_in_all=np.var(z_in,axis=0)
    #cov_all=np.cov(z_in,rowvar=False)
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
        print('faiss gpu...')
        gpures = faiss.StandardGpuResources()
        faiss_index=faiss.index_cpu_to_gpu(gpures, 0, faiss_index)
    
    Lambda=args.Lambda
    use_covdet= ('covdet' in args.flags)
    use_mean= ('locmean' in args.flags)
    use_moment= not ('locvar' in args.flags)
    if 'noloc' in args.flags:
        var_in_all=np.ones(ndim)
        Lambda=0
        print("no local distance adaptation.")
    print("lambda",Lambda)

    compute = lambda test_data,train_data : vglr(test_data,train_data,
                                           faiss_index,k_neighbor,var_in_all,Lambda,back_mean_cov=back_mean_cov,use_cov_det=use_covdet,dis_to_mean=use_mean,nn_moment=use_moment) 
    

    print('computing scores on IN test data...')
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
    workpath=environ.get("ood_ws","workspace")
    if args.repeats==0:
        res=run(args.data,args.data_second)
        logger=utils.logwriter("Local Mahalanobis",args.output_file)
        logger.write(str(args.__dict__))
        logger.write("AUROC: {:.2f}%  FPR@95TPR: {:.2f}%".format(res['auroc']*100,res['fpr95tpr']*100))
    else:
        utils.repeated_run(run,"Local Mahalanobis",args)
    print(" ")