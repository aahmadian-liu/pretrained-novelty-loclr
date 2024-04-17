"""
 K-means Clustering followed by Mahalanobis distance based novelty detection 
"""


import pickle
import utils
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import StandardScaler
import argparse
from os.path import join as joinpath
import json
import random
from os import environ


argsp=argparse.ArgumentParser()
argsp.add_argument("data",type=str,help="name of the data file (without pkl extension) which contains the representations in the 'workspace' directory")
argsp.add_argument("k",type=int,help="number of clusters in k-means")
argsp.add_argument("--max_iters",type=int,default=500,help="maximum iterations in k-means algorithm")
argsp.add_argument("--shared_covariance",action='store_true',help="whether to use a shared covariance matrix for all classes of in-distribution data when calculating Maha. distances")
argsp.add_argument("--data_second",type=str,default=None,help="the data representations file of the dataset assumed as OOD (only for far-OOD experiments)")
argsp.add_argument("--standardize",action='store_true',help="if enabled, the representation vectors will be standardized using mean and standard deviation")
argsp.add_argument("--not_l2_normal",action='store_true',help="if enabled, the representation vectors will not be normalized in l^2 norm")
argsp.add_argument("--use_rel_maha",action='store_true',help="whether to use Relative Maha. distances similar to Ren et al. 2021 (likelihood ratio)")
argsp.add_argument("--model_save_path",type=str,default='temp_km',help="path for saving the fitted scikit-learn k-means object inside the working directory")
argsp.add_argument("--output_file",type=str,default=None,help="name for a text file in the 'output' directory to write the results")
argsp.add_argument("--repeats",type=int,default=0,help="the number of times the method will be repeated using different normal/novel splits (the data files should be stored by appending rX to their name, where X is the number of split)")
argsp.add_argument("--randomseed",type=int,default=0)
argsp.add_argument("--comment",type=str,default='')

workpath=environ.get("ood_ws","workspace")

def cluster(data_path):
    """Clusters the representations of training (IN) data using k-means, and saves the obtained sklearn model to a file."""
    
    l2normal=not args.not_l2_normal
    standard=args.standardize

    print("loading data...")
    data=utils.load_data_reps(path=joinpath(workpath,data_path+".pkl"),l2normal=l2normal)

    z_in=data['in_train']

    if standard:
        sst=StandardScaler().fit(z_in)
        z_in=sst.transform(z_in)
        print('standardized')

    n_clus=args.k
    print('running k-means,  k=' + str(n_clus) + " ...")

    km=KMeans(n_clusters=n_clus,init="k-means++",max_iter=args.max_iters,algorithm="full",n_init=2)
    km.fit(z_in)

    fdic=dict()
    fdic['kmeans']=km
    fdic['k']=n_clus
    fdic['vars']=vars(args)
    outf=joinpath(workpath,args.model_save_path)

    with open(outf+'.pkl','wb') as f:
        pickle.dump(fdic,f)

    with open(outf+'.json','w') as f:
        json.dump(vars(args),f)

    print("fitted kmeans saved to",outf)

  
def maha_scores(
        indist_train_embeds_in,
        indist_train_labels_in,
        indist_test_embeds_in,
        outdist_test_embeds_in,
        subtract_train_distance=False,
        indist_classes=100,
        cal_cov_per_class=False,
        norm_name="L2",
):
    """Computes Mahalanobis distance of input data points w.r.t (per-class) mean and covariance matrices computed on labeled training (in-distribution/normal) data.
    The first four arguments correspond to the representations (embeddings) and labels of training IN data, and representations of the test IN/OOD data. The other arguments are: 
    subtract_train_distance: if enabled, a correction term is subtracted from every score, which is Maha. distance w.r.t to the pooled (unlabeled) training data as proposed by Ren et al. 2021 (relative Maha. distance) 
    indist_classes: number of the classes/clusters of IN data
    cal_cov_per_class: if enabled, a separate covariance matrix is obtained for each class of IN data instead of a shared covariance
    """

    # Based on https://github.com/stanislavfort/exploring_the_limits_of_OOD_detection
    
    
    # storing the replication results
    maha_intermediate_dict = dict()

    description = ""

    all_train_mean = np.mean(indist_train_embeds_in, axis=0, keepdims=True)

    indist_train_embeds_in_touse = indist_train_embeds_in
    indist_test_embeds_in_touse = indist_test_embeds_in
    outdist_test_embeds_in_touse = outdist_test_embeds_in

    #if subtract_mean:
    #    indist_test_embeds_in_touse -= all_train_mean
    #    outdist_test_embeds_in_touse -= all_train_mean
    #    description = description + " subtract mean,"

    #if normalize_to_unity:
    #    indist_train_embeds_in_touse = indist_train_embeds_in_touse / np.linalg.norm(indist_train_embeds_in_touse,
    #                                                                                 axis=1, keepdims=True)
    #    indist_test_embeds_in_touse = indist_test_embeds_in_touse / np.linalg.norm(indist_test_embeds_in_touse, axis=1,
    #                                                                               keepdims=True)
    #    outdist_test_embeds_in_touse = outdist_test_embeds_in_touse / np.linalg.norm(outdist_test_embeds_in_touse,
    #                                                                                 axis=1, keepdims=True)
    #    description = description + " unit norm,"

    # full train single fit
    mean = np.mean(indist_train_embeds_in_touse, axis=0)
    cov = np.cov((indist_train_embeds_in_touse - (mean.reshape([1, -1]))).T)

    cov_inv = np.linalg.inv(cov)

    # getting per class means and covariances
    class_means = []
    class_cov_invs = []
    class_covs = []
    for c in range(indist_classes):

        mean_now = np.mean(indist_train_embeds_in_touse[indist_train_labels_in == c], axis=0)

        nc=(indist_train_labels_in == c).sum()
        if nc<2:
            continue

        cov_now = np.cov((indist_train_embeds_in_touse[indist_train_labels_in == c]).T)
        if cal_cov_per_class:
            eps=0.0001*np.eye(cov_now.shape[0])
            class_cov_invs.append(np.linalg.inv(cov_now+eps))
        else:
            class_covs.append(cov_now*(nc-1))

        class_means.append(mean_now)

    ntot=indist_train_embeds_in_touse.shape[0]

    if not cal_cov_per_class:
        shcov=(1/ntot)*np.sum(np.stack(class_covs, axis=0), axis=0)
        class_cov_invs = [np.linalg.inv(shcov)] * len(class_covs)

    maha_intermediate_dict["class_cov_invs"] = class_cov_invs
    maha_intermediate_dict["class_means"] = class_means
    maha_intermediate_dict["cov_inv"] = cov_inv
    maha_intermediate_dict["mean"] = mean

    def maha_distance(xs, cov_inv_in, mean_in, norm_type=None):
        diffs = xs - mean_in.reshape([1, -1])

        second_powers = np.matmul(diffs, cov_inv_in) * diffs

        if norm_type in [None, "L2"]:
            return np.sum(second_powers, axis=1)
        elif norm_type in ["L1"]:
            return np.sum(np.sqrt(np.abs(second_powers)), axis=1)
        elif norm_type in ["Linfty"]:
            return np.max(second_powers, axis=1)

    out_totrain = maha_distance(outdist_test_embeds_in_touse, cov_inv, mean, norm_name)
    in_totrain = maha_distance(indist_test_embeds_in_touse, cov_inv, mean, norm_name)

    out_totrainclasses = [maha_distance(outdist_test_embeds_in_touse, class_cov_invs[c], class_means[c], norm_name) for
                          c in range(len(class_means))]
    in_totrainclasses = [maha_distance(indist_test_embeds_in_touse, class_cov_invs[c], class_means[c], norm_name) for c
                         in range(len(class_means))]

    out_scores = np.min(np.stack(out_totrainclasses, axis=0), axis=0)
    in_scores = np.min(np.stack(in_totrainclasses, axis=0), axis=0)

    if subtract_train_distance:
        out_scores = out_scores - out_totrain
        in_scores = in_scores - in_totrain

    onehots = np.array([1] * len(out_scores) + [0] * len(in_scores))
    scores = np.concatenate([out_scores, in_scores], axis=0)

    return onehots, scores, description, maha_intermediate_dict

def run(data_path,data_second_path):
    """The main function for running the method"""
    
    # clustering
    cluster(data_path)

    kmeanpath=args.model_save_path

    if args.k>1:
         with open(joinpath(workpath,kmeanpath + '.pkl'),'rb') as f:
             fdic=pickle.load(f)
             km=fdic['kmeans']
             n_clus=fdic['k']
             standard=fdic['vars']['standardize']
             l2normal=not fdic['vars']['not_l2_normal']
             print("kmeans loaded from:",joinpath(workpath,kmeanpath + '.pkl'))
    else:
        n_clus=1
        standard=args.standardize
        l2normal=not args.not_l2_normal

    # computing maha. scores by assuming the obtained clusters as the classes of in-distribution data
         
    print("loading data...")
    if args.data_second is None:
        data=utils.load_data_reps(path=joinpath(workpath,data_path+".pkl"),l2normal=l2normal)
    else:
        data=utils.load_data_reps(path=joinpath(workpath,data_path+".pkl"),path_out=joinpath(workpath,data_second_path+".pkl"),l2normal=l2normal)

    z_in=data['in_train']
    z_in_test=data['in_test']
    z_out_test=data['out_test']

    if standard:
        sst=StandardScaler().fit(z_in)
        z_in=sst.transform(z_in)
        z_in_test=sst.transform(z_in_test)
        z_out_test=sst.transform(z_out_test)
        print('standardized')

    print("Computing scores...","k=",n_clus)
    if n_clus>1:
        y_in=km.predict(z_in)
    else:
        y_in=np.zeros(z_in.shape[0],dtype=int)
    
    onehots,scores,_,_= maha_scores(z_in,y_in,z_in_test,z_out_test,indist_classes=y_in.max()+1,
                                    subtract_train_distance=args.use_rel_maha,cal_cov_per_class=not args.shared_covariance)

    res=utils.roc(onehots,scores)
    print("AUROC:",res['auroc'])
    print("FPR@95TPR:",res['fpr95tpr'])

    args.k=n_clus

    return res


if __name__=='__main__':
    args=argsp.parse_args()
    np.random.seed(args.randomseed)
    random.seed(args.randomseed)
    if args.repeats==0:
        res=run(args.data,args.data_second)
        logger=utils.logwriter("Kmeans+Maha",args.output_file)
        logger.write(str(args.__dict__))
        logger.write("AUROC: {:.2f}%  FPR@95TPR: {:.2f}%".format(res['auroc']*100,res['fpr95tpr']*100))
    else:
        utils.repeated_run(run,"Kmeans+Maha",args)
    print(" ")
