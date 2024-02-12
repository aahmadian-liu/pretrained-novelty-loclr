##############################################################
# k-means clustering + Mahalanobis distance novelty detection 
##############################################################


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
argsp.add_argument("data",type=str)
argsp.add_argument("--N",type=int)
argsp.add_argument("--max_iters",type=int,default=500)
argsp.add_argument("--shared_covariance",action='store_true')
argsp.add_argument("--data_second",type=str,default=None)
argsp.add_argument("--standardize",action='store_true')
argsp.add_argument("--not_l2_normal",action='store_true')
argsp.add_argument("--subtract_back_dis",action='store_true')
argsp.add_argument("--model_save_path",type=str,default='temp_km')
argsp.add_argument("--output_file",type=str,default=None)
argsp.add_argument("--repeats",type=int,default=0)
argsp.add_argument("--truncate_data",type=int,default=None)
argsp.add_argument("--randomseed",type=int,default=0)
argsp.add_argument("--comment",type=str,default='')

args=argsp.parse_args()

np.random.seed(args.randomseed)
random.seed(args.randomseed)
workpath=environ.get("ood_ws","workspace")

def cluster(data_path):

    l2normal=not args.not_l2_normal
    standard=args.standardize

    print("loading data...")
    data=utils.load_data(path=joinpath(workpath,data_path+".pkl"),l2normal=l2normal)

    z_in=data['in_train']

    if args.truncate_data is not None:
        z_in=z_in[0:args.truncate_data,:]

    if standard:
        sst=StandardScaler().fit(z_in)
        z_in=sst.transform(z_in)
        print('standardized')

    n_clus=args.N
    print('running k-means,  N=' + str(n_clus) + " ...")

    km=KMeans(n_clusters=n_clus,init="k-means++",max_iter=args.max_iters,algorithm="full",n_init=2)
    km.fit(z_in)

    fdic=dict()
    fdic['kmeans']=km
    fdic['n']=n_clus
    fdic['vars']=vars(args)
    outf=joinpath(workpath,'kmeans',args.model_save_path)

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
        subtract_mean=True,
        normalize_to_unity=True,
        subtract_train_distance=True,
        indist_classes=100,
        cal_cov_per_class=False,
        norm_name="L2",
):
    # based on https://github.com/stanislavfort/exploring_the_limits_of_OOD_detection
    
    # storing the replication results
    maha_intermediate_dict = dict()

    description = ""

    all_train_mean = np.mean(indist_train_embeds_in, axis=0, keepdims=True)

    indist_train_embeds_in_touse = indist_train_embeds_in
    indist_test_embeds_in_touse = indist_test_embeds_in
    outdist_test_embeds_in_touse = outdist_test_embeds_in

    if subtract_mean:
        indist_train_embeds_in_touse -= all_train_mean
        indist_test_embeds_in_touse -= all_train_mean
        outdist_test_embeds_in_touse -= all_train_mean
        description = description + " subtract mean,"

    if normalize_to_unity:
        indist_train_embeds_in_touse = indist_train_embeds_in_touse / np.linalg.norm(indist_train_embeds_in_touse,
                                                                                     axis=1, keepdims=True)
        indist_test_embeds_in_touse = indist_test_embeds_in_touse / np.linalg.norm(indist_test_embeds_in_touse, axis=1,
                                                                                   keepdims=True)
        outdist_test_embeds_in_touse = outdist_test_embeds_in_touse / np.linalg.norm(outdist_test_embeds_in_touse,
                                                                                     axis=1, keepdims=True)
        description = description + " unit norm,"

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

    cluster(data_path)

    kmeanpath=args.model_save_path

    if args.N>1:
         with open(joinpath(workpath,'kmeans',kmeanpath + '.pkl'),'rb') as f:
             fdic=pickle.load(f)
             km=fdic['kmeans']
             n_clus=fdic['n']
             standard=fdic['vars']['standardize']
             l2normal=not fdic['vars']['not_l2_normal']
             print("kmeans loaded from:",joinpath(workpath,'kmeans',kmeanpath + '.pkl'))
    else:
        n_clus=1
        standard=args.standardize
        l2normal=not args.not_l2_normal

    print("loading data...")
    if args.data_second is None:
        data=utils.load_data(path=joinpath(workpath,data_path+".pkl"),l2normal=l2normal)
    else:
        data=utils.load_data(path=joinpath(workpath,data_path+".pkl"),path_out=joinpath(workpath,data_second_path+".pkl"),l2normal=l2normal)

    z_in=data['in_train']
    z_in_test=data['in_test']
    z_out_test=data['out_test']

    if args.truncate_data is not None:
        z_in=z_in[0:args.truncate_data,:]
        print('training size', args.truncate_data)

    if standard:
        sst=StandardScaler().fit(z_in)
        z_in=sst.transform(z_in)
        z_in_test=sst.transform(z_in_test)
        z_out_test=sst.transform(z_out_test)
        print('standardized')

    print("Computing scores...","N=",n_clus)
    if n_clus>1:
        y_in=km.predict(z_in)
    else:
        y_in=np.zeros(z_in.shape[0],dtype=int)
    
    onehots,scores,_,_= maha_scores(z_in,y_in,z_in_test,z_out_test,indist_classes=y_in.max()+1,subtract_mean=False,
                                    normalize_to_unity=False,subtract_train_distance=args.subtract_back_dis,cal_cov_per_class=not args.shared_covariance)

    res=utils.roc(onehots,scores)
    print("AUROC:",res['auroc'])
    print("FPR@95TPR:",res['fpr95tpr'])

    args.N=n_clus

    return res



if args.repeats==0:
    res=run(args.data,args.data_second)
    logger=utils.logwriter("Kmeans",args.output_file)
    logger.write(str(args.__dict__))
    logger.write("AUROC: {:.2f}%  FPR@95TPR: {:.2f}%".format(res['auroc']*100,res['fpr95tpr']*100))
else:
    utils.repeated_run(run,"Kmeans",args)
print(" ")
