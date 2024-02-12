##############################################################
# One-class SVM novelty detection 
##############################################################


import numpy as np
import utils 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
import argparse
from os.path import join as joinpath
from os import environ
import pickle


argsp=argparse.ArgumentParser()
argsp.add_argument("data",type=str)
argsp.add_argument("--nu",type=float,default=0.5)
argsp.add_argument("--gamma",type=float)
argsp.add_argument("--data_second",type=str,default=None)
argsp.add_argument("--not_standardize",action='store_true')
argsp.add_argument("--not_l2_normal",action='store_true')
argsp.add_argument("--model_save_path",type=str,default=None)
argsp.add_argument("--output_file",type=str,default=None)
argsp.add_argument("--repeats",type=int,default=0)
argsp.add_argument("--truncate_data",type=int,default=None)
argsp.add_argument("--cuda",action='store_true')
argsp.add_argument("--comment",type=str,default='')


def run(data_path,data_second_path): 

    l2normal=not args.not_l2_normal
    standard=not args.not_standardize
    pca=False

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

    if pca:
        pcamap=PCA()
        pcamap=pcamap.fit(z_in)
        z_in=pcamap.transform(z_in)
        z_in_test=pcamap.transform(z_in_test)
        z_out_test=pcamap.transform(z_out_test)
        print('pca transformed')

    print("fitting osvm...")
    print("rbf kernel, gamma=",args.gamma)

    osvm=OneClassSVM(kernel='rbf',gamma=args.gamma,nu=args.nu)
    osvm=osvm.fit(z_in)

    if args.model_save_path is not None:
        with open(joinpath(workpath,'osvm',args.model_save_path),'wb') as f:
            pickle.dump(osvm,f)
            print("osvm saved to disk.")

    print("computing scores...")

    s_test_in=-1*osvm.score_samples(z_in_test)
    s_test_out=-1*osvm.score_samples(z_out_test)

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
        logger=utils.logwriter("OSVM",args.output_file)
        logger.write(str(args.__dict__))
        logger.write("AUROC: {:.2f}%  FPR@95TPR: {:.2f}%".format(res['auroc']*100,res['fpr95tpr']*100))
    else:
        utils.repeated_run(run,"OSVM",args)
    print(" ")