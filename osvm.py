"""
 One-class SVM novelty detection 
"""

import numpy as np
import utils 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import argparse
from os.path import join as joinpath
from os import environ
import pickle


# parameters
argsp=argparse.ArgumentParser()
argsp.add_argument("data",type=str,help="name of the data file (without pkl extension) which contains the representations in the 'workspace' directory")
argsp.add_argument("--nu",type=float,default=0.5,help="hyperparameter 'nu' in OC-SVM training (scikit-learn implementation)")
argsp.add_argument("--gamma",type=float,default=0.02,help="hyperparameter 'gamma' of RBF kernel in OC-SVM training (scikit-learn implementation)")
argsp.add_argument("--data_second",type=str,default=None,help="the data representations file of the dataset assumed as OOD (only for far-OOD experiments)")
argsp.add_argument("--not_standardize",action='store_true',help="if enabled, the representation vectors will not be standardized using mean and standard deviation")
argsp.add_argument("--not_l2_normal",action='store_true',help="if enabled, the representation vectors will not be normalized in l^2 norm")
argsp.add_argument("--model_save_path",type=str,default=None,help="if specified, the trained OC-SVM model is saved to this path")
argsp.add_argument("--output_file",type=str,default=None,help="name for a text file in the 'output' directory to write the results")
argsp.add_argument("--repeats",type=int,default=0,help="the number of times the method will be repeated using different normal/novel splits (the data files should be stored by appending rX to their name, where X is the number of split)")
argsp.add_argument("--comment",type=str,default='')

workpath=environ.get("ood_ws","workspace")

def run(data_path,data_second_path): 
    """The main function for running the method"""

    l2normal=not args.not_l2_normal
    standard=not args.not_standardize

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

    print("fitting oc-svm on IND training data...")
    print("rbf kernel, gamma=",args.gamma)

    osvm=OneClassSVM(kernel='rbf',gamma=args.gamma,nu=args.nu)
    osvm=osvm.fit(z_in)

    if args.model_save_path is not None:
        with open(joinpath(workpath,'osvm',args.model_save_path),'wb') as f:
            pickle.dump(osvm,f)
            print("osvm saved to disk.")

    print('computing scores on IND test data...')
    s_test_in=-1*osvm.score_samples(z_in_test)
    print('computing scores on OOD test data...')
    s_test_out=-1*osvm.score_samples(z_out_test)

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
        logger=utils.logwriter("OC-SVM",args.output_file)
        logger.write(str(args.__dict__))
        logger.write("AUROC: {:.2f}%  FPR@95TPR: {:.2f}%".format(res['auroc']*100,res['fpr95tpr']*100))
    else:
        utils.repeated_run(run,"OC-SVM",args)
    print(" ")