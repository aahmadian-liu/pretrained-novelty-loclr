"""
Basic k-Nearest Neighbor novelty detection
"""

import numpy as np
import utils 
import faiss
from sklearn.preprocessing import StandardScaler
import argparse
from os.path import join as joinpath
import matplotlib.pyplot as plot
from os import environ


# Parameters
argsp=argparse.ArgumentParser()
argsp.add_argument("data",type=str,help="name of the data file (without pkl extension) which contains the representations in the 'workspace' directory")
argsp.add_argument("K",type=int,help="hyperparameter K in K-nearest neighbor")
argsp.add_argument("--data_second",type=str,default=None,help="the data representations file of the dataset assumed as OOD (only for far-OOD experiments)")
argsp.add_argument("--standardize",action='store_true',help="if enabled, the representation vectors will be standardized using mean and standard deviation")
argsp.add_argument("--not_l2_normal",action='store_true',help="if enabled, the representation vectors will not be normalized in l^2 norm")
argsp.add_argument("--use_sum_distance",action='store_true',help="use sum of distances to neighbors instead of maximum distance")
argsp.add_argument("--output_file",type=str,default=None,help="name for a text file in the 'output' directory to write the results")
argsp.add_argument("--repeats",type=int,default=0,help="the number of times the method will be repeated using different normal/novel splits (the data files should be stored by appending rX to their name, where X is the number of split)")
argsp.add_argument("--dis_histogram_name",type=str,default=None,help="if specified, a histogram of scores (distances) is saved with this name")
argsp.add_argument("--cuda",action='store_true',help="to use GPU for nearest neighbor search")
argsp.add_argument("--comment",type=str,default='')

workpath=environ.get("ood_ws","workspace")


def run(data_path,data_second_path):
    """The main function for running the method"""

    l2normal=not args.not_l2_normal
    standard=args.standardize
    k_neighbor=args.K
    sumdis=args.use_sum_distance

    print("loading data...")
    if args.data_second is None:
        data=utils.load_data_reps(path=joinpath(workpath,data_path+".pkl"),l2normal=l2normal)
    else:
        data=utils.load_data_reps(path=joinpath(workpath,data_path+".pkl"),path_out=joinpath('workspace',data_second_path+".pkl"),l2normal=l2normal)

    z_in=data['in_train'].astype('float32')
    z_in_test=data['in_test'].astype('float32')
    z_out_test=data['out_test'].astype('float32')

    if standard:
        sst=StandardScaler().fit(z_in)
        z_in=sst.transform(z_in)
        z_in_test=sst.transform(z_in_test)
        z_out_test=sst.transform(z_out_test)
        print('standardized')

    ndim=z_in.shape[1]
    print('initializing faiss...','k=',k_neighbor,'d=',ndim)
    index= faiss.IndexFlatL2(ndim)
    index.add(z_in)

    if args.cuda:
        print('faiss running on GPU.')
        gpures = faiss.StandardGpuResources()
        index=faiss.index_cpu_to_gpu(gpures, 0, index)

    print("computing scores on IND test data...")
    dist_in,nns_in=index.search(z_in_test,k_neighbor)
    print("computing scores on OOD test data...")
    dist_out,nns_out=index.search(z_out_test,k_neighbor)

    if not sumdis:
        s_test_in=dist_in[:,k_neighbor-1] #distance to the k'th nearest neighbor
        s_test_out=dist_out[:,k_neighbor-1]
    else:
        s_test_in=dist_in.sum(axis=1)
        s_test_out=dist_out.sum(axis=1)
        print("sum of dis mode")

    onehots = np.array([1] * len(s_test_out) + [0] * len(s_test_in))
    scores = np.concatenate([np.array(s_test_out), np.array(s_test_in)], axis=0)

    res=utils.roc(onehots,scores)
    print("AUROC:",res['auroc'])
    print("FPR@95TPR:",res['fpr95tpr'])

    if args.dis_histogram_name is not None:
        plot.figure()
        plot.hist(dist_in,label="IN (test)")
        plot.hist(dist_out,label="OOD (test)",alpha=0.7)
        plot.legend()
        plot.savefig(joinpath('output',args.dis_histogram_name+".png"))
        print("plot saved")

    return res

if __name__=='__main__':
    args=argsp.parse_args()
    if args.repeats==0:
        res=run(args.data,args.data_second)
        logger=utils.logwriter("Basic KNN",args.output_file)
        logger.write(str(args.__dict__))
        logger.write("AUROC: {:.2f}%  FPR@95TPR: {:.2f}%".format(res['auroc']*100,res['fpr95tpr']*100))
    else:
        utils.repeated_run(run,"Basic KNN",args)
    print(" ")