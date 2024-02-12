##############################################################
# Basic k-Nearest Neighbors novelty detection
##############################################################


import numpy as np
import utils 
import faiss
from sklearn.preprocessing import StandardScaler
import argparse
from os.path import join as joinpath
import matplotlib.pyplot as plot
from os import environ


argsp=argparse.ArgumentParser()
argsp.add_argument("data",type=str)
argsp.add_argument("K",type=int)
argsp.add_argument("--data_second",type=str,default=None)
argsp.add_argument("--standardize",action='store_true')
argsp.add_argument("--not_l2_normal",action='store_true')
argsp.add_argument("--use_sum_distance",action='store_true')
argsp.add_argument("--use_squared",action='store_true')
argsp.add_argument("--output_file",type=str,default=None)
argsp.add_argument("--repeats",type=int,default=0)
argsp.add_argument("--truncate_data",type=int,default=None)
argsp.add_argument("--dis_histogram_name",type=str,default=None)
argsp.add_argument("--cuda",action='store_true')
argsp.add_argument("--comment",type=str,default='')

args=argsp.parse_args()
workpath=environ.get("ood_ws","workspace")

def run(data_path,data_second_path):

    l2normal=not args.not_l2_normal
    standard=args.standardize
    k_neighbor=args.K
    sumdis=args.use_sum_distance

    print("loading data...")
    if args.data_second is None:
        data=utils.load_data(path=joinpath(workpath,data_path+".pkl"),l2normal=l2normal)
    else:
        data=utils.load_data(path=joinpath(workpath,data_path+".pkl"),path_out=joinpath('workspace',data_second_path+".pkl"),l2normal=l2normal)

    z_in=data['in_train'].astype('float32')
    z_in_test=data['in_test'].astype('float32')
    z_out_test=data['out_test'].astype('float32')

    if args.truncate_data is not None:
        z_in=z_in[0:args.truncate_data,:]
        print('training size', args.truncate_data)

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
        print('faiss gpu...')
        gpures = faiss.StandardGpuResources()
        index=faiss.index_cpu_to_gpu(gpures, 0, index)

    print("computing in-data scores...")
    dist_in,nns_in=index.search(z_in_test,k_neighbor)
    print("computing ood scores...")
    dist_out,nns_out=index.search(z_out_test,k_neighbor)

    if args.use_squared:
        dist_in=dist_in**2
        dist_out=dist_out**2
        print("dis squared mode")
        
    if not sumdis:
        s_test_in=dist_in[:,k_neighbor-1]
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
    if args.repeats==0:
        res=run(args.data,args.data_second)
        logger=utils.logwriter("Basic KNN",args.output_file)
        logger.write(str(args.__dict__))
        logger.write("AUROC: {:.2f}%  FPR@95TPR: {:.2f}%".format(res['auroc']*100,res['fpr95tpr']*100))
    else:
        utils.repeated_run(run,"Basic KNN",args)
    print(" ")