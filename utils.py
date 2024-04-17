"""
 Common functions used by all methods (loading data representations, evaluation, etc.)
"""


import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import datetime
from os.path import join as joinpath
import platform


def load_data_reps(path,path_out=None,l2normal=True):
    """ Loads pre-computed representation vectors of IN and OOD data from a single or two separate files. It normalizes vectors by their l^2 norm by default."""

    sepood= not (path_out is None) # load OOD data from a separate file

    with open(path,'rb') as file:
        data1=pickle.load(file)
        print('loaded:',path)

    if sepood:
        with open(path_out,'rb') as file:
            data2=pickle.load(file)
            print('loaded:',path_out)
    
    has_out_train= 'out_train' in data1['reps']

    
    data1=data1['reps']
    if not sepood:
        z_in_train=data1['in_train']
        z_in_test=data1['in_test']
        z_out_test=data1['out_test']
        if has_out_train:
            z_out_train=data1['out_train']
    else:
        data2=data2['reps']
        z_in_train=data1['in_train']
        z_in_test=data1['in_test']
        z_out_test=data2['in_test']

    print("data shape:",z_in_train.shape,z_in_test.shape,z_out_test.shape)

    if l2normal:
        z_in_train=z_in_train/np.linalg.norm(z_in_train, axis=1,keepdims=True)
        z_in_test=z_in_test/np.linalg.norm(z_in_test, axis=1,keepdims=True)
        z_out_test=z_out_test/np.linalg.norm(z_out_test, axis=1,keepdims=True)
        if has_out_train:
            z_out_train=z_out_train/np.linalg.norm(z_out_train, axis=1,keepdims=True)
        print('l2 normalized')

    if not has_out_train:
        return {'in_train':z_in_train,'in_test':z_in_test,'out_test':z_out_test}
    else:
        return {'in_train':z_in_train,'in_test':z_in_test,'out_test':z_out_test,'out_train':z_out_train}


def load_data_class(path):
    """ Loads class labels of IN and OOD data from a file."""

    with open(path,'rb') as file:
        data1=pickle.load(file)
        print('loaded:',path)
    
    has_out_train= 'out_train' in data1['reps']
    
    z_in_train=data1['labels']['in_train']
    z_in_test=data1['labels']['in_test']
    z_out_test=data1['labels']['out_test']
    if has_out_train:
        z_out_train=data1['labels']['out_train']


    if not has_out_train:
        return {'in_train':z_in_train,'in_test':z_in_test,'out_test':z_out_test}
    else:
        return {'in_train':z_in_train,'in_test':z_in_test,'out_test':z_out_test,'out_train':z_out_train}
    

def roc(one_hots, scores, make_plot=False, add_to_title=None,plot_legend=['in','out']):
    """Computes AUROC, FPR@95TPR and ROC curve given the groundtruth binary labels and predicted scores of IN and OOD data.
    OOD and IN data points should correspond to 1 and 0 in 'labels' respectively. Score should be higher on OOD.
    """
    # Based on https://github.com/stanislavfort/exploring_the_limits_of_OOD_detection

    auroc = roc_auc_score(one_hots, scores) # using the scikit-learn function 

    out_scores, in_scores = scores[one_hots == 1], scores[one_hots == 0]
    
    if make_plot:
        plt.figure(figsize=(5.5, 3), dpi=100)

        if add_to_title is not None:
            plt.title(add_to_title + " AUROC=" + str(float(auroc * 100))[:6] + "%", fontsize=14)
        else:
            plt.title(" AUROC=" + str(float(auroc * 100))[:6] + "%", fontsize=14)

    vals, bins = np.histogram(in_scores, bins=51)
    bin_centers = (bins[1:] + bins[:-1]) / 2.0

    if make_plot:
        plt.plot(bin_centers, vals, linewidth=4, color="navy", marker="", label=plot_legend[0])
        plt.fill_between(bin_centers, vals, [0] * len(vals), color="navy", alpha=0.3)

    vals, bins = np.histogram(out_scores, bins=51)
    bin_centers = (bins[1:] + bins[:-1]) / 2.0

    if make_plot:
        plt.plot(bin_centers, vals, linewidth=4, color="crimson", marker="", label=plot_legend[1])
        plt.fill_between(bin_centers, vals, [0] * len(vals), color="crimson", alpha=0.3)

    if make_plot:
        plt.xlabel("Score", fontsize=14)
        plt.ylabel("Count", fontsize=14)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.ylim([0, None])

        plt.legend(fontsize=14)

        plt.tight_layout()

    fpr=fpr_at_95_tpr(one_hots,scores)

    if make_plot:
        out={'auroc':auroc,'fpr95tpr':fpr,'plot':plt.gcf()}
    else:
        out={'auroc':auroc,'fpr95tpr':fpr}
    
    return out


def fpr_at_95_tpr(labels,preds, pos_label=1):
    """Return the FPR when TPR is at minimum 95%.
        
    preds: array, shape = [n_samples]
           Target normality scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           i.e.: an high value means sample predicted "normal", belonging to the positive class
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    pos_label: label of the positive class (1 by default)
    """
    # From https://github.com/tayden/ood-metrics

    fpr, tpr, _ = roc_curve(labels, preds, pos_label=pos_label)

    if all(tpr < 0.95):
        # No threshold allows TPR >= 0.95
        return 0
    elif all(tpr >= 0.95):
        # All thresholds allow TPR >= 0.95, so find lowest possible FPR
        idxs = [i for i, x in enumerate(tpr) if x >= 0.95]
        return min(map(lambda idx: fpr[idx], idxs))
    else:
        # Linear interp between values to get FPR at TPR == 0.95
        return np.interp(0.95, tpr, fpr)
    

class logwriter:
    """Class for writing results of experiments to text file"""

    def __init__(self,title,filename=None):

        if filename=="0":
            self.nolog=True
            return
        else:
            self.nolog=False

        now=datetime.datetime.now().replace(microsecond=0)
        
        if filename is None:
            filename="log_" + str(now.year) + "-" + str(now.month) + "-" + str(now.day) + ".txt"
        self.filename=filename

        with open(joinpath("output",filename),'at') as file:
            file.write("\n\n" + "### " + title + " | " + str(now) + " | " + platform.node() + " ###" + "\n")
        
    def write(self,text):
        if not self.nolog:
            with open(joinpath("output",self.filename),'at') as file:
                file.write(text + "\n")
    
    def write_error(self,exception):
        print("log write")
        if not self.nolog:
            with open(joinpath("output",self.filename),'at') as file:
                file.write("Error: " + str(exception) + "\n")

def repeated_run(runfun,title,args):
    """Runs a method for several times to compute the average AUROC and FPR@95TPR over several IN/ODD datasets. 
    The data files names should be identical except for an 'r' postfix showing the index, e.g. 'DATA.pkl' , 'DATAr0.pkl' , 'DATAr1.pkl' """

    auroc=[]
    fpr=[]

    for i in range(0,args.repeats+1):
        path=args.data
        if i>0:
            path=path+"r"+str(i-1)
        res=runfun(path,args.data_second)
        auroc.append(res['auroc'])
        fpr.append(res['fpr95tpr']) 

    maurco=np.mean(auroc)
    sdauroc=np.std(auroc)
    mfpr=np.mean(fpr)
    sdfpr=np.std(fpr)
    logger=logwriter(title,args.output_file)
    logger.write(str(args.__dict__))
    logger.write("AUROC: {:.2f}% (+/-{:.2f}%)  FPR@95TPR: {:.2f}% (+/-{:.2f}%)".format(maurco*100,sdauroc*100,mfpr*100,sdfpr*100))
    logger.write("AUROCs: " + str(auroc) + " FPR@95TPRs: " + str(fpr))
    print("Average AUROC:",maurco)
    print("Average FPR@95TPR:",mfpr)
    