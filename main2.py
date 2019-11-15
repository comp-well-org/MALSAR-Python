import os
import numpy as np
from glob import glob
from scipy.io import loadmat, savemat
from shutil import move
from tqdm import tqdm
import matplotlib.pyplot as plt

from py.Least_L21 import Least_L21


# rootPath = 'LR+0.001|BS+20|OPT+adam|GC+5|NOI+0.1|LAT+4|NC+1|SR+8HZ|UNSUPV|LocaGausLstmAutoencoderWithBnSRTied_v2+well+v2+debug+'
rootPath = 'LR+|BS+50|OPT+adam|GC+5|NOI+0.1|LAT+4|NC+1|SR+1HZ|UNSUPV|LocaGausAutoencoderWithBnSRTied_v2+well+v2+log+'

channelList = ["*", "sc", "st", "ac"]
ftFoldList = ["f0","f1","f2","f3"]
numDayList = [10]
rndLabList = list(range(1,6))
wbLabList = list(range(10))

pTotal = len(channelList)*len(ftFoldList)*len(numDayList)*len(rndLabList)*len(wbLabList)


def cell2mat(x):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] = x[i,j].item()
    return x


class Options:
    init = 0     # guess start point from data. 
    tFlag = 0     # terminate after relative objective value does not changes much.
    tol = 10**(-3)   # tolerance. 
    maxIter = 500 # maximum iteration number of optimization.
    rho_L2 = 0.1


def run_L21(new):
    global X_train_prepared, labels_train_prepared

    # d = size(X_train_prepared{1}, 2)  % dimensionality.

    Lambda = [0.5]

    opts = Options
    opts.init = 0     
    opts.tFlag = 0 
    opts.tol = 10**(-3)
    opts.maxIter = 500 
    opts.rho_L2 = 0.1

    if new and 'W0' in dir(opts): 
        delattr(opts, 'W0')
    # print('W0' in dir(opts))

    sparsity = np.zeros(len(Lambda))
    log_lam  = [np.log10(l) for l in Lambda]
    W_store = []
    for i,l in enumerate(Lambda):
        W,funcVal = Least_L21(X_train_prepared, labels_train_prepared, l, opts)
    #     % set the solution as the next initial point. 
    #     % this gives better efficiency. 
        opts.init = 1
        opts.W0 = W
        sparsity[i]= np.count_nonzero(W)
        W_store.append(W)

    return W_store, Lambda

def process(path, pbar):
    global X_train, X_val, cohort_train, cohort_test, X_train_prepared, labels_train_prepared
    # return np.random.randn(10).T

    r_store = np.zeros(10)

    MAE_store = np.zeros(10)
    MAE_store_train = np.zeros(10)
    MAE_store_train_val = np.zeros(10)
    MAE_store_val = np.zeros(10)

    for label in wbLabList:
        X_train_prepared = []
        X_val_prepared = []
        labels_train_prepared = []
        labels_val_prepared = []
        tran_val_prepared = []

        # print(X_train.shape, cohort_train.shape)
        for i in np.unique(cohort_train):
            X_train_prepared.append(X_train[cohort_train==i,:])
            labels_train_prepared.append(labels_train[cohort_train==i,label])

        pbar.update(1)

        W_store, Lambda = run_L21(not label)
        W = W_store[-1]

        n_cohort_test  = len(np.unique(cohort_test))
        n_cohort_train = len(np.unique(cohort_train))

        y_predict           = [None]*n_cohort_test
        y_predict_val       = [None]*n_cohort_test
        y_predict_train_val = [None]*n_cohort_test
        y_predict_train     = [None]*n_cohort_train
        
        sum_absolute_error           = np.zeros(n_cohort_test)
        sum_absolute_error_val       = np.zeros(n_cohort_test)
        sum_absolute_error_train_val = np.zeros(n_cohort_test)
        sum_absolute_error_train     = np.zeros(n_cohort_train)
        
        error_tran = []
        for i in np.unique(cohort_test):
            y_predict[i] = np.clip([X_test[cohort_test==i,:] @ W[:,i], labels_test[cohort_test==i,label]], 0, 100)
            sum_absolute_error[i] = np.sum(np.abs(y_predict[i][1] - y_predict[i][0])) # true - pred
    #         % error_tran{i} = [abs(y_predict{i}(:,2) - y_predict{i}(:,1)),transpose(stdTrantest(cohort_test==i))]
        
        for i in np.unique(cohort_train):
            y_predict_train[i] = np.clip([X_train[cohort_train==i,:] @ W[:,i], labels_train[cohort_train==i,label]], 0, 100)
            sum_absolute_error_train[i] = np.sum(np.abs(y_predict_train[i][1] - y_predict_train[i][0]))
    #         % error_tran{i} = [abs(y_predict{i}(:,2) - y_predict{i}(:,1)),transpose(stdTrantest(cohort_test==i))]
        
        MAE = np.nansum(sum_absolute_error) / len(X_test)
        MAE_train = np.nansum(sum_absolute_error_train) / len(X_train)
        MAE_store[label] = MAE
        MAE_store_train[label] = MAE_train
    #     % get_r
    #     % r_store(label) = stat(1)
        label_path = os.path.join(path,str(label+1))
        if not os.path.exists(label_path):
            os.makedirs(label_path)

        y_predict_stack = np.empty((len(y_predict),), dtype=np.object)
        for i in range(len(y_predict)):
            y_predict_stack[i] = y_predict[i]

        savemat(os.path.join(label_path,'W_store.mat'), {'W_store': W_store})
        savemat(os.path.join(label_path,'lambda.mat'), {'lambda': Lambda})
        savemat(os.path.join(label_path,'y_pred.mat'), {'y_predict': y_predict_stack})
    #     % save([label_path,'/tranMap'],'test')

    return MAE_store, pbar


with tqdm(total=pTotal, desc="MTL") as pbar:

    const = ['const']+[v for v in vars() if v[:2]!='__']

    for channel in channelList:
        varlist_to_del = ','.join([v for v in vars() if v[:2]!='__' and v not in ['channel']+const])
        if varlist_to_del: exec('del %s'%(varlist_to_del))
        
        storePath = os.path.join(os.getcwd(),'MTL_results_py2',rootPath,channel)
        dataPath = os.path.join(os.getcwd(),'MATLAB_data',rootPath)
        
        for fd in ftFoldList:
            for days in numDayList:
                for random_label in rndLabList:
                    varlist_to_del = ','.join([v for v in vars() if v[:2]!='__' and v not in \
                        const+['random_label', 'days', 'fd', 'channel', 'storePath', 'dataPath']])
                    if varlist_to_del: exec('del %s'%(varlist_to_del))
                    
                    path = os.path.join(storePath, fd, str(random_label),'label')
                    MMSE_path = os.path.join(storePath, fd)
                    
                    dataset_path = os.path.join(dataPath,channel,fd,str(random_label),'*.mat')
                    dataset = sorted(glob(dataset_path))

                    # Load dataset
                    if channel == '*':
                        X_train_loaded, X_val_loaded = [],[]

                    for data_path in dataset:
                        mat = loadmat(data_path)
                        varname = [v for v in mat.keys() if v[:2]!='__']
                        for v in varname:
                            exec("%s = %s"%(v,'mat[v]'))

                        if channel == '*':
                            if 'X_train' in data_path:
                                X_train_loaded.append(X_train)
                            if 'X_test' in data_path:
                                X_val_loaded.append(X_val)          

                    # object -> float
                    if 'X_train_loaded' in vars() and 'X_val_loaded' in vars():
                        X_train = np.concatenate(X_train_loaded,axis=-1).astype(float)
                        X_test = np.concatenate(X_val_loaded,axis=-1).astype(float)
                    else: 
                        X_train = X_train.astype(float)
                        X_test = X_val.astype(float)

                    labels_train = cell2mat(labels_train).astype(float)
                    labels_test = cell2mat(labels_val).astype(float)                

                    time_test = np.squeeze(time_test)
                    time_train = np.squeeze(time_train)

                    cohort_train = np.squeeze(cohort_train) #+ 1
                    cohort_test = np.squeeze(cohort_test) #+ 1

                    # process and update pbar
                    MAE_store, pbar = process(path, pbar)

                    mat_path = os.path.join(MMSE_path,str(random_label),'MAE_store.mat')
                    if not os.path.exists(os.path.dirname(mat_path)):
                        os.makedirs(os.path.dirname(mat_path))
                    savemat(mat_path,{'MAE_store':MAE_store})
                    
        if "*" in storePath:
            move(storePath, storePath.replace('*','all'))
     