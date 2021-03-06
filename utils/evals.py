import numpy
import scipy.sparse as sp
import logging
from six.moves import xrange
from collections import OrderedDict
import sys
import pdb
from sklearn import metrics
from threading import Lock
from threading import Thread
import torch
import math
from pdb import set_trace as stop
import os
import pandas as pd
# import pylab as pl
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, hamming_loss, f1_score

FORMAT = '[%(asctime)s] %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
LOGGER = logging.getLogger(__name__)



def compute_metrics(predictions, targets, loss, args, elapsed_time, br_thresholds=None, verbose=True):
    targets = targets.cpu().numpy()
    predictions = predictions.cpu().numpy()

    if br_thresholds is None:
        br_thresholds = {'ACC': 0, 'HA': 0, 'ebF1': 0, 'miF1': 0, 'maF1': 0}
        metrics_dict = {'ACC': 0, 'HA': 0, 'ebF1': 0, 'miF1': 0, 'maF1': 0, 'loss': loss, 'time': elapsed_time}
        for tau in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            pred = predictions.copy()
            pred[pred < tau] = 0
            pred[pred >= tau] = 1
            ACC = accuracy_score(targets, pred)
            HA = 1 - hamming_loss(targets, pred)
            ebF1 = f1_score(targets, pred, average='samples')
            miF1 = f1_score(targets, pred, average='micro')
            maF1 = f1_score(targets, pred, average='macro')
            if ACC >= metrics_dict['ACC']:
                metrics_dict['ACC'] = ACC
                br_thresholds['ACC'] = tau
            if HA >= metrics_dict['HA']:
                metrics_dict['HA'] = HA
                br_thresholds['HA'] = tau
            if ebF1 >= metrics_dict['ebF1']:
                metrics_dict['ebF1'] = ebF1
                br_thresholds['ebF1'] = tau
            if miF1 >= metrics_dict['miF1']:
                metrics_dict['miF1'] = miF1
                br_thresholds['miF1'] = tau
            if maF1 >= metrics_dict['maF1']:
                metrics_dict['maF1'] = maF1
                br_thresholds['maF1'] = tau
    else:
        pred = predictions.copy()
        pred[pred < br_thresholds['ACC']] = 0
        pred[pred >= br_thresholds['ACC']] = 1
        ACC = accuracy_score(targets, pred)
        pred = predictions.copy()
        pred[pred < br_thresholds['HA']] = 0
        pred[pred >= br_thresholds['HA']] = 1
        HA = 1 - hamming_loss(targets, pred)
        pred = predictions.copy()
        pred[pred < br_thresholds['ebF1']] = 0
        pred[pred >= br_thresholds['ebF1']] = 1
        ebF1 = f1_score(targets, pred, average='samples')
        pred = predictions.copy()
        pred[pred < br_thresholds['miF1']] = 0
        pred[pred >= br_thresholds['miF1']] = 1
        miF1 = f1_score(targets, pred, average='micro')
        pred = predictions.copy()
        pred[pred < br_thresholds['maF1']] = 0
        pred[pred >= br_thresholds['maF1']] = 1
        maF1 = f1_score(targets, pred, average='macro')
        metrics_dict = {'ACC': ACC, 'HA': HA, 'ebF1': ebF1, 'miF1': miF1, 'maF1': maF1, 'loss': loss, 'time': elapsed_time}

    if verbose:
        print('####################################')
        print('time:            ' + str(elapsed_time))
        print('loss:            ' + str(loss))
        print('ACC:             ' + str(ACC))
        print('HA:              ' + str(HA))
        print('ebF1:            ' + str(ebF1))
        print('miF1:            ' + str(miF1))
        print('maF1:            ' + str(maF1))
        print('####################################')

    return metrics_dict, br_thresholds



class Logger:
    def __init__(self,args):
        self.model_name = args.model_name

        if args.model_name:
            try:
                os.makedirs(args.model_name)
            except OSError as exc:
                pass

            try:
                os.makedirs(args.model_name+'/epochs/')
            except OSError as exc:
                pass

            self.file_names = {}
            self.file_names['train'] = os.path.join(args.model_name,'train_results.csv')
            self.file_names['valid'] = os.path.join(args.model_name,'valid_results.csv')
            self.file_names['test'] = os.path.join(args.model_name,'test_results.csv')

            self.file_names['valid_all_aupr'] = os.path.join(args.model_name,'valid_all_aupr.csv')
            self.file_names['valid_all_auc'] = os.path.join(args.model_name,'valid_all_auc.csv')
            self.file_names['test_all_aupr'] = os.path.join(args.model_name,'test_all_aupr.csv')
            self.file_names['test_all_auc'] = os.path.join(args.model_name,'test_all_auc.csv')
            

            f = open(self.file_names['train'],'w+'); f.close()
            f = open(self.file_names['valid'],'w+'); f.close()
            f = open(self.file_names['test'],'w+'); f.close()
            f = open(self.file_names['valid_all_aupr'],'w+'); f.close()
            f = open(self.file_names['valid_all_auc'],'w+'); f.close()
            f = open(self.file_names['test_all_aupr'],'w+'); f.close()
            f = open(self.file_names['test_all_auc'],'w+'); f.close()
            os.utime(args.model_name,None)
        
        self.best_valid = {'loss':1000000,'ACC':0,'HA':0,'ebF1':0,'miF1':0,'maF1':0,'meanAUC':0,'medianAUC':0,'meanAUPR':0,'medianAUPR':0,'meanFDR':0,'medianFDR':0,'allAUC':None,'allAUPR':None}

        self.best_test = {'loss':1000000,'ACC':0,'HA':0,'ebF1':0,'miF1':0,'maF1':0,'meanAUC':0,'medianAUC':0,'meanAUPR':0,'medianAUPR':0,'meanFDR':0,'medianFDR':0,'allAUC':None,'allAUPR':None,'epoch':0}


    def evaluate(self,train_metrics,valid_metrics,test_metrics,epoch,num_params):
        if self.model_name:

            with open(self.file_names['valid_all_auc'],'a') as f:
                f.write(str(epoch))
                for i,val in enumerate(valid_metrics['allAUC']):
                    f.write(','+str(val))
                f.write('\n')
                f.close()

            with open(self.file_names['valid_all_aupr'],'a') as f:
                f.write(str(epoch))
                for i,val in enumerate(valid_metrics['allAUPR']):
                    f.write(','+str(val))
                f.write('\n')
                f.close()

            with open(self.file_names['test_all_auc'],'a') as f:
                f.write(str(epoch))
                for i,val in enumerate(test_metrics['allAUC']):
                    f.write(','+str(val))
                f.write('\n')
                f.close()

            with open(self.file_names['test_all_aupr'],'a') as f:
                f.write(str(epoch))
                for i,val in enumerate(test_metrics['allAUPR']):
                    f.write(','+str(val))
                f.write('\n')
                f.close()


        for metric in valid_metrics.keys():
            if not 'all' in metric and not 'time'in metric:
                if  valid_metrics[metric] >= self.best_valid[metric]:
                    self.best_valid[metric]= valid_metrics[metric]
                    self.best_test[metric]= test_metrics[metric]
                    if metric == 'ACC':
                        self.best_test['epoch'] = epoch

         
        print('\n')
        print('**********************************')
        print('best ACC:  '+str(self.best_test['ACC']))
        print('best HA:   '+str(self.best_test['HA']))
        print('best ebF1: '+str(self.best_test['ebF1']))
        print('best miF1: '+str(self.best_test['miF1']))
        print('best maF1: '+str(self.best_test['maF1']))
        print('best meanAUC:  '+str(self.best_test['meanAUC']))
        print('best meanAUPR: '+str(self.best_test['meanAUPR']))
        print('best meanFDR: '+str(self.best_test['meanFDR']))
        print('**********************************')

        return self.best_valid,self.best_test
