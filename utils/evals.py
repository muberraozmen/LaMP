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


def compute_metrics(all_predictions,all_targets,loss,args,elapsed,all_metrics=True,verbose=True):
    all_targets = all_targets.numpy()
    all_predictions = all_predictions.numpy()




    if all_metrics:
        meanAUC,medianAUC,varAUC,allAUC = compute_auc(all_targets,all_predictions)
        meanAUPR,medianAUPR,varAUPR,allAUPR = compute_aupr(all_targets,all_predictions)
        meanFDR,medianFDR,varFDR,allFDR = compute_fdr(all_targets,all_predictions)
    else:
        meanAUC,medianAUC,varAUC,allAUC = 0,0,0,0
        meanAUPR,medianAUPR,varAUPR,allAUPR = 0,0,0,0
        meanFDR,medianFDR,varFDR,allFDR = 0,0,0,0


    optimal_threshold = args.br_threshold

    # optimal_thresholds = Find_Optimal_Cutoff(all_targets,all_predictions)
    # optimal_threshold = numpy.mean(numpy.array(optimal_thresholds))


    if args.decoder in ['mlp','rnn_b','graph']:
        all_predictions[all_predictions < optimal_threshold] = 0
        all_predictions[all_predictions >= optimal_threshold] = 1
    else:
        all_predictions[all_predictions > 0.0] = 1



    acc_ = list(subset_accuracy(all_targets, all_predictions, axis=1, per_sample=True))
    hl_ = list(hamming_loss(all_targets, all_predictions, axis=1, per_sample=True))
    exf1_ = list(example_f1_score(all_targets, all_predictions, axis=1, per_sample=True))
    acc = numpy.mean(acc_)
    hl = numpy.mean(hl_)
    exf1 = numpy.mean(exf1_)


    tp, fp, fn = compute_tp_fp_fn(all_targets, all_predictions, axis=0)
    mif1 = f1_score_from_stats(tp, fp, fn, average='micro')
    maf1 = f1_score_from_stats(tp, fp, fn, average='macro')



    eval_ret = OrderedDict([('Subset accuracy', acc),
                        ('Hamming accuracy', 1 - hl),
                        ('Example-based F1', exf1),
                        ('Label-based Micro F1', mif1),
                        ('Label-based Macro F1', maf1)])


    ACC = eval_ret['Subset accuracy']
    HA = eval_ret['Hamming accuracy']
    ebF1 = eval_ret['Example-based F1']
    miF1 = eval_ret['Label-based Micro F1']
    maF1 = eval_ret['Label-based Macro F1']
    if verbose:
        print('ACC:   '+str(ACC))
        print('HA:    '+str(HA))
        print('ebF1:  '+str(ebF1))
        print('miF1:  '+str(miF1))
        print('maF1:  '+str(maF1))



    if verbose:
        print('uAUC:  '+str(meanAUC))
        # print('mAUC:  '+str(medianAUC))
        print('uAUPR: '+str(meanAUPR))
        # print('mAUPR: '+str(medianAUPR))
        print('uFDR: '+str(meanFDR))
        # print('mFDR:  '+str(medianFDR))

    metrics_dict = {}
    metrics_dict['ACC'] = ACC
    metrics_dict['HA'] = HA
    metrics_dict['ebF1'] = ebF1
    metrics_dict['miF1'] = miF1
    metrics_dict['maF1'] = maF1
    metrics_dict['meanAUC'] = meanAUC
    metrics_dict['medianAUC'] = medianAUC
    metrics_dict['meanAUPR'] = meanAUPR
    metrics_dict['allAUC'] = allAUC
    metrics_dict['medianAUPR'] = medianAUPR
    metrics_dict['allAUPR'] = allAUPR
    metrics_dict['meanFDR'] = meanFDR
    metrics_dict['medianFDR'] = medianFDR
    metrics_dict['loss'] = loss
    metrics_dict['time'] = elapsed

    return metrics_dict


def compute_metrics(predictions, targets, loss, args, elapsed_time, br_thresholds=None, verbose=True):
    targets = targets.cpu().numpy()
    predictions = predictions.cpu().numpy()
    loss = loss/len(predictions)

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
            # if train_metrics is not None:
            #     with open(self.file_names['train'],'a') as f:
            #         f.write(str(epoch)+','+str(train_metrics['loss'])
            #                           +','+str(train_metrics['ACC'])
            #                           +','+str(train_metrics['HA'])
            #                           +','+str(train_metrics['ebF1'])
            #                           +','+str(train_metrics['miF1'])
            #                           +','+str(train_metrics['maF1'])
            #                           +','+str(train_metrics['meanAUC'])
            #                           +','+str(train_metrics['medianAUC'])
            #                           +','+str(train_metrics['meanAUPR'])
            #                           +','+str(train_metrics['medianAUPR'])
            #                           +','+str(train_metrics['meanFDR'])
            #                           +','+str(train_metrics['medianFDR'])
            #                           +','+'{elapse:3.3f}'.format(elapse=train_metrics['time'])
            #                           +','+str(num_params)
            #                           +'\n')
            
            # with open(self.file_names['valid'],'a') as f:
            #     f.write(str(epoch)+','+str(valid_metrics['loss'])
            #                       +','+str(valid_metrics['ACC'])
            #                       +','+str(valid_metrics['HA'])
            #                       +','+str(valid_metrics['ebF1'])
            #                       +','+str(valid_metrics['miF1'])
            #                       +','+str(valid_metrics['maF1'])
            #                       +','+str(valid_metrics['meanAUC'])
            #                       +','+str(valid_metrics['medianAUC'])
            #                       +','+str(valid_metrics['meanAUPR'])
            #                       +','+str(valid_metrics['medianAUPR'])
            #                       +','+str(valid_metrics['meanFDR'])
            #                       +','+str(valid_metrics['medianFDR'])
            #                       +','+'{elapse:3.3f}'.format(elapse=train_metrics['time'])
            #                       +','+'{elapse:3.3f}'.format(elapse=valid_metrics['time'])
            #                       +','+str(num_params)
            #                       +'\n')

            # with open(self.file_names['test'],'a') as f:
            #     f.write(str(epoch)+','+str(test_metrics['loss'])
            #                       +','+str(test_metrics['ACC'])
            #                       +','+str(test_metrics['HA'])
            #                       +','+str(test_metrics['ebF1'])
            #                       +','+str(test_metrics['miF1'])
            #                       +','+str(test_metrics['maF1'])
            #                       +','+str(test_metrics['meanAUC'])
            #                       +','+str(test_metrics['medianAUC'])
            #                       +','+str(test_metrics['meanAUPR'])
            #                       +','+str(test_metrics['medianAUPR'])
            #                       +','+str(test_metrics['meanFDR'])
            #                       +','+str(test_metrics['medianFDR'])
            #                       +','+'{elapse:3.3f}'.format(elapse=train_metrics['time'])
            #                       +','+'{elapse:3.3f}'.format(elapse=test_metrics['time'])
            #                       +','+str(num_params)
            #                       +'\n')


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
