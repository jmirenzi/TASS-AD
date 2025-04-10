import tensorflow as tf
import keras
import numpy as np

import time
import gc
import os
from keras import backend as K
from math import ceil
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.utils import timeseries_dataset_from_array
from adbench.run import RunPipeline
from adbench.myutils import Utils
from adbench.datasets.data_generator import DataGenerator

from model import TAADModelBaseClass

class ADBenchCustomModel(TAADModelBaseClass):
    def __init__(self,temporal_length,in_feature_len,temp_encoder_lstm_units=32,temporal_depth=2, **kwargs):
        kwargs.update({ "raw_error_len":in_feature_len,"temporal_length":temporal_length,"temp_encoder_lstm_units":temp_encoder_lstm_units,"temporal_depth":temporal_depth,
                       "model_type":"ADBench"})
        super().__init__(**kwargs)
        self.in_feature_len = in_feature_len

    def get_config(self):
        base_config = super().get_config()
        base_config.update({"in_feature_len": self.in_feature_len})
        return base_config

    def get_modality_description(self):
        return f"featLen{self.in_feature_len}"

def build_dataset_collection(raw_X,raw_labels,temporal_length,ae_adn_train_ratio=None,semi_supervised_flag=True,temporal_bleed_ratio=0.,stride=1,test_ratio=.3,
                             batch_size=12,shuffle_size=250,shuffle_seed=142,debug=False):
    data_col={}
    temporal_bleed_ratio = min(max(temporal_bleed_ratio,0.),1-1/temporal_length)
    idx_shift = int(1 - (temporal_bleed_ratio*temporal_length))
    if debug: print(f"{temporal_length = }, {temporal_bleed_ratio = }, {idx_shift = }")
    if semi_supervised_flag:
        data_col['N']=None; data_col['A']=None
        idx_lists={'N':[],'A':[]}
        change_list_bd = [0,]+((np.where(raw_labels[1:]!=raw_labels[:-1])[0]+idx_shift).tolist()) + [len(raw_labels),]
        if debug: print(f"{change_list_bd = }")
        for idx in range(len(change_list_bd)-1):
            idx_lists[('A' if (raw_labels[change_list_bd[idx]]) else 'N')].append((change_list_bd[idx],change_list_bd[idx+1]))
        for label in idx_lists.keys():
            for idx_pair in idx_lists[label]:
                if debug:
                    print(f"{idx_pair}, {raw_X.shape} {raw_X[idx_pair[0]:idx_pair[1]].shape}")
                dataset_  = timeseries_dataset_from_array(raw_X[idx_pair[0]:idx_pair[1]],raw_labels[idx_pair[0]:idx_pair[1]],temporal_length,stride,batch_size=None)
                data_col[label] = dataset_ if data_col[label] is None else data_col[label].concatenate(dataset_)

        # tot_len_adn = min(len(data_col['N']),len(data_col['A']))
        # tot_len_adn = min(data_col['N'].cardinality(),data_col['A'].cardinality())
        tot_len_adn = data_col['A'].cardinality()
        shuffled_norm = data_col['N'].shuffle(shuffle_size,seed=shuffle_seed)
        shuffled_abnorm = data_col['A'].shuffle(shuffle_size,seed=shuffle_seed)

        norm_adn = shuffled_norm.skip(int(tot_len_adn * test_ratio)).take(int(tot_len_adn * (1-test_ratio)))
        abnorm_adn = shuffled_abnorm.skip(int(tot_len_adn * test_ratio))

        data_col['train'] = data_col['N'].shuffle(shuffle_size,seed=shuffle_seed).skip(tot_len_adn)
        data_col['val_t'] = tf.data.Dataset.from_tensor_slices([norm_adn, abnorm_adn]).interleave(lambda ds: ds,cycle_length=2,block_length=1,num_parallel_calls=tf.data.AUTOTUNE)
    else:
        raise NotImplementedError('semi_supervised_flag must be True!')

    data_col['val_t'] = data_col['val_t'].shuffle(shuffle_size,seed=shuffle_seed).batch(batch_size)
    data_col['train'] = data_col['train'].shuffle(shuffle_size,seed=shuffle_seed).batch(batch_size)

    return data_col

def build_dataset_collection_from_temporal_safe(raw_X,raw_labels,batch_size=12,shuffle_size=250,shuffle_seed=142,debug=False):
    norm_arr = raw_X[raw_labels==0]
    abnorm_arr = raw_X[raw_labels==1]
    data_col={'N':tf.data.Dataset.from_tensor_slices((norm_arr, np.zeros(norm_arr.shape[:1],dtype=np.int64))),
              'A':tf.data.Dataset.from_tensor_slices((abnorm_arr, np.ones(abnorm_arr.shape[:1],dtype=np.int64)))}
    # tot_len_adn = min(len(data_col['N']),len(data_col['A']))
    # tot_len_adn = min(data_col['N'].cardinality(),data_col['A'].cardinality())
    tot_len_adn = abnorm_arr.shape[0]

    norm_adn = data_col['N'].take(tot_len_adn)
    abnorm_adn = data_col['A']

    data_col['train'] = data_col['N']
    data_col['val_t'] = tf.data.Dataset.from_tensor_slices([norm_adn, abnorm_adn]).interleave(lambda ds: ds,cycle_length=2,block_length=1,num_parallel_calls=tf.data.AUTOTUNE)

    data_col['val_t'] = data_col['val_t'].shuffle(shuffle_size,seed=shuffle_seed).batch(batch_size)
    data_col['train'] = data_col['train'].shuffle(shuffle_size,seed=shuffle_seed).batch(batch_size)

    return data_col

def make_dataset_temporal_safe(raw_X:np.ndarray,raw_labels:np.ndarray,temporal_length,temporal_bleed_ratio=0.,debug=False):
    temporal_bleed_ratio = min(max(temporal_bleed_ratio,0.),1-1/temporal_length)
    idx_shift = int(1 - (temporal_bleed_ratio*temporal_length))
    if debug: print(f"{temporal_length = }, {temporal_bleed_ratio = }, {idx_shift = }")
    X_temp_safe=None; y_temp_safe=None
    idx_lists={'N':[],'A':[]}
    change_list_bd = [0,]+((np.where(raw_labels[1:]!=raw_labels[:-1])[0]+idx_shift).tolist()) + [len(raw_labels),]
    if debug: print(f"{change_list_bd = }")
    for idx in range(len(change_list_bd)-1):
        idx_lists[('A' if (raw_labels[change_list_bd[idx]]) else 'N')].append((change_list_bd[idx],change_list_bd[idx+1]))
    for label in idx_lists.keys():
        for idx_pair in idx_lists[label]:
            if debug:
                print(f"{idx_pair}, {raw_X.shape} {raw_X[idx_pair[0]:idx_pair[1]].shape}")
            if (idx_pair[1]-idx_pair[0])<temporal_length:
                continue
            X_arr = np.swapaxes(np.lib.stride_tricks.sliding_window_view(raw_X[idx_pair[0]:idx_pair[1]],temporal_length,axis=0),1,2)
            label_arr = raw_labels[idx_pair[0]:idx_pair[1]-(temporal_length-1)]
            X_temp_safe = X_arr if X_temp_safe is None else np.concatenate((X_temp_safe,X_arr),axis=0)
            y_temp_safe = label_arr if y_temp_safe is None else np.concatenate((y_temp_safe,label_arr),axis=0)

    return X_temp_safe, y_temp_safe

class ADBenchWrapper():
    def __init__(self, seed:int, model_name:str=None,temporal_length=10,temporal_depth=2,temporal_width=16,
                 batch_size=12,shuffle_seed=142,shuffle_size=250,train_epoches=(10,10),tau=.5,self_supervised_flag=False,VAE_mode_flag=False,debug=False):
        self.seed = seed
        self.utils = Utils()
        self.model_name = model_name
        self.temporal_length=temporal_length
        self.temporal_depth=temporal_depth
        self.temporal_width=temporal_width
        self.debug=debug
        self.batch_size=batch_size
        self.shuffle_seed=shuffle_seed
        self.shuffle_size=shuffle_size
        self.train_epoches=train_epoches
        self.self_supervised_flag=self_supervised_flag
        self.VAE_mode_flag=VAE_mode_flag
        self.tau = tau
        self.round_labels = False


    def fit(self, X_train, y_train):
        # Initialization
        data_shape = X_train.shape
        if len(data_shape)==1:
            feature_len = 1
            X_train = X_train[:, np.newaxis]
        elif len(data_shape) == 2:
            feature_len = data_shape[1]
            dataset_collection = build_dataset_collection(X_train,y_train,self.temporal_length,shuffle_size=self.shuffle_size,batch_size=self.batch_size,
                                                        debug=self.debug,
                                                        #   temporal_bleed_ratio=1.0,
                                                        )
        elif len(data_shape) == 3:
            feature_len = data_shape[-1]
            dataset_collection = build_dataset_collection_from_temporal_safe(X_train,y_train,shuffle_size=self.shuffle_size,batch_size=self.batch_size,debug=self.debug)
        else:
            raise Exception("Invaild Data Shape!")
        self.dataset_collection=dataset_collection
        self.model = ADBenchCustomModel(self.temporal_length,feature_len,self.temporal_width,self.temporal_depth,
                                        self_supervised_flag=self.self_supervised_flag,VAE_mode_flag=self.VAE_mode_flag)
        self.model.full_train(dataset_collection,ae_epoch_num=self.train_epoches[0],adn_epoch_num=self.train_epoches[1],test_model_flag=False,
                                     debug=self.debug,batch_size=self.batch_size,shuffle_seed=self.shuffle_seed,shuffle_size=self.shuffle_size)

        return self

    def predict_score(self, X):
        # dd_x = timeseries_dataset_from_array(X,np.zeros_like(X),self.temporal_length,batch_size=X.shape[0])
        dd_x = tf.data.Dataset.from_tensor_slices((X, np.zeros(X.shape[:1],dtype=np.int64))).batch(X.shape[0])
        score: np.ndarray = np.array(self.model(next(iter(dd_x))[0])[-1])
        if not self.round_labels: return score
        labels = np.zeros_like(score); labels[score>=self.tau]=1.0
        return labels

class TemporalSafeRunPipeline(RunPipeline):
    def __init__(self,temporal_length,model_kargs={},dataset_list_org=None,**kwargs):
        super().__init__(**kwargs)
        self.data_generator = TemporalSafeDataGenerator(generate_duplicates=self.generate_duplicates,
                                                        n_samples_threshold=self.n_samples_threshold)
        self.data_generator.temporal_length = temporal_length
        self.model_kargs = model_kargs
        self.dataset_list_org=dataset_list_org
        for k,v in model_kargs.items():
            self.model_dict[k]=ADBenchWrapper
    def dataset_filter(self):
        # dataset list in the current folder
        dataset_list_org = list(itertools.chain(*self.data_generator.generate_dataset_list())) if self.dataset_list_org is None else self.dataset_list_org
        dataset_list, dataset_size = [], []
        for dataset in dataset_list_org:
            add = True
            for seed in self.seed_list:
                if not add: continue
                self.data_generator.seed = seed
                self.data_generator.dataset = dataset
                try:
                    data = self.data_generator.generator(la=1.00, at_least_one_labeled=True)
                except Exception as e:
                    print(f'Error with {dataset}: {e}')
                    add = False
                    continue
                if not self.generate_duplicates and len(data['y_train']) + len(data['y_test']) < self.n_samples_threshold:
                    add = False

                else:
                    if self.mode == 'nla' and sum(data['y_train']) >= self.nla_list[-1]:
                        pass

                    elif self.mode == 'rla' and sum(data['y_train']) > 0:
                        pass

                    else:
                        add = False

            # remove high-dimensional CV and NLP datasets if generating synthetic anomalies or robustness test
            if self.realistic_synthetic_mode is not None or self.noise_type is not None:
                if self.isin_NLPCV(dataset):
                    add = False

            if add:
                dataset_list.append(dataset)
                dataset_size.append(len(data['y_train']) + len(data['y_test']))
            else:
                print(f"remove the dataset {dataset}")

        # sort datasets by their sample size
        dataset_list = [dataset_list[_] for _ in np.argsort(np.array(dataset_size))]

        return dataset_list
    # model fitting function
    def model_fit(self):
        try:
            # model initialization, if model weights are saved, the save_suffix should be specified
            if self.model_name in ['DevNet', 'FEAWAD', 'REPEN']:
                self.clf = self.clf(seed=self.seed, model_name=self.model_name, save_suffix=self.suffix)
            else:
                if 'myAD' in self.model_name:
                    self.clf = self.clf(seed=self.seed, model_name=self.model_name,**self.model_kargs[self.model_name])
                else:
                    self.clf = self.clf(seed=self.seed, model_name=self.model_name)

        except Exception as error:
            print(f'Error in model initialization. Model:{self.model_name}, Error: {error}')
            pass

        try:
            # fitting
            start_time = time.time()
            self.clf = self.clf.fit(X_train=self.data['X_train'], y_train=self.data['y_train'])
            end_time = time.time(); time_fit = end_time - start_time

            # predicting score (inference)
            start_time = time.time()
            if self.model_name == 'DAGMM':
                score_test = self.clf.predict_score(self.data['X_train'], self.data['X_test'])
            else:
                score_test = self.clf.predict_score(self.data['X_test'])
            end_time = time.time(); time_inference = end_time - start_time

            # performance
            result = self.utils.metric(y_true=self.data['y_test'], y_score=score_test, pos_label=1)

            K.clear_session()
            print(f"Model: {self.model_name}, AUC-ROC: {result['aucroc']}, AUC-PR: {result['aucpr']}")

            del self.clf
            gc.collect()

        except Exception as error:
            print(f'Error in model fitting. Model:{self.model_name}, Error: {error}')
            time_fit, time_inference = None, None
            result = {'aucroc': np.nan, 'aucpr': np.nan}
            pass

        return time_fit, time_inference, result

class TemporalSafeDataGenerator(DataGenerator):
    def generator(self, X=None, y=None, minmax=False,
                  la=None, at_least_one_labeled=False,
                  realistic_synthetic_mode=None, alpha:int=5, percentage:float=0.1,
                  noise_type=None, duplicate_times:int=2, contam_ratio=1.00, noise_ratio:float=0.05):
        '''
        la: labeled anomalies, can be either the ratio of labeled anomalies or the number of labeled anomalies
        at_least_one_labeled: whether to guarantee at least one labeled anomalies in the training set
        '''

        # set seed for reproducible results
        self.utils.set_seed(self.seed)

        # load dataset
        if self.dataset is None:
            assert X is not None and y is not None, "For customized dataset, you should provide the X and y!"
            print('Testing on customized dataset...')
        else:
            if self.dataset in self.dataset_list_classical:
                data = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Classical', self.dataset + '.npz'), allow_pickle=True)
            elif self.dataset in self.dataset_list_cv:
                data = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CV_by_ResNet18', self.dataset + '.npz'), allow_pickle=True)
            elif self.dataset in self.dataset_list_nlp:
                data = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'NLP_by_BERT', self.dataset + '.npz'), allow_pickle=True)
            else:
                raise NotImplementedError

            X = data['X']
            y = data['y']


        # if the dataset is too small, generating duplicate smaples up to n_samples_threshold
        if len(y) < self.n_samples_threshold and self.generate_duplicates:
            print(f'generating duplicate samples for dataset {self.dataset}...')
            raise NotImplementedError("Too Few samples!")
            self.utils.set_seed(self.seed)
            idx_duplicate = np.random.choice(np.arange(len(y)), self.n_samples_threshold, replace=True)
            X = X[idx_duplicate]
            y = y[idx_duplicate]

        # if the dataset is too large, subsampling for considering the computational cost
        if len(y) > 10000:
            # print(f'subsampling for dataset {self.dataset}...')
            # self.utils.set_seed(self.seed)
            # idx_sample = np.random.choice(np.arange(len(y)), 10000, replace=False)
            # X = X[idx_sample]
            # y = y[idx_sample]
            X=X[:10000]
            y=y[:10000]

        # whether to generate realistic synthetic outliers
        if realistic_synthetic_mode is not None:
            raise NotImplementedError("NOT IMPLENTS")
            # we save the generated dependency anomalies, since the Vine Copula could spend too long for generation
            if realistic_synthetic_mode == 'dependency':
                filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synthetic')
                filename = 'dependency_anomalies_' + self.dataset + '_' + str(self.seed) + '.npz'

                if not os.path.exists(filepath):
                    os.makedirs(filepath)
                try:
                    data_dependency = np.load(os.path.join(filepath, filename), allow_pickle=True)
                    X = data_dependency['X']; y = data_dependency['y']
                except:
                    # raise NotImplementedError
                    print(f'Generating dependency anomalies...')
                    X, y = self.generate_realistic_synthetic(X, y,
                                                             realistic_synthetic_mode=realistic_synthetic_mode,
                                                             alpha=alpha, percentage=percentage)
                    np.savez_compressed(os.path.join(filepath, filename), X=X, y=y)
                    pass

            else:
                X, y = self.generate_realistic_synthetic(X, y,
                                                         realistic_synthetic_mode=realistic_synthetic_mode,
                                                         alpha=alpha, percentage=percentage)

        # whether to add different types of noise for testing the robustness of benchmark models
        if noise_type is None:
            pass

        elif noise_type == 'duplicated_anomalies':
            # X, y = self.add_duplicated_anomalies(X, y, duplicate_times=duplicate_times)
            pass

        elif noise_type == 'irrelevant_features':
            X, y = self.add_irrelevant_features(X, y, noise_ratio=noise_ratio)

        elif noise_type == 'label_contamination':
            pass

        else:
            raise NotImplementedError

        print(f'current noise type: {noise_type}')

        # show the statistic
        self.utils.data_description(X=X, y=y)

        # spliting the current data to the training set and testing set
        X, y = make_dataset_temporal_safe(X, y, self.temporal_length)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, shuffle=True, stratify=y)

        # we respectively generate the duplicated anomalies for the training and testing set
        if noise_type == 'duplicated_anomalies':
            raise NotImplementedError("NOT IMPLENTS")
            X_train, y_train = self.add_duplicated_anomalies(X_train, y_train, duplicate_times=duplicate_times)
            X_test, y_test = self.add_duplicated_anomalies(X_test, y_test, duplicate_times=duplicate_times)

        # notice that label contamination can only be added in the training set
        elif noise_type == 'label_contamination':
            raise NotImplementedError("NOT IMPLENTS")
            X_train, y_train = self.add_label_contamination(X_train, y_train, noise_ratio=noise_ratio)

        # minmax scaling
        if minmax:
            raise NotImplementedError("NOT IMPLENTS")
            scaler = MinMaxScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        # idx of normal samples and unlabeled/labeled anomalies
        idx_normal = np.where(y_train == 0)[0]
        idx_anomaly = np.where(y_train == 1)[0]

        if type(la) == float:
            if at_least_one_labeled:
                idx_labeled_anomaly = np.random.choice(idx_anomaly, ceil(la * len(idx_anomaly)), replace=False)
            else:
                idx_labeled_anomaly = np.random.choice(idx_anomaly, int(la * len(idx_anomaly)), replace=False)
        elif type(la) == int:
            if la > len(idx_anomaly):
                raise AssertionError(f'the number of labeled anomalies are greater than the total anomalies: {len(idx_anomaly)} !')
            else:
                idx_labeled_anomaly = np.random.choice(idx_anomaly, la, replace=False)
        else:
            raise NotImplementedError

        idx_unlabeled_anomaly = np.setdiff1d(idx_anomaly, idx_labeled_anomaly)
        # whether to remove the anomaly contamination in the unlabeled data
        if noise_type == 'anomaly_contamination':
            raise NotImplementedError("NOT IMPLENTS")
            idx_unlabeled_anomaly = self.remove_anomaly_contamination(idx_unlabeled_anomaly, contam_ratio)

        # unlabel data = normal data + unlabeled anomalies (which is considered as contamination)
        idx_unlabeled = np.append(idx_normal, idx_unlabeled_anomaly)

        del idx_anomaly, idx_unlabeled_anomaly

        # the label of unlabeled data is 0, and that of labeled anomalies is 1
        y_train[idx_unlabeled] = 0
        y_train[idx_labeled_anomaly] = 1

        return {'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test}

def build_model_kargs(t_depth_list=None,t_width_list=None,ss_cycle=False,base_kargs={}):
    model_kargs={}
    t_depth_list = t_depth_list if t_depth_list!=None else [base_kargs['temporal_depth'],]
    t_width_list = t_width_list if t_width_list!=None else [base_kargs['temporal_width'],]
    flag_list = [True,False,] if ss_cycle else [base_kargs['self_supervised_flag'],]
    for t_dep in t_depth_list:
        for t_wid in t_width_list:
            for ss_flag in flag_list:
                    model_kargs[f'myAD_{t_dep}x{t_wid}_ss{ss_flag}']=base_kargs.copy()
                    model_kargs[f'myAD_{t_dep}x{t_wid}_ss{ss_flag}'].update({'temporal_width':t_wid,'temporal_depth':t_dep,'self_supervised_flag':ss_flag})
    return model_kargs

if __name__ == '__main__':
  TEMPORAL_LENGTH = 40
  TEMPORAL_WIDTH = 8
  TEMPORAL_DEPTH = 2
  BATCH_SIZE = 6
  AE_EPOCHES = 40
  ADN_EPOCHES = 20
  NUM_TRAILS = 3
  SS_FLAG = True
  DATASET_LIST = ['31_satimage-2', '41_Waveform', '3_backdoor', '24_mnist', '33_skin', '10_cover', '1_ALOI', '26_optdigits', '35_SpamBase', '25_musk', '44_Wilt', '36_speech']
  DATASET_LIST = ['30_satellite','20_letter']
  # %% Single Run
  base_kargs={'temporal_length':TEMPORAL_LENGTH,'batch_size':BATCH_SIZE,'shuffle_seed':42,'self_supervised_flag': SS_FLAG,
              'temporal_width': TEMPORAL_WIDTH,'temporal_depth': TEMPORAL_DEPTH, 'shuffle_size':500,'train_epoches':(AE_EPOCHES,ADN_EPOCHES)}
  pipeline = TemporalSafeRunPipeline(TEMPORAL_LENGTH,model_kargs={f'myAD_{TEMPORAL_DEPTH}x{TEMPORAL_WIDTH}_ss{SS_FLAG}':base_kargs},
                                      suffix=f'single_tseq{TEMPORAL_LENGTH}', parallel='None', realistic_synthetic_mode=None, noise_type=None,n_samples_threshold=TEMPORAL_LENGTH*10)
  pipeline.dataset_list_org = ['30_satellite','20_letter']
  pipeline.rla_list = [1.0,.75,.5]
  # pipeline.rla_list = [1.0,]
  pipeline.seed_list = list(np.arange(NUM_TRAILS) + 1)
  # pipeline.seed_list = [1,]
  results = pipeline.run()
  # %%
  for temporal_len in [40,]:
      base_kargs={'temporal_length':temporal_len,'batch_size':BATCH_SIZE,'shuffle_seed':42,'self_supervised_flag': SS_FLAG,
                  'temporal_width': TEMPORAL_WIDTH,'temporal_depth': TEMPORAL_DEPTH,'shuffle_size':500,'train_epoches':(AE_EPOCHES,ADN_EPOCHES)}
      mult_pipline = TemporalSafeRunPipeline(temporal_len,model_kargs=build_model_kargs(t_width_list=[2,8,32,64,128,256],ss_cycle=False,base_kargs=base_kargs),
                                          suffix=f'mulit_tseq{temporal_len}', parallel='None', realistic_synthetic_mode=None, noise_type=None,n_samples_threshold=temporal_len*10)
      mult_pipline.dataset_list_org = DATASET_LIST
      mult_pipline.rla_list = [1.0,.75,.5]
      mult_pipline.seed_list = list(np.arange(NUM_TRAILS) + 1)
      results = mult_pipline.run()
