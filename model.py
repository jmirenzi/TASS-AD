import numpy as np
import pandas as pd
import yaml
import os

import tensorflow as tf
from keras import metrics
from keras.constraints import NonNeg
from keras.models import Model, Sequential
from keras.layers import Dense, Reshape, Layer, MaxPooling1D, AveragePooling1D, Conv1D, Normalization, LayerNormalization, LSTM, RepeatVector, TimeDistributed
from keras.optimizers import Adam
from keras.constraints import Constraint
from keras.callbacks import Callback
from keras.backend import floatx
from keras.saving.object_registration import register_keras_serializable
from keras.callbacks import EarlyStopping, CSVLogger
import tensorflow_probability as tfp

from tqdm import tqdm

from datetime import datetime

from keras.losses import binary_crossentropy

def bal_dataset_from_thres(model, base_dataset : tf.data.Dataset,batch_size=None,shuffle_seed=142,shuffle_size=250,cycle_dataset_flag=True,debug=False):
        if model.ss_thres_key is None: raise ValueError("model.ss_thres_key must be set!")
        data_list = []; true_label_list=[]; new_label_list=[]
        for data in base_dataset:
            if data_list == []:
                for _ in range(len(data)-1):data_list.append([])
            if batch_size is None:
                batch_size = tf.shape(data[0])[0]
            for idx in range(len(data)-1): data_list[idx].append(np.array(data[idx]))
            true_label_list.append(np.array(tf.reshape(data[-1],shape=(-1,1))))
            new_label_list.append(np.array(model.thresholders[model.ss_thres_key].threshold_compare(model(data[:-1])[-2])))

        true_label_arr=np.reshape(np.concatenate(true_label_list,axis=0),newshape=(-1,)); new_label_arr=np.concatenate(new_label_list,axis=0)
        norm_label_list = np.where(new_label_arr==0)[0]
        abnorm_label_list = np.where(new_label_arr==1)[0]
        min_num_samples = min(norm_label_list.shape[0],abnorm_label_list.shape[0])
        if min_num_samples < batch_size*10 : print(f"{bcolors.FAIL} Inadequate Labeling! Using base_dataset instead{bcolors.ENDC}"); return base_dataset
        norm_labels = np.zeros(true_label_arr.shape[0],dtype=bool); norm_labels[norm_label_list[:min_num_samples]]=True
        abnorm_labels = np.zeros(true_label_arr.shape[0],dtype=bool); abnorm_labels[abnorm_label_list[:min_num_samples]]=True
        norm_data_list=[];abnorm_data_list=[]
        for idx in range(len(data_list)):
            norm_data_list.append(np.concatenate(data_list[idx],axis=0)[norm_labels])
            abnorm_data_list.append(np.concatenate(data_list[idx],axis=0)[abnorm_labels])
        norm_dataset = tf.data.Dataset.from_tensor_slices((*norm_data_list,true_label_arr[norm_labels]))
        abnorm_dataset = tf.data.Dataset.from_tensor_slices((*abnorm_data_list,true_label_arr[abnorm_labels]))
        if debug: print(f"{norm_label_list.shape[0] = }, {abnorm_label_list.shape[0] = } -> {len(norm_dataset) = }, {len(abnorm_dataset) = }")
        if cycle_dataset_flag:
            bal_dataset = tf.data.Dataset.from_tensor_slices([norm_dataset, abnorm_dataset]).interleave(lambda ds: ds,cycle_length=2,block_length=1,
                                                                                                        num_parallel_calls=tf.data.AUTOTUNE)
        else:
            bal_dataset = norm_dataset.concatenate(abnorm_dataset)
        return bal_dataset.shuffle(shuffle_size,seed=shuffle_seed).batch(batch_size)

@tf.keras.utils.register_keras_serializable(package="AD_Model")
class TAADModelBaseClass(Model):
    def __init__(self,model_type,temporal_length,temp_encoder_lstm_units,temporal_depth,raw_error_len,error_selector=2,thres_description=None,
                inner_model_flag=False,only_encoder_flag=False,self_supervised_flag=False,VAE_mode_flag=False,learning_rate=0.01,
                current_epoch_ae=0, current_epoch_adn=0, **kwargs):
        super().__init__(**kwargs)
        reduce_error=error_selector<3
        if error_selector == 0: error_mode = 'Flatten'
        elif error_selector == 1 or error_selector == 3: error_mode = 'Max'
        elif error_selector == 2 or error_selector == 4: error_mode = 'Mean'

        if (not error_mode in ["Flatten","Mean","Max"]) or (not reduce_error and error_mode=='Flatten'):
            raise ValueError("if reduce_error==False error_mode must be either 'Mean' or 'Max' and if reduce_error==True error_mode must be either 'Flatten', 'Mean', or 'Max'")

        self.error_len = raw_error_len*temporal_length if reduce_error and error_mode=='Flatten' else raw_error_len
        self.train_loss_tracker = metrics.Mean(name="total_loss")
        self.train_reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.train_classification_loss_tracker = metrics.Mean(name="classification_loss")
        self.train_acc = metrics.Mean(name="acc")
        self.thres_acc = metrics.Mean(name="thres_acc")
        self.model_type=model_type
        self.temporal_length=temporal_length
        self.temporal_depth=temporal_depth
        self.temp_encoder_lstm_units=temp_encoder_lstm_units
        self.raw_error_len=raw_error_len; self.error_selector=error_selector
        self.reduce_error = reduce_error; self.error_mode = error_mode
        self.inner_model_flag = inner_model_flag
        # TODO: remove need for this
        self.only_encoder_flag = only_encoder_flag and inner_model_flag
        self.current_epoch_ae = current_epoch_ae; self.current_epoch_adn = current_epoch_adn
        self.train_adn_mode = False
        self.self_supervised_mode = self_supervised_flag
        self.VAE_mode_flag=VAE_mode_flag
        self.thres_description = {"SamplePercentThresholdMetric":{'thres_percentile':.96, 'any_anomaly_thres':0.08}} if thres_description is None else thres_description
        self.ss_thres_key = None

        # Create Unsupervised Threshsolders
        self.reduce_labels_thres = None if reduce_error else error_mode

        self.create_thresholder(self.thres_description)
        self.ae_optimizer = Adam(learning_rate=learning_rate)

        # Create Modality Normalizer
        self.modality_normalizer = LayerNormalization(center=False)
        self.modality_embedder = Layer(name='modality_embedder_layer')

        if not self.inner_model_flag:
            # Create Temporal AutoEncoder
            if temporal_depth<2:
                print("ERROR temporal_depth must be > 2. 2 will be used")
            self.temporal_depth = max(temporal_depth,2)

            self.temporal_encoder = Sequential([LSTM(units=temp_encoder_lstm_units,input_shape=(temporal_length,raw_error_len),return_sequences=True)],name=f'{model_type}_temporal_encoder')
            for _ in range(self.temporal_depth-2):
                self.temporal_encoder.add(LSTM(units=temp_encoder_lstm_units,return_sequences=True))
            self.temporal_encoder.add(LSTM(units=temp_encoder_lstm_units))
            if self.VAE_mode_flag:
                self.temporal_encoder.add(Dense(temp_encoder_lstm_units*2))

            self.temporal_decoder = Sequential([RepeatVector(temporal_length,input_shape=(temp_encoder_lstm_units,))],name=f'{model_type}_temporal_decoder')
            for _ in range(self.temporal_depth):
                self.temporal_decoder.add(LSTM(units=temp_encoder_lstm_units,return_sequences=True))
            self.temporal_decoder.add(TimeDistributed(Dense(raw_error_len)))

            # Create Error Reshaper
            if self.reduce_error:
                if self.error_mode == 'Flatten':
                    self.error_reshape = Reshape((-1,self.error_len,))
                elif self.error_mode == 'Mean':
                    self.error_reshape = AveragePooling1D(pool_size=self.temporal_length)
                elif self.error_mode == 'Max':
                    self.error_reshape = MaxPooling1D(pool_size=self.temporal_length)
            else:
                self.error_reshape = Layer()

            # Create Anomaly Discriminator Network
            self.adn_optimizer = Adam(learning_rate=learning_rate)
            self.anomaly_discriminator = Sequential(name=f'{model_type}_anomaly_discriminator')
            if not self.reduce_error:
                self.anomaly_discriminator.add(Conv1D(1,1,data_format="channels_first",input_shape=(self.temporal_length,self.error_len),kernel_constraint=NonNeg()))
            self.anomaly_discriminator.add(Dense(2,activation="softmax",kernel_constraint=NonNeg(),bias_constraint=NonPosNonNeg(flipped=True)))
            self.anomaly_discriminator.add(Dense(1,activation="sigmoid",kernel_constraint=NonPosNonNeg()))

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,"temporal_length": self.temporal_length,"model_type":self.model_type,"error_selector":self.error_selector,"raw_error_len":self.raw_error_len,
                "inner_model_flag":self.inner_model_flag,"only_encoder_flag":self.only_encoder_flag, "VAE_mode_flag":self.VAE_mode_flag,
                "self_supervised_flag":self.self_supervised_mode,"thres_description":self.thres_description,
                "temp_encoder_lstm_units": self.temp_encoder_lstm_units,"temporal_depth":self.temporal_depth,
                "current_epoch_ae": self.current_epoch_ae, "current_epoch_adn": self.current_epoch_adn}

    def get_modality_description(self):
        return "MOADLITY_NOT_DEFINED"

    def get_description(self, epoch_flag=True):
        description = (
                f"{datetime.now().date()}_{self.model_type}_tseq{self.temporal_length}_{self.temporal_depth}x{self.temp_encoder_lstm_units}"
                f"_{self.get_modality_description()}_reduceError{self.reduce_error}_errorMode{self.error_mode}"
                f"{f'_ss{self.thresholders[self.ss_thres_key].get_description()}' if self.self_supervised_mode else ''}"
                f"_VAE{self.VAE_mode_flag}"
        )
        if epoch_flag:
            description += f"_{self.current_epoch_ae}-{self.current_epoch_adn}E"
        return description

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be called automatically at the start of each epoch
        return [self.train_loss_tracker,self.train_reconstruction_loss_tracker,self.train_classification_loss_tracker,self.train_acc,self.thres_acc]

    def enable_adn_training(self,enable_adn=True):
        self.train_adn_mode = enable_adn
        self.modality_normalizer.trainable = not enable_adn
        self.temporal_encoder.trainable = not enable_adn
        self.temporal_decoder.trainable = not enable_adn
        if not self.inner_model_flag:
            self.anomaly_discriminator.trainable = enable_adn
        if not self.run_eagerly: self.compile()

    def on_epoch_begin(self,epoch,logs):
        if self.train_adn_mode:
            self.current_epoch_adn += 1
        else:
            self.current_epoch_ae += 1
            for key_,thresholder in self.thresholders.items():
                thresholder.reset_state()

    def update_thresholders(self,error_tensor):
        for key_,thresholder in self.thresholders.items():
            thresholder.update_state(error_tensor)

    def create_thresholder(self,thres_description_dict):
        threshold_objects = {"MaxThresholdMetric":MaxThresholdMetric,"MeanThresholdMetric":MeanThresholdMetric,
                                    "GammaThresholdMetric":GammaThresholdMetric,"SamplePercentThresholdMetric":SamplePercentThresholdMetric}
        self.thresholders={}


        try:
            thresholder_name = list(thres_description_dict.keys())[0]
            threshold_type = threshold_objects[thresholder_name]
        except KeyError:
            raise Exception(f"{thresholder_name} is not a valid thresholder, please use {threshold_objects.keys()}")

        thres_description_dict[thresholder_name]['reduce_labels'] = self.reduce_labels_thres
        thresholder_ = threshold_type((self.error_len,),debug=self.run_eagerly,**(list(thres_description_dict.values())[0]))
        self.thresholders[thresholder_.get_description()] = thresholder_
        if self.self_supervised_mode and self.ss_thres_key is None:
            self.ss_thres_key = thresholder_.get_description()

    def recalculate_threshold(self,dataset,only_required=False):
        if self.thresholders=={}: return
        for key_,thresholder in self.thresholders.items():
            if not only_required: thresholder.reset_state()
            if isinstance(thresholder,SamplePercentThresholdMetric):
                thresholder.set_threshold(self,dataset)
        if not only_required:
            for batch in dataset:
                in_tensor = batch[:-1]
                out = self(in_tensor)
                error_tensor:tf.Tensor = tf.reshape(out[-2],shape=(-1,self.error_len))
                self.update_thresholders(error_tensor)

    def expand_labels(self,labels):
        return tf.repeat(tf.expand_dims(labels,axis=1),self.temporal_length,axis=1)

    def single_sample(self, z_mean, z_log_var):
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

    def get_trainable_vars(self):
        if self.inner_model_flag: return []
        return self.temporal_encoder.trainable_weights + self.temporal_decoder.trainable_weights

    def call(self,input_tensor):
        if isinstance(input_tensor,tuple): input_tensor=input_tensor[0]
        norm_input_tensor = self.modality_normalizer(input_tensor,training=tf.logical_not(self.train_adn_mode))
        embedded_tensor = self.modality_embedder(norm_input_tensor)
        if self.inner_model_flag: return embedded_tensor, input_tensor, 0, 0

        if self.VAE_mode_flag:
            latent_temporal_params          = self.temporal_encoder(embedded_tensor,training=tf.logical_not(self.train_adn_mode))
            latent_mean, latent_logvar      = tf.split(latent_temporal_params,num_or_size_splits=2,axis=1)
            latent_sample                   = self.single_sample(latent_mean, latent_logvar)
            reconstructed_embedded_tensor   = self.temporal_decoder(latent_sample,training=tf.logical_not(self.train_adn_mode))
            latent_temporal_feat            = (latent_mean, latent_logvar)
        else:
            latent_temporal_feat            = self.temporal_encoder(embedded_tensor,        training=tf.logical_not(self.train_adn_mode))
            reconstructed_embedded_tensor   = self.temporal_decoder(latent_temporal_feat,   training=tf.logical_not(self.train_adn_mode))

        squared_reconstruct_error       = self.error_reshape(tf.square((embedded_tensor - reconstructed_embedded_tensor)))

        classifition_probability = self.anomaly_discriminator(squared_reconstruct_error,training=self.train_adn_mode)
        return embedded_tensor, latent_temporal_feat, squared_reconstruct_error, tf.squeeze(classifition_probability)

    def get_loss(self,input_tensor,label_tensor):
        embedded_tensor, latent_tensor, squared_diff, classified = self(input_tensor)
        reconstruction_loss = tf.reduce_mean(squared_diff)
        if self.inner_model_flag:
            return reconstruction_loss

        # Update internal metrics
        self.train_acc.update_state(tf.reduce_mean(tf.cast(tf.round(classified)==tf.cast(label_tensor,tf.float32),tf.float32)))

        if self.self_supervised_mode and self.train_adn_mode:
            psuedo_label_tensor = tf.squeeze(tf.cast(self.thresholders[self.ss_thres_key].threshold_compare(squared_diff),tf.float32))
            self.thres_acc.update_state(tf.reduce_mean(tf.cast(psuedo_label_tensor==tf.cast(label_tensor,tf.float32),tf.float32)))
            label_tensor = psuedo_label_tensor

        classification_loss = binary_crossentropy(label_tensor, classified,from_logits=False)

        if self.VAE_mode_flag:
            latent_mean, latent_logvar = latent_tensor
            kl_loss = -0.5 * tf.reduce_mean(1 + latent_logvar - tf.square(latent_mean) - tf.exp(latent_logvar))
            return kl_loss, reconstruction_loss, classification_loss

        return reconstruction_loss, classification_loss

    def train_step(self, data):
        if self.inner_model_flag: raise Exception("Only run train_step with outer models!")
        # Unpack the data
        input_tensor = data[:-1]
        label_tensor = data[-1]
        with tf.GradientTape() as tape:
            losses = self.get_loss(input_tensor,label_tensor)
            reconstruction_loss = losses[-2]
            classification_loss = losses[-1]
            total_loss = sum(losses[:-1])

        if self.train_adn_mode:
            gradients = tape.gradient(classification_loss, self.anomaly_discriminator.trainable_weights)
            self.adn_optimizer.apply_gradients(zip(gradients, self.anomaly_discriminator.trainable_weights))

        else:
            gradients = tape.gradient(total_loss, self.get_trainable_vars())
            self.ae_optimizer.apply_gradients(zip(gradients, self.get_trainable_vars()))

        # Update custom metrics
        self.train_loss_tracker.update_state(total_loss)
        self.train_reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.train_classification_loss_tracker.update_state(classification_loss)

        logs = {"tot": self.train_loss_tracker.result(), "rec_l": self.train_reconstruction_loss_tracker.result()}

        if self.train_adn_mode:
            logs["class_l"] = self.train_classification_loss_tracker.result()
            logs["acc"] = self.train_acc.result()
            if self.self_supervised_mode:
                logs['thres_a'] = self.thres_acc.result()

        return logs

    def full_train(self,dataset_col,ae_epoch_num=2,adn_epoch_num=10,test_model_flag = None,use_thres_bal=None,
                   cycle_dataset_flag=True,
                    batch_size=12,shuffle_seed=142, shuffle_size=250,log_path="logs",
                    debug=False):
        if use_thres_bal is None: use_thres_bal=self.self_supervised_mode
        dataset_list = ['val_t','thres_bal','test'] if use_thres_bal else ['val_t','test']
        callbacks=[EpochEndCallback()]
        if log_path:
            os.makedirs(f'{log_path}/{self.get_description(epoch_flag=False)}', exist_ok=True)
            callbacks.append(CSVLogger(f'{log_path}/{self.get_description(epoch_flag=False)}/ae_logs.csv',append=False))
        if debug: old_weights = [w.numpy() for w in self.get_trainable_vars()]
        if ae_epoch_num >= 1:
            self.enable_adn_training(False)
            print(f"AE Trainging: {self.get_description()}")
            hist = self.fit(dataset_col['train'],epochs=ae_epoch_num,callbacks=callbacks)
            print(f"Done Training AutoEncoder for {self.current_epoch_ae} epoch{'es' if self.current_epoch_ae!=1 else ''}")
        if debug: print("ReCalc Thres")
        self.recalculate_threshold(dataset_col['train'])
        if use_thres_bal:
            if debug: print("Building Balanced Dataset")
            dataset_col['thres_bal'] = bal_dataset_from_thres(self,dataset_col['val_t'],batch_size=batch_size,shuffle_seed=shuffle_seed, shuffle_size=shuffle_size,cycle_dataset_flag=cycle_dataset_flag,debug=debug)

        if test_model_flag: simple_acc_test(self,dataset_col,dataset_list)
        elif self.self_supervised_mode and test_model_flag is None: simple_acc_test(self,dataset_col,['thres_bal',] if use_thres_bal else ['val_t',])

        if debug:
            new_weights = [w.numpy() for w in self.get_trainable_vars()]
            print(f"Change of AE weights:{(sum([ np.max(np.square(new_weights[i] - old_weights[i])) for i in range(len(old_weights))]))}")
            old_weights = new_weights
            old_weights_ad = [w.numpy() for w in self.anomaly_discriminator.weights]

        self.enable_adn_training()
        print(f"ADN Trainging: {self.get_description()}")
        callbacks=[EpochEndCallback()]
        if log_path:
            callbacks.append(CSVLogger(f'{log_path}/{self.get_description(epoch_flag=False)}/adn_logs.csv',append=False))
        hist = self.fit(dataset_col['thres_bal'] if use_thres_bal else dataset_col['val_t'],epochs=adn_epoch_num,callbacks=callbacks)
        if debug:
            self.enable_adn_training(False)
            new_weights = [w.numpy() for w in self.get_trainable_vars()]
            self.enable_adn_training(True)
            print(f"Change of AE weights:{(sum([ np.max(np.square(new_weights[i] - old_weights[i])) for i in range(len(old_weights))]))}")
            print(f"Change of ADN weights:{sum([ np.max(np.square(self.anomaly_discriminator.weights[i].numpy() - old_weights_ad[i])) for i in range(len(old_weights_ad))])}")
        print(f"Done Training ADN for {self.current_epoch_adn} epoch{'s' if self.current_epoch_adn!=1 else ''}")
        if test_model_flag: simple_acc_test(self,dataset_col,dataset_list)

@register_keras_serializable(package="AD_Model")
class MaxThresholdMetric(metrics.Metric):
    def __init__(self,error_shape:tuple, any_anomaly_thres=0.0, reduce_labels = False,debug=False,**kwargs):
        dtype_ = tf.float32
        kwargs['dtype'] =dtype_
        kwargs['name'] = 'max_threshold_metric'
        super().__init__(**kwargs)
        self.reduce_labels = reduce_labels
        self.max_threshold : tf.Variable = self.add_weight(shape=(1,)+error_shape,name='thrs', initializer='zeros',dtype=dtype_)
        self.error_shape = error_shape
        self.any_anomaly_thres = any_anomaly_thres
        self.debug = debug

    def config(self,config_dict):
        for key, value in config_dict.items():
            setattr(self,key,value)

    def get_description(self):
        return (f"MaxThres_{self.any_anomaly_thres}").replace('.','p')

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "error_shape": self.error_shape, "any_anomaly_thres":self.any_anomaly_thres,"reduce_labels":self.reduce_labels}

    def update_state(self, error_tensor : tf.Tensor,sample_weight=None):
        if sample_weight is not None:
            raise Exception("Sample weights not used!")
        self.max_threshold.assign(tf.reduce_max(tf.concat([self.max_threshold,error_tensor],axis=-2), axis=-2,keepdims=True))

    def threshold_compare(self,input_tensor : tf.Tensor):
        comparison = tf.cast(tf.greater(input_tensor,tf.broadcast_to(self.max_threshold,input_tensor.shape)),tf.float32)
        pred_labels = tf.cast(tf.reduce_mean(comparison,axis=-1)  >= self.any_anomaly_thres,dtype=tf.int32)

        if self.reduce_labels == "Mean":
            pred_labels = tf.reduce_mean(pred_labels,axis=-1)
        elif self.reduce_labels == "Max":
            pred_labels = tf.reduce_max(pred_labels,axis=-1)

        return tf.round(pred_labels)

    def __call__(self,diff_tensor):
        return diff_tensor, self.threshold_compare(diff_tensor)

    def result(self):
        return self.max_threshold

    def reset_state(self):
        self.max_threshold.assign(tf.zeros(self.max_threshold.shape))

@register_keras_serializable(package="AD_Model")
class MeanThresholdMetric(metrics.Metric):
    def __init__(self, error_shape,max_std_dist=2, any_anomaly_thres=0.06, reduce_labels=False, name='',debug=False,**kwargs):
        dtype_ = tf.float32
        kwargs['name'] ="mean_threshold_metric "+name
        kwargs['dtype'] =dtype_
        super().__init__(**kwargs)
        self.count              = self.add_weight(shape=(1),name='Number_of_samples',initializer='zeros',dtype=dtype_)
        self.sum_error          = self.add_weight(shape=(1,)+error_shape,name='Sum'  ,initializer='zeros',dtype=dtype_)
        self.mean_error         = self.add_weight(shape=(1,)+error_shape,name='mean' ,initializer='zeros',dtype=dtype_)
        self.summed_square_diff = self.add_weight(shape=(1,)+error_shape,name='sqSum',initializer='zeros',dtype=dtype_)
        self.error_shape = error_shape
        self.max_std_dist = max_std_dist
        self.any_anomaly_thres = any_anomaly_thres
        self.reduce_labels = reduce_labels
        self.debug = debug

    def config(self,config_dict):
        for key, value in config_dict.items():
            setattr(self,key,value)

    def get_description(self):
        return (f"MeanThres_{self.max_std_dist}_{self.any_anomaly_thres}").replace('.','p')

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "error_shape": self.error_shape, "max_std_dist": self.max_std_dist,
                "any_anomaly_thres":self.any_anomaly_thres,"reduce_labels":self.reduce_labels}

    def update_state(self, error_tensor : tf.Tensor,sample_weight=None):
        if sample_weight is not None:
            raise Exception("Sample weights not used!")
        batch_size = tf.cast(tf.shape(error_tensor)[0:1],self.count.dtype)
        self.count.assign_add(batch_size)
        self.sum_error.assign(tf.reduce_sum(tf.concat((self.sum_error,error_tensor),axis=0),axis=0,keepdims=True))
        self.mean_error.assign(self.sum_error / self.count)
        new_square_diff = tf.square(error_tensor - self.mean_error)
        self.summed_square_diff.assign(tf.reduce_sum(tf.concat((self.summed_square_diff,new_square_diff),axis=0),axis=0,keepdims=True))

    def threshold_compare(self,input_tensor : tf.Tensor):
        mean_, std_ = self.result()
        diff_ = tf.abs((input_tensor - mean_) / std_)
        comparison = tf.cast(tf.greater(diff_,tf.ones(input_tensor.shape)*self.max_std_dist),tf.float32)
        pred_labels = tf.cast(tf.reduce_mean(comparison,axis=-1)  >= self.any_anomaly_thres,dtype=tf.int32)

        if self.reduce_labels == "Mean":
            pred_labels = tf.reduce_mean(pred_labels,axis=-1)
        elif self.reduce_labels == "Max":
            pred_labels = tf.reduce_max(pred_labels,axis=-1)

        return tf.round(pred_labels)

    def __call__(self,diff_tensor):
        return diff_tensor, self.threshold_compare(diff_tensor)

    def result(self):
        return self.mean_error, tf.sqrt(self.summed_square_diff / (self.count - 1))

    def reset_state(self):
        self.count.assign(tf.zeros(self.count.shape))
        self.sum_error.assign(tf.zeros(self.sum_error.shape))
        self.mean_error.assign(tf.zeros(self.mean_error.shape))
        self.summed_square_diff.assign(tf.zeros(self.summed_square_diff.shape))

@register_keras_serializable(package="AD_Model")
class GammaThresholdMetric(metrics.Metric):
    def __init__(self, error_shape,thres_percentile=.94, any_anomaly_thres=0.02, reduce_labels=False, name="",debug=False,**kwargs):
        dtype_ = tf.float32
        kwargs['name'] ="gamma_threshold_metric "+name
        kwargs['dtype'] =dtype_
        super().__init__(**kwargs)
        self.count                  = self.add_weight(shape=(1),name='Number_of_samples',initializer='zeros',dtype=dtype_)
        self.sum_error              = self.add_weight(shape=error_shape,name='Sum'  ,initializer='zeros',dtype=dtype_)
        self.sum_error_log_error    = self.add_weight(shape=error_shape,name='sum_of_x_iln_x_i' ,initializer='zeros',dtype=dtype_)
        self.sum_log_error          = self.add_weight(shape=error_shape,name='sum_of_ln_x_i',initializer='zeros',dtype=dtype_)
        self.var                    = self.add_weight(shape=error_shape,name='variance_sum',initializer='zeros',dtype=dtype_)
        self.gamma_dist_built       = self.add_weight(shape=(1),name="flag_built_gamma_dist",initializer='zeros',dtype=tf.uint8)
        self.error_shape = error_shape
        self.thres_percentile = thres_percentile
        self.any_anomaly_thres = any_anomaly_thres
        self.reduce_labels = reduce_labels
        gamma_shape_tensor, gamma_scale_tensor = self.result()
        self.gamma_dist = tfp.distributions.gamma.Gamma(gamma_shape_tensor, 1/gamma_scale_tensor)
        self.debug = debug

    def config(self,config_dict):
        for key, value in config_dict.items():
            setattr(self,key,value)

    def get_description(self):
        return (f"GammaThres_{self.thres_percentile}_{self.any_anomaly_thres}").replace('.','p')

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "error_shape": self.error_shape,"thres_percentile":self.thres_percentile,
                "any_anomaly_thres":self.any_anomaly_thres,"reduce_labels":self.reduce_labels}

    def update_state(self, error_tensor : tf.Tensor,sample_weight=None):
        if sample_weight is not None:
            raise Exception("Sample weights not used!")
        error_tensor = tf.cast(error_tensor,self.sum_error.dtype)
        batch_size = tf.cast(tf.shape(error_tensor)[0:1],self.count.dtype)
        self.count.assign_add(batch_size)
        self.sum_error.assign_add(tf.reduce_sum(error_tensor,axis=0))
        self.sum_log_error.assign_add(tf.reduce_sum(tf.math.log(error_tensor),axis=0))
        self.sum_error_log_error.assign_add(tf.reduce_sum(error_tensor * tf.math.log(error_tensor),axis=0))
        self.var.assign_add(tf.reduce_sum(tf.square(error_tensor - self.sum_error / self.count),axis=0))
        self.gamma_dist_built.assign(tf.zeros_like(self.gamma_dist_built,dtype=tf.uint8))

    def threshold_compare(self,input_tensor : tf.Tensor):
        if (self.gamma_dist_built == tf.zeros_like(self.gamma_dist_built,dtype=tf.uint8)):
            gamma_shape_tensor, gamma_scale_tensor = self.result()
            if tf.reduce_any(gamma_shape_tensor<0):
                self.gamma_dist = tfp.distributions.gamma.Gamma.experimental_from_mean_variance(self.sum_error / self.count,self.var / (self.count-1))
                self.gamma_dist_built.assign(tf.ones_like(self.gamma_dist_built,dtype=tf.uint8)*2)
            else:
                self.gamma_dist = tfp.distributions.gamma.Gamma(gamma_shape_tensor, 1/gamma_scale_tensor,validate_args=True)
                self.gamma_dist_built.assign(tf.ones_like(self.gamma_dist_built,dtype=tf.uint8))

        if self.any_anomaly_thres is not None:
            pred_labels =  tf.cast(tf.reduce_mean(tf.cast(self.gamma_dist.cdf(input_tensor) >= self.thres_percentile,tf.float32),axis=-1) >= self.any_anomaly_thres,tf.float32)
        else:
            pred_labels = tf.cast(tf.reduce_mean(self.gamma_dist.cdf(input_tensor),axis=-1) >= self.thres_percentile,tf.float32)

        if self.reduce_labels == "Mean":
            pred_labels = tf.reduce_mean(pred_labels,axis=-1)
        elif self.reduce_labels == "Max":
            pred_labels = tf.reduce_max(pred_labels,axis=-1)

        return tf.round(pred_labels)

        return tf.round(pred_labels)

    def __call__(self,diff_tensor):
        return diff_tensor, self.threshold_compare(diff_tensor)

    def result(self):
        mean_ = self.sum_error / self.count
        scale_theta = (self.sum_error_log_error / self.count) - mean_*(self.sum_log_error / self.count)
        shape_k = mean_ / scale_theta
        self.require_recalc_gamma = False
        return tf.where(tf.math.is_nan(shape_k), 1e-10, shape_k), tf.where(tf.math.is_nan(scale_theta), 0., scale_theta)

    def reset_state(self):
        self.count.assign(tf.zeros_like(self.count))
        self.sum_error.assign(tf.zeros_like(self.sum_error))
        self.sum_error_log_error.assign(tf.zeros_like(self.sum_error_log_error))
        self.sum_log_error.assign(tf.zeros_like(self.sum_log_error))
        self.var.assign(tf.zeros_like(self.var))
        self.gamma_dist_built.assign(tf.zeros_like(self.gamma_dist_built,dtype=tf.uint8))

@register_keras_serializable(package="AD_Model")
class SamplePercentThresholdMetric(metrics.Metric):
    def __init__(self, error_shape, thres_percentile=.9, any_anomaly_thres=0.02,reduce_labels=False,name="",debug=False,**kwargs):
        dtype_ = tf.float32
        kwargs['name'] ="sample_threshold_metric "+name
        kwargs['dtype'] =dtype_
        super().__init__(**kwargs)
        self.comparison_tensor = self.add_weight(shape=error_shape,name='percentile_comparison',initializer='zeros',dtype=dtype_)
        self.error_shape = error_shape
        self.any_anomaly_thres = any_anomaly_thres
        self.thres_percentile = thres_percentile
        self.reduce_labels = reduce_labels
        self.debug = debug

    def config(self,config_dict):
        for key, value in config_dict.items():
            setattr(self,key,value)

    def get_description(self):
        return (f"SampleThres_{self.thres_percentile}_{self.any_anomaly_thres}").replace('.','p')

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "error_shape": self.error_shape,"thres_percentile":self.thres_percentile,
                "any_anomaly_thres":self.any_anomaly_thres,"reduce_labels":self.reduce_labels}

    def update_state(self, error_tensor : tf.Tensor,sample_weight=None):
        if sample_weight is not None:
            raise Exception("Sample weights not used!")
        pass

    def threshold_compare(self,input_tensor : tf.Tensor):
        if self.debug:
            if tf.reduce_sum(self.comparison_tensor) == 0: raise Exception("Threshold has not been set. Please run set_theshold(model,datatset,thres_percentile)")
        pred_labels = tf.cast(tf.reduce_mean(tf.cast(input_tensor >= self.comparison_tensor,tf.float32),axis=-1) >= self.any_anomaly_thres,tf.float32)

        if self.reduce_labels == "Mean":
            pred_labels = tf.reduce_mean(pred_labels,axis=-1)
        elif self.reduce_labels == "Max":
            pred_labels = tf.reduce_max(pred_labels,axis=-1)

        return tf.round(pred_labels)

    def set_threshold(self,model:TAADModelBaseClass,dataset):
        sample_tensor = None
        for batch in dataset:
            error_tensor = model(batch[:-1])[-2]
            error_tensor_flattened = tf.reshape(error_tensor,shape=(-1,self.error_shape[-1]))
            if sample_tensor is None:
                sample_tensor : tf.Tensor = error_tensor_flattened
            else:
                sample_tensor = tf.concat((sample_tensor,error_tensor_flattened),axis=0)
        self.comparison_tensor.assign(tfp.stats.percentile(sample_tensor,self.thres_percentile*100,axis=0))

    def result(self):
        return self.comparison_tensor

    def reset_state(self):
        self.comparison_tensor.assign(tf.zeros(self.error_shape))

@register_keras_serializable(package="AD_Model")
class NonPosNonNeg(Constraint):
    def __init__(self,column_focused=False,flipped=False,debug=False) -> None:
        super().__init__()
        self.column_focused = column_focused
        self.flipped = flipped
        self.debug = debug
        self.pos_idx = 0 if self.flipped else 1
        self.neg_idx = (self.pos_idx+1)%2 # opposite of pos_idx
    def __call__(self, w):
        pos_idx = self.pos_idx; neg_idx = self.neg_idx
        if self.column_focused:
            w_out = [0,]*w.shape[1]
            for _ in range((w.shape[1]+1)//2):
                if w.shape[1]>pos_idx:
                    # Enforce non-positive weights in column 0
                    w_pos = w[..., pos_idx] * tf.cast((tf.greater_equal(w[..., pos_idx], 0.0)))
                    w_out[pos_idx]=w_pos; pos_idx+=2
                if w.shape[1]>self.neg_idx:
                    # Enforce non-negative weights in column 1
                    w_neg = w[..., neg_idx] * tf.cast((tf.less_equal(w[..., neg_idx], 0.0)))
                    w_out[neg_idx]=w_neg; neg_idx+=2
            return tf.stack(w_out,axis=1)
        else:
            w_out = [0,]*w.shape[0]
            for _ in range((w.shape[0]+1)//2):
                if w.shape[0]>pos_idx:
                    # Enforce non-negative weights
                    w_pos = w[pos_idx] * tf.cast((tf.greater_equal(w[pos_idx], 0.0)), floatx())
                    w_out[pos_idx]=w_pos; pos_idx+=2
                if w.shape[0]>self.neg_idx:
                    # Enforce non-negative weights
                    w_neg = w[neg_idx] * tf.cast((tf.less_equal(w[neg_idx], 0.0)), floatx())
                    w_out[neg_idx]=w_neg; neg_idx+=2
            if self.debug:print("pre :",w.value().numpy());print("post:",tf.stack(w_out,axis=0).numpy())
            return tf.stack(w_out,axis=0)
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "flipped": self.flipped}

class EpochEndCallback(Callback):
    def on_test_begin(self, logs=None):
        try: self.model.after_training_step()
        except: pass
    def on_epoch_begin(self,epoch,logs=None):
        try: self.model.on_epoch_begin(epoch,logs)
        except: pass
    def on_train_end(self, logs=None):
        try: self.model.on_train_end()
        except: pass

def simple_acc_test(model, dataset, subset_names, config_thres=None):
    def evaluate_labels_pred(pred_labels,true_labels):
        return tf.cast(pred_labels,tf.bool)==true_labels
    def get_true_pos(pred_labels,true_labels):
        return tf.reduce_sum(tf.cast(tf.logical_and(evaluate_labels_pred(pred_labels,true_labels),true_labels),tf.int64))
    def get_true_neg(pred_labels,true_labels):
        return tf.reduce_sum(tf.cast(tf.logical_and(evaluate_labels_pred(pred_labels,true_labels),tf.logical_not(true_labels)),tf.int64))
    print(f"Testing {model.get_description()}")
    print("CYCLE::{METHOD: TRUE_POSITIVE|TRUE_NEGATIVE|ACCURACY}")
    output_metrics={}
    if config_thres is not None:
        model.config_thresholders(config_thres)
    for training_cycle in subset_names:
        count=0;anomaly_count=0;output_metric={}
        for key in (list(('ADN',))+list(model.thresholders.keys())):
            output_metric[key]={};output_metric[key]['TP']=0;output_metric[key]['TN']=0
        for batch in tqdm(dataset[training_cycle],desc=f"Testing {training_cycle} set"):
            in_tensor = batch[:-1]
            labels_true = tf.cast(batch[-1],tf.bool)
            out = model(in_tensor)
            count+=in_tensor[0].shape[0]
            labels_pred = tf.round(tf.squeeze(out[-1]))
            output_metric['ADN']['TP'] += get_true_pos(labels_pred,labels_true)
            output_metric['ADN']['TN'] += get_true_neg(labels_pred,labels_true)
            anomaly_count+=(tf.reduce_sum(tf.cast(labels_true,tf.dtypes.float32)))
            for key_ in model.thresholders.keys():
                labels_pred_ = tf.squeeze(model.thresholders[key_].threshold_compare(out[-2]))
                output_metric[key_]['TP'] += get_true_pos(labels_pred_,labels_true)
                output_metric[key_]['TN'] += get_true_neg(labels_pred_,labels_true)

        for key_ in output_metric.keys():
            output_metric[key_]['Acc'] = (output_metric[key_]['TP'] + output_metric[key_]['TN']).numpy() / count
            output_metric[key_]['TP'] = (tf.cast(output_metric[key_]['TP'],tf.float32) / anomaly_count).numpy() if anomaly_count > 0 else 0
            output_metric[key_]['TN'] = (tf.cast(output_metric[key_]['TN'],tf.float32) / (count - anomaly_count)).numpy() if (count - anomaly_count) > 0 else 0
        formatted_dict = {k: f"{v['TP']:.3f}|{v['TN']:.3f}|{v['Acc']:.3f}" for k, v in output_metric.items()}
        output_metric['Ano_ratio'] = anomaly_count / count; formatted_dict['Ano_ratio'] = f"{anomaly_count / count:.3f}"
        output_metric['Count'] = count; formatted_dict['Count'] = count
        output_metrics[training_cycle] = output_metric
        print((f"{training_cycle:10}::{formatted_dict}").replace("'",""))

    output_metrics['All']={}
    for key_ in output_metric:
        if key_ in ['Count','Ano_ratio']: continue
        output_metrics['All'][key_]={}
        for sub_metric in ['TP','TN','Acc']:
            output_metrics['All'][key_][sub_metric] = 0;total_count=0
            for cycle in subset_names:
                sub_count = output_metrics[cycle]['Count']
                if sub_metric == 'TP': sub_count*= output_metrics[cycle]['Ano_ratio']
                elif sub_metric == 'TN': sub_count*= (1-output_metrics[cycle]['Ano_ratio'])
                output_metrics['All'][key_][sub_metric] += output_metrics[cycle][key_][sub_metric]*sub_count
                total_count+=sub_count
            output_metrics['All'][key_][sub_metric]/=total_count
    formatted_dict = {k: f"{v['TP']:.3f}|{v['TN']:.3f}|{v['Acc']:.3f}" for k, v in output_metrics['All'].items()}
    formatted_dict['Ano_ratio']=0;formatted_dict['Count']=total_count
    for cycle in subset_names:
        formatted_dict['Ano_ratio'] += output_metrics[cycle]['Ano_ratio']*output_metrics[cycle]['Count']
    formatted_dict['Ano_ratio']  = f"{(formatted_dict['Ano_ratio'] / total_count).numpy():.3f}"
    print((f"All       ::{formatted_dict}").replace("'",""))
