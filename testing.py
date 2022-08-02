from utils.tools import read_yaml
# GENERAL LIBRARIES 
import math
import numpy as np
import joblib
from pathlib import Path
# MACHINE LEARNING LIBRARIES
import sklearn
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# OPTUNA
import optuna
from optuna.trial import TrialState
from utils.transformer import TransformerEncoder, PatchClassEmbedding, Patches
from utils.data import load_mpose, random_flip, random_noise, one_hot
from utils.tools import CustomSchedule, CosineSchedule
from utils.tools import Logger

config = read_yaml('utils/config.yaml')

print(config)
model_size = config['MODEL_SIZE']
n_heads = config[model_size]['N_HEADS']
n_layers = config[model_size]['N_LAYERS']
embed_dim = config[model_size]['EMBED_DIM']
dropout = config[model_size]['DROPOUT']
mlp_head_size = config[model_size]['MLP']
activation = tf.nn.gelu
d_model = 64 * n_heads
d_ff = d_model * 4

split = 1
fold = 0
trial = None
bin_path = config['MODEL_DIR']
def get_data():
    X_train, y_train, X_test, y_test = load_mpose(config['DATASET'], split, verbose=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                        test_size=config['VAL_SIZE'],
                                                        random_state=42,
                                                        stratify=y_train)
            
    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    ds_train = ds_train.map(lambda x,y : one_hot(x,y,config['CLASSES']), 
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.map(random_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.map(random_noise, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.shuffle(X_train.shape[0])
    ds_train = ds_train.batch(config['BATCH_SIZE'])
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    ds_val = ds_val.map(lambda x,y : one_hot(x,y,config['CLASSES']), 
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.cache()
    ds_val = ds_val.batch(config['BATCH_SIZE'])
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    ds_test = ds_test.map(lambda x,y : one_hot(x,y,config['CLASSES']), 
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.cache()
    ds_test = ds_test.batch(config['BATCH_SIZE'])
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
    return ds_train, ds_test,ds_val

def build_act(transformer):
        inputs = tf.keras.layers.Input(shape=(config['FRAMES'], 
                                              config[config['DATASET']]['KEYPOINTS']*config['CHANNELS']))
        x = tf.keras.layers.Dense(d_model)(inputs)
        x = PatchClassEmbedding(d_model, config['FRAMES'])(x)
        x = transformer(x)
        x = tf.keras.layers.Lambda(lambda x: x[:,0,:])(x)
        x = tf.keras.layers.Dense(mlp_head_size)(x)
        outputs = tf.keras.layers.Dense(config['CLASSES'])(x)
        return tf.keras.models.Model(inputs, outputs)


transformer = TransformerEncoder(d_model, n_heads,d_ff, dropout, activation, n_layers)
model = build_act(transformer)

print(model.summary())

#lr = CustomSchedule(d_model, 
#             warmup_steps=len(ds_train)*config['N_EPOCHS']*config['WARMUP_PERC'],
#             decay_step=len(ds_train)*config['N_EPOCHS']*config['STEP_PERC'])

optimizer = tfa.optimizers.AdamW(weight_decay=config['WEIGHT_DECAY'])

model.compile(optimizer=optimizer,
                           loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
                           metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")])


model.load_weights("bin/AcT_micro_3_9.h5")

ds_train, ds_test,ds_val = get_data()

_, accuracy_test = model.evaluate(ds_test)

X, y = tuple(zip(*ds_test))
y_pred = np.argmax(tf.nn.softmax(model.predict(tf.concat(X, axis=0)), axis=-1),axis=1)

print(y_pred)
print("*"*50)
print(y)