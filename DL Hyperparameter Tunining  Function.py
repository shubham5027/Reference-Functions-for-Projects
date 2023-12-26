pip install -q -U keras-tuner

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import keras_tuner
from kerastuner.tuners import RandomSearch
from tensorflow.keras import layers

def build_model(hp):

    model = keras.Sequential()
    # Add variable number of layers
    for i in range(hp.Int('num_layers', min_value=1, max_value=5)):
        model.add(layers.Dense(units=hp.Int(f'layer_{i}_units', min_value=32, max_value=512, step=32),
                               activation=hp.Choice(f'layer_{i}_activation', values=['relu', 'tanh', 'sigmoid',]),
                               kernel_initializer=hp.Choice(f'layer_{i}_kernel_initializer', values=['glorot_uniform', 'he_normal', 'lecun_normal']),
                               kernel_regularizer=hp.Choice(f'layer_{i}_kernel_regularizer', values=['l1', 'l2'])))

    model.add(layers.Dropout(hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=keras.optimizers.get(hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd'])),
                  loss=hp.Choice('loss', values=['binary_crossentropy']),
                  metrics=['accuracy'])

    return model

tuner = RandomSearch(build_model,
                     objective='val_loss',
                     max_trials=5,
                     directory='my_tuner_directory',
                     project_name='my_tuner_project')

tuner.search(X_train_processed, y_train_encoded, epochs=50, validation_data=(X_test_processed, y_test_encoded))

tuner.get_best_hyperparameters()[0].values

model=tuner.get_best_models(num_models=1)[0]
callback=EarlyStopping(verbose=1,patience=1)
history=model.fit(X_train_processed,y_train_encoded,epochs=100,validation_data=(X_test_processed,y_test_encoded),callbacks=callback)

# Display model summary
model.summary()

# Get the list of all metric names
metrics_names = model.metrics_names

# Print all metrics
for metric_name in metrics_names:
    print(f"{metric_name}: {model.evaluate(X_test_processed, y_test_encoded, verbose=0, return_dict=True)[metric_name]}")

import matplotlib.pyplot as plt
