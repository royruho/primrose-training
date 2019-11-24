# first neural network with keras tutorial
import numpy as np
from numpy import loadtxt
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import MinMaxScaler as Scaler



seed=7
np.random.seed(seed)

# load the dataset
dataset=loadtxt('pima-indians-diabetes.csv', delimiter=',')
scaler = Scaler()
scaler.fit(dataset)
transformed_dataset = scaler.transform(dataset)
# split into input (X) and output (y) variables
X=transformed_dataset[:, 0:8]
y=transformed_dataset[:, 8]

# define the keras model
model1=Sequential()
model1.add(Dense(16, input_dim=8, activation='relu'))
model1.add(Dense(8, activation='relu'))
model1.add(Dense(1, activation='sigmoid'))
# compile the keras model
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model1.fit(X, y, epochs=1, batch_size=10)
# evaluate the keras model
_, accuracy=model1.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy * 100))


#### define model function ####
def create_model(learning_rate = 0.005, init_mode='glorot_uniform', optimizer='rmsprop', activation='sigmoid', units = 8, rate = 0.5):
    model=Sequential()
    model.add(Dense(16, input_dim=8, kernel_initializer=init_mode, activation='relu'))
    model.add(Dropout(rate = rate, noise_shape=None, seed=None))
    model.add(Dense(units = units, kernel_initializer=init_mode, activation='relu'))
    model.add(Dense(1, kernel_initializer=init_mode, activation=activation))
    # compile the keras model
    optimizer=RMSprop(learning_rate= learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# checked epochs[10,20,50,100 and batch_size[10,20,30,50] best results 100 , 10
# checked optimizer[rmsprop,adam,adagard] best results rmsprop
# checked learning_rates [0.0001, 0.001, 0.005, 0.01] best results [0.005]
# checked kernel inits ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
# best results 'glorot_uniform'
# chaecked activation of first 2 layers['relu', 'sigmoid', 'tanh'] - best results - 'relu'
# checked acitvation of last layer['relu', 'sigmoid', 'tanh'] - best 'Sigmoid
# hidden layer sizes [4,8,16,24,32] - best 24
# drop out layer rate [0, 0.01, 0.1, 0.25, 0.5, 0.75] - best - 0.1

model=KerasClassifier(build_fn=create_model)
optimizers=['rmsprop']
learning_rates=[0.0001, 0.001, 0.005, 0.01]
inits=['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
activators=['sigmoid']
layer_neuron_number=[8,16,24]
dropout_rate = [0.1]
epochs = [150, 250]
param_grid1=dict(learning_rate= learning_rates, epochs=epochs, batch_size=[10], optimizer=optimizers, activation = activators, units = layer_neuron_number, rate = dropout_rate)
grid1=GridSearchCV(estimator=model, param_grid=param_grid1, n_jobs=-1, cv=3)
grid1_result=grid1.fit(X, y)
