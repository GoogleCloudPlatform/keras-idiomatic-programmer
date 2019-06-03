# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from keras import optimizers
import random

def model_fn(learning_rate, optimizer, dropout):
        ''' make an instance of the model '''
        # ADD CODE to construct the model here
        # set dropout rates based on the dropout parameter.
        # common convention is to ½ the dropout amount on each subsequent dropout 
        # layer.

        # select the optimizer and set the learning rate
        if optimizer == 'adam':
                opt = optimizers.Adam(lr=learning_rate)
        elif optimizer == 'adagrad':
                opt = optimizers.Adagrad(lr=learning_rate)
        elif optimizer == 'rmsprop':
                opt = optimizers.RMSprop(lr=learning_rate)
        elif optimizer == 'sgd':
                opt = optimizers.SGD(lr=learning_rate)

        # compile the model
        model.compile(loss='categorical_crossentropy', optimizer=opt, 
              metrics=['accuracy'])

        # return the model
        return model

def train_fn(model, nepochs, batch_size, x_train, y_train, x_val, y_val):
        ''' train the model for a fixed number of epochs '''
        
        # create a feeder and add some basic image augmentation
        datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, 
                                     rotation=30)
        
        # train the model for the short number of epochs
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size, 
                            shuffle=True),
                            steps_per_epoch=len(x_train) / batch_size, 
                            epochs=nepochs, verbose=1, 
                            validation_data=(x_val, y_val))

def hyper_search(nepochs, ncombos, x_train, y_train, x_val, y_val):
        ''' Perform a Hyperparameter Search'''
        # the hyperparameter ranges to search from
        learning_rates = [ 0.1, 0.01, 0.001, 0.0001, 0.00001 ]
        dropouts = [ 0.10, 0.25, 0.25 ]
        batch_sizes = [ 32, 128 ]
        optimizers = [ 'adam', 'adagrad', 'rmsprop' ]

        # Generate the specified (ncombos) random combinations
        for n in range(ncombos):
                learning_rate = random.choice(learning_rates)
                dropout = random.choice(dropouts)
                batch_size = random.choice(batch_sizes)
                optimizer = random.choice(optimizers)

                # Construct the model instance
                model = model_fn(learning_rate, optimizer, dropout)
                
                # Do the short run training session
                train_fn(model, nepochs, batch_size, x_train, y_train, x_val, 
                         y_val)
