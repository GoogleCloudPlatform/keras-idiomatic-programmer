# Copyright 2020 Google LLC
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

# DCGAN + composable (2016)
# Paper: https://arxiv.org/pdf/1511.06434.pdf

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Flatten, Reshape, Dropout, Dense, ReLU
from tensorflow.keras.layers import LeakyReLU, Activation, ZeroPadding2D
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import numpy as np

import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from models_c import Composable

class DCGAN(Composable):

    # Initial Hyperparameters
    hyperparameters = { 'initializer': 'glorot_uniform',
                        'regularizer': None,
                        'relu_clip'  : None,
                        'bn_epsilon' : None,
                        'use_bias'   : False
                      }
    
    def __init__(self, latent=100, input_shape=(28, 28, 1), 
                 **hyperparameters): 
        """ Construct a Deep Convolutional GAN (DC-GAN)
            latent      : dimension of latent space
            input_shape : input shape
            initializer : kernel initializer
            regularizer : kernel regularizer
            relu_clip   : max value for ReLU
            bn_epsilon  : epsilon for batch normalization
            use_bias    : whether to include bias
        """
        Composable.__init__(self, input_shape, None, self.hyperparameters, **hyperparameters)
        
        # Construct the generator
        self.g = self.generator(latent=latent, height=input_shape[0], channels=input_shape[2])

        # Construct the discriminator
        self.d = self.discriminator(input_shape=input_shape, optimizer=Adam(0.0002, 0.5))
        
        # Construct the combined (stacked) generator/discriminator model (GAN)
        self.model = self.gan(latent=latent, optimizer=Adam(0.0002, 0.5))
         
    def generator(self, latent=100, height=28, channels=1):
        """ Construct the Generator
            latent   : dimension of latent space
            channels : number of channels
        """
        def stem(inputs):
            factor = height // 4
            x = self.Dense(inputs, 128 * factor * factor)
            x = self.ReLU(x)
            x = Reshape((factor, factor, 128))(x)
            return x
        
        def learner(x):
            x = self.Conv2DTranspose(x, 128, (3, 3), strides=2, padding='same')
            x = self.Conv2D(x, 128, (3, 3), padding="same")
            x = self.BatchNormalization(x, momentum=0.8)
            x = self.ReLU(x)
        
            x = self.Conv2DTranspose(x, 64, (3, 3), strides=2, padding='same')
            x = self.Conv2D(x, 64, (3, 3), padding="same")
            x = self.BatchNormalization(x, momentum=0.8)
            x = self.ReLU(x)
            return x
        
        def classifier(x):
            outputs = self.Conv2D(x, channels, (3, 3), activation='tanh', padding="same")
            return outputs
        
        # Construct the Generator
        inputs = Input(shape=(latent,))
        x = stem(inputs)
        x = learner(x)
        outputs = classifier(x)

        return Model(inputs, outputs)

    
    def discriminator(self, input_shape=(28, 28, 1), optimizer=Adam(0.0002, 0.5)):
        """ Construct the discriminator
            input_shape : the input shape of the images
            optimizer   : the optimizer
        """
        
        def stem(inputs):
            x = self.Conv2D(inputs, 32, (3, 3), strides=2, padding="same")
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(0.25)(x)
            return x
        
        def learner(x):
            x = self.Conv2D(x, 64, (3, 3), strides=2, padding="same")
            x = ZeroPadding2D(padding=((0,1),(0,1)))(x)
            x = self.BatchNormalization(x, momentum=0.8)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(0.25)(x)
        
            x = self.Conv2D(x, 128, (3, 3), strides=2, padding="same")
            x = self.BatchNormalization(x, momentum=0.8)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(0.25)(x)
        
            x = self.Conv2D(x, 256, (3, 3), strides=1, padding="same")
            x = self.BatchNormalization(x, momentum=0.8)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(0.25)(x)
            return x
        
        def classifier(x):
            x = Flatten()(x)
            outputs = self.Dense(x, 1, activation='sigmoid')
            return outputs
        
        # Construct the discriminator
        inputs = Input(shape=input_shape)
        x = stem(inputs)
        x = learner(x)
        outputs = classifier(x)
        model = Model(inputs, outputs) 
        
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        # For the combined model we will only train the generator
        model.trainable = False
        return model
    
    def gan(self, latent=100, optimizer=Adam(0.0002, 0.5)):
        """ Construct the Combined Generator/Discrimator (GAN)
            latent       : the latent space dimension
            optimizer    : the optimizer
        """
        # The generator takes noise as input and generates fake images
        noise = Input(shape=(latent,))
        fake  = self.g(noise)

        # The discriminator takes generated images as input and determines if real or fake
        valid = self.d(fake)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        model = Model(noise, valid)
        model.compile(loss='binary_crossentropy', optimizer=optimizer)
        return model
    
    def train(self, images, latent=100, epochs=4000, batch_size=128, save_interval=50):
        """ Train the GAN
            images : images from the training data
            latent : dimension of the latent space
            
            credit: https://github.com/eriklindernoren
        """
        # Adversarial ground truths
        valid_labels = np.ones ((batch_size, 1))
        fake_labels  = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            #  Train the Discriminator

            # Select a random half of the images
            idx   = np.random.randint(0, images.shape[0], batch_size)
            batch = images[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, latent))
            fakes = self.g.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.d.train_on_batch(batch, valid_labels)
            d_loss_fake = self.d.train_on_batch(fakes, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            #  Train the Generator

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.model.train_on_batch(noise, valid_labels)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))


            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
                
    def save_imgs(self, epoch, latent=100):
        import os
        if not os.path.isdir('images'):
            os.mkdir('images')
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, latent))
        gen_imgs = self.g.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                #MNIST: axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].imshow(gen_imgs[cnt, :,:,0])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()

     
    
# Example
# model = DCGAN()

def example():
    # Build/Train a DCGAN for CIFAR-10

    gan = DCGAN(input_shape=(32, 32, 3))
    gan.model.summary()

    from tensorflow.keras.datasets import cifar10
    (x_train, _), (_, _) = cifar10.load_data()
    x_train, _ = gan.normalization(x_train, centered=True)
    gan.train(x_train, latent=100, epochs=6000)

# example()
