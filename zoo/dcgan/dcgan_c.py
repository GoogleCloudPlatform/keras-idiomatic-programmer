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

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, Reshape
from tensorflow.keras.layers import ReLU, LeakyReLU, Activation
from tensorflow.keras.regularizers import l2

import sys
sys.path.append('../')
from models_c import Composable

class DCGAN(Composable):
    
    def __init__(self, latent=100, input_shape=(28, 28, 1), reg=l2(0.001), init_weights='he_normal', relu=None): 
        """ Construct a Deep Convolutional GAN (DC-GAN)
            latent      : dimension of latent space
            input_shape : input shape
            reg         : kernel regularizer
            init_weights: kernel initializer
            relu        : max value for ReLU
        """
        super().__init__(reg=reg, init_weights=init_weights, relu=relu)
        
        # Construct the generator
        self.g = self.generator(latent=latent, channels=input_shape[2])
        
        # Construct the discriminator
        self.d = self.discriminator(input_shape=input_shape, optimizer=Adam(0.0002, 0.5))
        
        # Construct the combined (stacked) generator/discriminator model (GAN)
        self.model = self.gan(latent=latent, optimizer=Adam(0.0002, 0.5))
         
    def generator(self, latent=100, channels=1):
        """ Construct the Generator
            latent   : dimension of latent space
            channels : number of channels
        """
        def stem(inputs):
            x = Dense(128 * 7 * 7, activation="relu")(inputs)
            x = Reshape((7, 7, 128))(x)
            return x
        
        def learner(x):
            x = Conv2DTranspose(128, (3, 3), strides=2, padding='same')(x)
            x = Conv2D(128, (3, 3), padding="same")(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = ReLU()(x)
        
            x = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x) 
            x = Conv2D(64, (3, 3), padding="same")(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = ReLU()(x)
            return x
        
        def classifier(x):
            outputs = Conv2D(channels, (3, 3), activation='tanh', padding="same")(x)
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
            x = Conv2D(32, (3, 3), strides=2, padding="same")(inputs)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(0.25)(x)
            return x
        
        def learner(x):
            x = Conv2D(64, (3, 3), strides=2, padding="same")(x)
            x = ZeroPadding2D(padding=((0,1),(0,1)))(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(0.25)(x)
        
            x = Conv2D(128, (3, 3), strides=2, padding="same")(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(0.25)(x)
        
            x = Conv2D(256, (3, 3), strides=1, padding="same")(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(0.25)(x)
            return x
        
        def classifier(x):
            x = Flatten()(x)
            outputs = Dense(1, activation='sigmoid')(x)
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
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, latent))
        gen_imgs = self.g.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()

     
    
model = DCGAN()
(X_train, _), (_, _) = mnist.load_data()

# Rescale -1 to 1
X_train = X_train / 127.5 - 1.
X_train = np.expand_dims(X_train, axis=3)

model.train(X_train, latent=100, epochs=4000)
