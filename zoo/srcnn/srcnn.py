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

# Super Resolution CNN (SRCNN)
# Paper: https://arxiv.org/pdf/1501.00092.pdf

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Conv2DTranspose, Activation
from tensorflow.keras.optimizers import Adam

def stem(inputs):
    # dimensionality expansion with large coarse filter
    x = Conv2D(64, (9, 9), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def encoder(x):
    # 1x1 bottleneck convolution
    x = Conv2D(32, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Reconstruction
    x = Conv2D(3, (5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    outputs = Activation('sigmoid')(x)
    return outputs
    
inputs = Input((32, 32, 3))
x = stem(inputs)
outputs = encoder(x)

model = Model(inputs, outputs)
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=['accuracy'])

from tensorflow.keras.datasets import cifar10
import numpy as np
import cv2
def train():
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()

	x_train_lr = []
	for image in x_train:
    		image = cv2.resize(image, (16, 16), interpolation=cv2.INTER_CUBIC)
    		x_train_lr.append(cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC))
	x_train_lr = np.asarray(x_train_lr)

	x_test_lr = []
	for image in x_test:
    		image = cv2.resize(image, (16, 16), interpolation=cv2.INTER_CUBIC)
	x_test_lr.append(cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC))
	x_test_lr = np.asarray(x_test_lr)

	x_train = (x_train / 255.0).astype(np.float32)
	x_train_lr = (x_train_lr / 255.0).astype(np.float32)

	x_test = (x_test / 255.0).astype(np.float32)
	x_test_lr = (x_test_lr / 255.0).astype(np.float32)
	model.fit(x_train_lr, x_train, epochs=25, batch_size=32, verbose=1, validation_split=0.1)
