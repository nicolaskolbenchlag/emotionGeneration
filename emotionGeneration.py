'''
GAN for generation of emotional faces
v.1.: no specific emotions
Author: Nicolas Kolbenschlag
'''
import tensorflow.keras as keras
import numpy as np
import data

# GAN in keras: https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0
# Source: https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/GAN/dcgan_mnist.py
# Inspiration: https://github.com/llSourcell/Pokemon_GAN

class EmotionGAN():
    def __init__(self):
        self.discriminator = None
        self.generator = None
        self.adversialModel = None
        self.discriminatorModel = None
    
    def getDiscriminator(self):
        if self.discriminator:
            return self.discriminator
        DEPTH = 64
        DROPOUT = .4
        model = keras.Sequential([
            keras.layers.Conv2D(DEPTH * 1, 5, strides=2, padding="same"),
            keras.layers.LeakyReLU(alpha=.2),
            keras.layers.Dropout(DROPOUT),

            keras.layers.Conv2D(DEPTH * 2, 5, strides=2, padding="same"),
            keras.layers.LeakyReLU(alpha=.2),
            keras.layers.Dropout(DROPOUT),

            keras.layers.Conv2D(DEPTH * 4, 5, strides=2, padding="same"),
            keras.layers.LeakyReLU(alpha=.2),
            keras.layers.Dropout(DROPOUT),

            keras.layers.Conv2D(DEPTH * 8, 5, strides=2, padding="same"),
            keras.layers.LeakyReLU(alpha=.2),
            keras.layers.Dropout(DROPOUT),

            keras.layers.Flatten(),
            keras.layers.Dense(1, activation="sigmoid")
        ])
        self.discriminator = model
        return self.discriminator
    
    def getGenerator(self):
        if self.generator:
            return self.generator
        DEPTH = 256
        DROPOUT = .4
        DIM = 7
        model = keras.Sequential([
            keras.layers.Dense(DIM * DIM * DEPTH, input_dim=100),
            keras.layers.BatchNormalization(momentum=.9),
            keras.layers.Activation("relu"),
            keras.layers.Reshape((DIM, DIM, DEPTH)),
            keras.layers.Dropout(DROPOUT),
            
            keras.layers.UpSampling2D(),
            keras.layers.Conv2DTranspose(DEPTH // 2, 5, padding="same"),
            keras.layers.BatchNormalization(momentum=.9),
            keras.layers.Activation("relu"),

            keras.layers.UpSampling2D(),
            keras.layers.Conv2DTranspose(DEPTH // 4, 5, padding="same"),
            keras.layers.BatchNormalization(momentum=.9),
            keras.layers.Activation("relu"),

            keras.layers.UpSampling2D(),
            keras.layers.Conv2DTranspose(DEPTH // 8, 5, padding="same"),
            keras.layers.BatchNormalization(momentum=.9),
            keras.layers.Activation("relu"),

            keras.layers.UpSampling2D(),
            keras.layers.Conv2DTranspose(DEPTH // 8, 5, padding="same"),
            keras.layers.BatchNormalization(momentum=.9),
            keras.layers.Activation("relu"),

            keras.layers.Conv2DTranspose(3, 5, padding="same"),
            keras.layers.Activation("sigmoid")
        ])
        self.generator = model
        return self.generator
    
    def getDiscriminatorModel(self):
        if self.discriminatorModel:
            return self.discriminatorModel
        self.discriminatorModel = keras.Sequential([self.getDiscriminator()])
        self.discriminatorModel.compile(loss="binary_crossentropy", optimizer=keras.optimizers.RMSprop(lr=0.0001, decay=6e-8), metrics=["accuracy"])
        return self.discriminatorModel
    
    def getAdversialModel(self):
        if self.adversialModel:
            return self.adversialModel
        self.adversialModel = keras.Sequential([
            self.getGenerator(),
            self.getDiscriminator()
        ])
        self.adversialModel.compile(loss="binary_crossentropy", optimizer=keras.optimizers.RMSprop(lr=0.0001, decay=3e-8), metrics=["accuracy"])
        return self.adversialModel

    def train(self, iamges, epochs, batch_size=256):
        for epoch in range(epochs):
            print("Epoch:", str(epoch + 1))
            for i in range(0, len(images), batch_size):
                imagesReal = images[i : i + batch_size]
                noise = np.random.uniform(-1., 1., size=[len(imagesReal), 100])
                imagesFake = self.getGenerator().predict(noise)
                discriminatorX = np.concatenate((imagesReal, imagesFake))
                discriminatorY = np.ones([2 * len(imagesReal), 1])# real = 1
                discriminatorY[batch_size:,:] = 0# fake = 0
                discriminatorLoss = self.getDiscriminatorModel().fit(discriminatorX, discriminatorY, batch_size=batch_size, epochs=1, verbose=0)
                discriminatorLoss = discriminatorLoss.history["loss"][-1]
                adversialY = np.ones([len(imagesReal), 1])
                noise = np.random.uniform(-1., 1., size=[len(imagesReal), 100])
                adversialLoss = self.getAdversialModel().fit(noise, adversialY, verbose=0)
                adversialLoss = adversialLoss.history["loss"][-1]
                print("Discriminator:\t{}Adversial:\t{}".format(discriminatorLoss, adversialLoss))

    def genImages(self, samples, save=False):
        pass

if __name__ == "__main__":
    images = data.loadDataset()[0]
    print("Images:", images.shape)

    model = EmotionGAN()
    model.train(images, epochs=10000)
    # model.genImages(100)
    # model.getGenerator().summary()