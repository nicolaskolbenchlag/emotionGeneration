"""
wacGAN (WassersteinConditionalGAN) for generation of emotional faces
v.2.3: specific emotions
Author: Nicolas Kolbenschlag
"""

# TODO: add https://github.com/bobchennan/Wasserstein-GAN-Keras/blob/master/mnist_wacgan.py

import tensorflow.keras as keras
import numpy as np
np.random.seed(1337)
from collections import deque
import time
import PIL
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gc

def normImage(img):
    img = (img / 127.5) - 1
    return img

def denormImage(img):
    img = (img + 1) * 127.5
    return img.astype(np.uint8)

def wassersteinLoss(y_true, y_pred):
    return keras.backend.mean(y_true * y_pred)

class EmotionGAN():
    def __init__(self, noiseShape, imageShape):
        self.nClasses = 3
        self.noiseShape = noiseShape
        self.imageShape = imageShape
        self.generator = self.generateGenerator()
        self.criticer = self.generateCriticer()
        self.adversial = self.generateAdversial()
        self.imageSaveDir = "generatedImages"
        self.datasetDir = "AffWild2_some_shuffled"
    
    def generateGenerator(self):
        cnn = keras.Sequential([
            keras.layers.Dense(1024, input_dim=self.noiseShape),
            keras.layers.LeakyReLU(),
            keras.layers.Dense(16 * 16 * 32),
            keras.layers.LeakyReLU(),
            keras.layers.Reshape((16, 16, 32)),
            # (32, 32, 256)
            keras.layers.UpSampling2D(size=2),
            keras.layers.Conv2D(256, 5, padding="same", kernel_initializer="glorot_uniform"),
            keras.layers.LeakyReLU(),
            # (64, 64, 128)
            keras.layers.UpSampling2D(size=2),
            keras.layers.Conv2D(128, 5, padding="same", kernel_initializer="glorot_uniform"),
            keras.layers.LeakyReLU(),
            # (64, 64, 3)
            keras.layers.Conv2D(3, 2, padding="same", activation="tanh", kernel_initializer="glorot_uniform")
        ])
        latent = keras.layers.Input(shape=(self.noiseShape,))
        image_class = keras.layers.Input(shape=(1,), dtype="int32")
        cls = keras.layers.Flatten()(keras.layers.Embedding(self.nClasses, self.noiseShape)(image_class))
        h = keras.layers.Multiply()([latent, cls])
        fake_image = cnn(h)
        model = keras.Model(inputs=[latent, image_class], outputs=fake_image)
        model.compile(optimizer=keras.optimizers.Adam(lr=.0002, beta_1=.5), loss="binary_crossentropy")
        return model
    
    def generateCriticer(self):
        cnn = keras.Sequential([
            keras.layers.Conv2D(32, 3, padding="same", input_shape=self.imageShape),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(.3),
            keras.layers.Conv2D(64, 3, padding="same"),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(.3),
            keras.layers.Conv2D(128, 3, padding="same"),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(.3),
            keras.layers.Conv2D(256, 3, padding="same"),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(.3),
            keras.layers.Flatten()
        ])
        image = keras.layers.Input(shape=self.imageShape)
        features = cnn(image)
        # fake = keras.layers.Dense(128, activation="sigmoid")(features)
        fake = keras.layers.Dense(1, activation="linear", name="generation")(features)
        # aux = keras.layers.Dense(128, activation="sigmoid")(features)
        aux = keras.layers.Dense(self.nClasses, activation="softmax", name="auxiliary")(features)
        model = keras.Model(inputs=image, outputs=[fake, aux])
        # model.compile(optimizer=keras.optimizers.SGD(clipvalue=.01), loss=[wassersteinLoss, "sparse_categorical_crossentropy"])
        model.compile(optimizer=keras.optimizers.Adam(lr=.0002, beta_1=.5), loss=[wassersteinLoss, "sparse_categorical_crossentropy"])
        return model
    
    def generateAdversial(self):
        latent = keras.layers.Input(shape=(self.noiseShape,))
        image_class = keras.layers.Input(shape=(1,), dtype="int32")
        fake = self.generator([latent, image_class])
        self.criticer.trainable = False
        fake, aux = self.criticer(fake)
        combined = keras.Model(inputs=[latent, image_class], outputs=[fake, aux])
        combined.compile(optimizer="RMSprop", loss=[wassersteinLoss, "sparse_categorical_crossentropy"])
        return combined
    
    def fit(self, epochs, batchSize):
        averageDiscriminatorLoss = deque([0], maxlen=250)
        averageGanLoss = deque([0], maxlen=250)
        print("Images loaded.")
        for epoch in range(epochs):
            print("Epoch:", epoch)
            startTime = time.time()
            # NOTE Loop over dataset
            for iBatch in range(0, 1110, batchSize):
                # NOTE load real images
                realImagesX, labels = self.getSamplesFromDataset(iBatch, iBatch + batchSize)
                if len(realImagesX) == 0: break
                # NOTE generate fake images with generator
                noise = self.generateNoise(len(realImagesX))
                fakeImagesX = self.generator.predict([noise, labels])
                # NOTE save generator samples
                if epoch % 2 == 0 and iBatch == 0 and epoch != 0:
                    stepNum = str(epoch).zfill(len(str(epochs)))
                    self.saveImageBatch(fakeImagesX, str(stepNum) + "_image.png")
                # NOTE prepare data for training
                sampledLabels = np.random.randint(0, self.nClasses, len(realImagesX))
                yRealness = np.concatenate((- np.ones(len(realImagesX)), np.ones(len(realImagesX))), axis=0)
                yLabel = np.concatenate((labels, sampledLabels), axis=0)
                x = np.concatenate((realImagesX, fakeImagesX))
                # NOTE train criticer
                discriminatorMetrics = self.criticer.train_on_batch(x, [yRealness, yLabel])
                print("Discriminator: loss: %f" % (discriminatorMetrics[0]))
                averageDiscriminatorLoss.append(discriminatorMetrics[0])
                # NOTE train adversial model
                ganX = self.generateNoise(len(realImagesX) * 2)
                sampledLabels = np.random.randint(0, self.nClasses, len(realImagesX) * 2)
                ganY = - np.ones(len(realImagesX) * 2)
                ganMetrics = self.adversial.train_on_batch([ganX, sampledLabels], [ganY, sampledLabels])
                print("GAN loss: %f" % (ganMetrics[0]))
                averageGanLoss.append(ganMetrics[0])
                gc.collect()
            # NOTE finish epoch and log results
            diffTime = int(time.time() - startTime)
            print("Epoch %d completed. Time took: %s secs." % (epoch, diffTime))
            if (epoch + 1) % 500 == 0:
                print("-----------------------------------------------------------------")
                print("Average Disc loss: %f" % (np.mean(averageDiscriminatorLoss)))
                print("Average GAN loss: %f" % (np.mean(averageGanLoss)))
                print("-----------------------------------------------------------------")
        return {"Discriminator": averageDiscriminatorLoss, "Adversial": averageGanLoss}

    def generateNoise(self, batchSize):
        return np.random.normal(0, 1, size=(batchSize,self.noiseShape))

    def saveImageBatch(self, imageBatch, fileName):
        plt.figure(figsize=(4,4))
        gs1 = gridspec.GridSpec(4, 4)
        gs1.update(wspace=0, hspace=0)
        rand_indices = np.random.choice(imageBatch.shape[0], 16, replace=False)
        for i in range(16):
            ax1 = plt.subplot(gs1[i])
            ax1.set_aspect("equal")
            rand_index = rand_indices[i]
            image = imageBatch[rand_index, :,:,:]
            fig = plt.imshow(denormImage(image))
            plt.axis("off")
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(self.imageSaveDir + "/" + fileName, bbox_inches="tight", pad_inches=0)
        plt.close()

    def loadImage(self, fileName):
        image = PIL.Image.open(self.datasetDir + "/images/" + fileName)
        image = image.resize(self.imageShape[:-1])
        image = image.convert("RGB")
        image = np.array(image)
        image = normImage(image)
        return image
    
    def getSamplesFromDataset(self, countStart, countEnd):
        images, labels = [], []
        fileNames = os.listdir(self.datasetDir + "/images3classes")
        fileNames = [file for file in fileNames if len(file.split(".")) == 2 and file.split(".")[1] == "jpg"][countStart : countEnd]
        images = [self.loadImage(file) for file in fileNames]# if len(file.split(".")) == 2 and file.split(".")[1] == "jpg"]        
        with open(self.datasetDir + "/labels_images3classes.txt") as file: labels = file.readlines()[countStart : countEnd]
        labels_ = []
        for label_ in labels:
            label = int(label_)
            if label == 1:
                labels_.append(0)
            elif label == 4:
                labels_.append(1)
            elif label == 5:
                labels_.append(2)
            else:
                assert 1==2, "impossible case: " + str(label) + str(type(label))
        labels = labels_
        return np.array(images), np.array(labels)

def plotLosses(losses:dict):
    for key, value in losses.items():
        plt.figure()
        plt.plot(value, label=key)
    plt.ylabel("loss")
    plt.legend()
    plt.show()

NOISE_SHAPE = 100
EPOCHS = 30
BATCH_SIZE = 64
IMAGE_SHAPE = (64,64,3)

if __name__ == "__main__":
    gan = EmotionGAN(NOISE_SHAPE, IMAGE_SHAPE)
    losses = gan.fit(EPOCHS, BATCH_SIZE)
    print("Training finished.")