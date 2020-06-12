'''
GAN for generation of emotional faces
v.1.2: no specific emotions
Author: Nicolas Kolbenschlag
'''

import tensorflow.keras as keras
import numpy as np
np.random.seed(1337)
from collections import deque
import time
import PIL
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def normImage(img):
    img = (img / 127.5) - 1
    return img

def denormImage(img):
    img = (img + 1) * 127.5
    return img.astype(np.uint8)

class EmotionGANRandom2():
    def __init__(self, noiseShape, imageShape):
        self.generator = self.generateGenerator(noiseShape)
        self.discriminator = self.generateDiscriminator(imageShape)
        self.adversial = self.generateAdversial()
        self.noiseShape = noiseShape
        self.imageShape = imageShape
        self.imageSaveDir = "generatedImages"
        self.datasetDir = "AffWild2"
    
    def generateGenerator(self, noiseShape):
        model = keras.Sequential([
            keras.layers.Input(shape=noiseShape),
            keras.layers.Conv2DTranspose(filters=512, kernel_size=(4,4), strides=(1,1), padding="valid", data_format="channels_last", kernel_initializer="glorot_uniform"),
            keras.layers.BatchNormalization(momentum=.5),
            keras.layers.LeakyReLU(.2),

            keras.layers.Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer="glorot_uniform"),
            keras.layers.BatchNormalization(momentum=.5),
            keras.layers.LeakyReLU(.2),

            keras.layers.Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer="glorot_uniform"),
            keras.layers.BatchNormalization(momentum=.5),
            keras.layers.LeakyReLU(.2),

            keras.layers.Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer="glorot_uniform"),
            keras.layers.BatchNormalization(momentum=.5),
            keras.layers.LeakyReLU(.2),

            keras.layers.Conv2DTranspose(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", kernel_initializer="glorot_uniform"),
            keras.layers.BatchNormalization(momentum=.5),
            keras.layers.LeakyReLU(.2),

            keras.layers.Conv2DTranspose(filters=3, kernel_size=(3,3), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer="glorot_uniform"),
            keras.layers.Activation("tanh")
        ])
        model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(lr=0.00015, beta_1=0.5), metrics=["accuracy"])
        return model
    
    def generateDiscriminator(self, imageShape):
        model = keras.Sequential([
            keras.layers.Input(shape=imageShape),
            keras.layers.Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer="glorot_uniform"),
            keras.layers.LeakyReLU(.2),

            keras.layers.Conv2D(filters=128, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer="glorot_uniform"),
            keras.layers.BatchNormalization(momentum=.5),
            keras.layers.LeakyReLU(.2),

            keras.layers.Conv2D(filters=256, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer="glorot_uniform"),
            keras.layers.BatchNormalization(momentum=.5),
            keras.layers.LeakyReLU(.2),

            keras.layers.Conv2D(filters=512, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer="glorot_uniform"),
            keras.layers.BatchNormalization(momentum=.5),
            keras.layers.LeakyReLU(.2),

            keras.layers.Flatten(),
            keras.layers.Dense(1, activation="sigmoid")
        ])
        model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5), metrics=["accuracy"])
        return model
    
    def generateAdversial(self):
        model = keras.Sequential([self.generator, self.discriminator])
        model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(lr=0.00015, beta_1=0.5), metrics=["accuracy"])
        return model
    
    def fit(self, epochs, batchSize):  
        averageDiscriminatorRealLoss = deque([0], maxlen=250)
        averageDiscriminatorFakeLoss = deque([0], maxlen=250)   
        averageGanLoss = deque([0], maxlen=250)
        for epoch in range(epochs):
            print("Epoch:", epoch)
            startTime = time.time()
            # NOTE Loop over dataset
            for iBatch in range(0, 100, batchSize):
                # NOTE load real images
                realImagesX = self.getSamplesFromDataset(iBatch, iBatch + batchSize)[0]
                # NOTE generate fake images with generator
                noise = self.generateNoise(len(realImagesX))
                fakeImagesX = self.generator.predict(noise)
                # NOTE save generator samples
                if epoch % 10 == 0 and iBatch == 0:
                    stepNum = str(epoch).zfill(len(str(epochs)))
                    self.saveImageBatch(fakeImagesX, str(stepNum) + "_image.png")
                # NOTE prepare data for training
                #dataX = np.concatenate((realImagesX, fakeImagesX))
                realDataY = np.ones(len(realImagesX)) - np.random.random_sample(len(realImagesX)) * .2
                fakeDataY = np.random.random_sample(len(realImagesX)) * .2
                # dataY = np.concatenate((realDataY, fakeDataY))
                # NOTE train discriminator seperately on real and fake
                self.discriminator.trainable = True
                #self.generator.trainable = False
                discriminatorMetricsReal = self.discriminator.train_on_batch(realImagesX, realDataY)
                discriminatorMetricsFake = self.discriminator.train_on_batch(fakeImagesX, fakeDataY)
                print("Discriminator: real loss: %f fake loss: %f" % (discriminatorMetricsReal[0], discriminatorMetricsFake[0]))
                averageDiscriminatorRealLoss.append(discriminatorMetricsReal[0])
                averageDiscriminatorFakeLoss.append(discriminatorMetricsFake[0])
                # NOTE train adversial model
                ganX = self.generateNoise(len(realImagesX))
                ganY = realDataY
                self.generator.trainable = True
                self.discriminator.trainable = False
                ganMetrics = self.adversial.train_on_batch(ganX, ganY)
                print("GAN loss: %f" % (ganMetrics[0]))
                averageGanLoss.append(ganMetrics[0])
            # NOTE finish epoch and log results
            diffTime = int(time.time() - startTime)
            print("Epoch %d completed. Time took: %s secs." % (epoch, diffTime))
            if (epoch + 1) % 500 == 0:
                print("-----------------------------------------------------------------")
                print("Average Disc_fake loss: %f" % (np.mean(averageDiscriminatorFakeLoss)))
                print("Average Disc_real loss: %f" % (np.mean(averageDiscriminatorRealLoss)))
                print("Average GAN loss: %f" % (np.mean(averageGanLoss)))
                print("-----------------------------------------------------------------")
        return {"Discriminator real": averageDiscriminatorRealLoss, "Discriminator fake": averageDiscriminatorFakeLoss, "Adversial": averageGanLoss}

    def generateNoise(self, batchSize):
        return np.random.normal(0, 1, size=(batchSize,) + self.noiseShape)

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
        plt.clf()
    
    def loadImage(self, imagesDir, fileName):
        image = PIL.Image.open(self.datasetDir + "/cropped_aligned/cropped_aligned/" + imagesDir + "/" + fileName)
        image = image.resize(self.imageShape[:-1])
        image = image.convert("RGB")
        image = np.array(image)
        image = normImage(image)
        return image
    
    def getSamplesFromDataset(self, countStart, countEnd):
        images, lines = [], []
        countSkippedLabels = 0
        for labelsFileName in os.listdir(self.datasetDir + "/annotations/EXPR_Set/Training_Set"):
            with open(self.datasetDir + "/annotations/EXPR_Set/Training_Set/" + labelsFileName) as labelsFile:
                newLines = labelsFile.readlines()[1:]
                if countStart > countSkippedLabels + len(newLines):
                    countSkippedLabels += len(newLines)
                    continue
                linesToAdd = newLines[countStart - countSkippedLabels : countStart - countSkippedLabels + countEnd - countStart - len(lines)]
                lines += linesToAdd
                imagesDir = labelsFileName.split(".")[0]
                imageFilesNames = os.listdir(self.datasetDir + "/cropped_aligned/cropped_aligned/" + imagesDir)[countStart - countSkippedLabels : countStart - countSkippedLabels + countEnd - countStart - len(images)]
                images += [self.loadImage(imagesDir, fileName) for fileName in imageFilesNames]
                if len(lines) == countEnd - countStart: break
                countSkippedLabels += len(linesToAdd)
        return np.array(images), np.array(lines)

def plotLosses(losses:dict):
    for key, value in losses.items():
        plt.plot(value, label=key)
    plt.ylabel("loss")
    plt.legend()
    plt.show()

NOISE_SHAPE = (1,1,100)
EPOCHS = 500
BATCH_SIZE = 64
# IMAGE_SHAPE = (112,112,3)
IMAGE_SHAPE = (64,64,3)

if __name__ == "__main__":
    gan = EmotionGANRandom2(NOISE_SHAPE, IMAGE_SHAPE)
    losses = gan.fit(EPOCHS, BATCH_SIZE)
    plotLosses(losses)