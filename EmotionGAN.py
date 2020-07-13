'''
cGAN for generation of emotional faces
v.2.1: specific emotions
Author: Nicolas Kolbenschlag
'''

# cGAN: https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/

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

class EmotionGAN():
    def __init__(self, noiseShape, imageShape, generator=None, discriminator=None):
        self.n_classes = 2
        self.noiseShape = noiseShape
        self.imageShape = imageShape
        if not generator: self.generator = self.generateGenerator()
        else: self.generator = generator
        if not discriminator: self.discriminator = self.generateDiscriminator()
        else: self.discriminator = discriminator
        self.imageSaveDir = "generatedImages"
        self.datasetDir = "AffWild2_some_shuffled"
    
    def generateGenerator(self):
        in_label = keras.layers.Input(shape=(1,))
        li = keras.layers.Embedding(self.n_classes, 10) (in_label)
        li = keras.layers.Dense(10) (li)
        li = keras.layers.Reshape((1, 1, 10)) (li)

        in_lat = keras.layers.Input(shape=self.noiseShape)
        n_nodes = 100 * 1 * 1
        gen = keras.layers.Dense(n_nodes) (in_lat)
        gen = keras.layers.LeakyReLU(alpha=.2) (gen)
        gen = keras.layers.Reshape((1, 1, 100)) (gen)
        
        merge = keras.layers.Concatenate() ([gen, li])

        gen = keras.layers.Conv2DTranspose(filters=512, kernel_size=(4,4), strides=(1,1), padding="valid", data_format="channels_last", kernel_initializer="glorot_uniform") (merge)
        gen = keras.layers.BatchNormalization(momentum=.5) (gen)
        gen = keras.layers.LeakyReLU(.2) (gen)

        gen = keras.layers.Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer="glorot_uniform") (gen)
        gen = keras.layers.BatchNormalization(momentum=.5) (gen)
        gen = keras.layers.LeakyReLU(.2) (gen)

        gen = keras.layers.Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer="glorot_uniform") (gen)
        gen = keras.layers.BatchNormalization(momentum=.5) (gen)
        gen = keras.layers.LeakyReLU(.2) (gen)

        gen = keras.layers.Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer="glorot_uniform") (gen)
        gen = keras.layers.BatchNormalization(momentum=.5) (gen)
        gen = keras.layers.LeakyReLU(.2) (gen)

        gen = keras.layers.Conv2DTranspose(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", kernel_initializer="glorot_uniform") (gen)
        gen = keras.layers.BatchNormalization(momentum=.5) (gen)
        gen = keras.layers.LeakyReLU(.2) (gen)

        gen = keras.layers.Conv2DTranspose(filters=3, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer="glorot_uniform") (gen)
        out_layer = keras.layers.Activation("tanh") (gen)
        model = keras.Model([in_lat, in_label], out_layer)
        # N = 1
        # model = keras.Sequential([
        # keras.layers.Input(shape=self.noiseShape),
        # keras.layers.Conv2DTranspose(filters=512 * N, kernel_size=(4,4), strides=(1,1), padding="valid", data_format="channels_last", kernel_initializer="glorot_uniform"),
        # keras.layers.BatchNormalization(momentum=.5),
        # keras.layers.LeakyReLU(.2),

        # keras.layers.Conv2DTranspose(filters=256 * N, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer="glorot_uniform"),
        # keras.layers.BatchNormalization(momentum=.5),
        # keras.layers.LeakyReLU(.2),

        # keras.layers.Conv2DTranspose(filters=128 * N, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer="glorot_uniform"),
        # keras.layers.BatchNormalization(momentum=.5),
        # keras.layers.LeakyReLU(.2),

        # keras.layers.Conv2DTranspose(filters=64 * N, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer="glorot_uniform"),
        # keras.layers.BatchNormalization(momentum=.5),
        # keras.layers.LeakyReLU(.2),

        # keras.layers.Conv2DTranspose(filters=64 * N, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", kernel_initializer="glorot_uniform"),
        # keras.layers.BatchNormalization(momentum=.5),
        # keras.layers.LeakyReLU(.2),

        # keras.layers.Conv2DTranspose(filters=3, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer="glorot_uniform"),
        # keras.layers.Activation("tanh")
        # ])
        model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(lr=.00015, beta_1=.5), metrics=["accuracy"])
        return model
    
    def generateDiscriminator(self):
        in_label = keras.layers.Input(shape=(1,))
        li = keras.layers.Embedding(self.n_classes, 10) (in_label)
        li = keras.layers.Reshape((10,)) (li)
        li = keras.layers.Dense(10) (li)
        # li = keras.layers.Reshape((64, 64, 1)) (li)
        
        in_image = keras.layers.Input(shape=self.imageShape)
        
        # merge = keras.layers.Concatenate() ([in_image, li])

        fe = keras.layers.Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer="glorot_uniform") (in_image) # (merge)
        fe = keras.layers.LeakyReLU(.2) (fe)

        fe = keras.layers.Conv2D(filters=128, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer="glorot_uniform") (fe)
        fe = keras.layers.BatchNormalization(momentum=.5) (fe)
        fe = keras.layers.LeakyReLU(.2) (fe)

        fe = keras.layers.Conv2D(filters=256, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer="glorot_uniform") (fe)
        fe = keras.layers.BatchNormalization(momentum=.5) (fe)
        fe = keras.layers.LeakyReLU(.2) (fe)

        fe = keras.layers.Conv2D(filters=512, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer="glorot_uniform") (fe)
        fe = keras.layers.BatchNormalization(momentum=.5) (fe)
        fe = keras.layers.LeakyReLU(.2) (fe)

        fe = keras.layers.Flatten() (fe)

        merge = keras.layers.Concatenate() ([fe, li])
        merge = keras.layers.Dense(100) (merge)

        out_layer = keras.layers.Dense(1 + self.n_classes, activation="sigmoid") (merge) # NOTE out.shape = (fake, class1, class2, ...)
        model = keras.Model([in_image, in_label], out_layer)
        # model = keras.Sequential([
        # keras.layers.Input(shape=self.imageShape),
        # keras.layers.Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer="glorot_uniform"),
        # keras.layers.LeakyReLU(.2),

        # keras.layers.Conv2D(filters=128, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer="glorot_uniform"),
        # keras.layers.BatchNormalization(momentum=.5),
        # keras.layers.LeakyReLU(.2),

        # keras.layers.Conv2D(filters=256, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer="glorot_uniform"),
        # keras.layers.BatchNormalization(momentum=.5),
        # keras.layers.LeakyReLU(.2),

        # keras.layers.Conv2D(filters=512, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer="glorot_uniform"),
        # keras.layers.BatchNormalization(momentum=.5),
        # keras.layers.LeakyReLU(.2),

        # keras.layers.Flatten(),
        # keras.layers.Dense(1, activation="sigmoid")
        # ])
        model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(lr=.0002, beta_1=.5), metrics=["accuracy"])
        return model

    def setDiscriminatorTrainable(self, trainable):
        for i in range(len(self.discriminator.layers)):
            self.discriminator.layers[i].trainable = trainable
    
    def generateAdversial(self):
        self.setDiscriminatorTrainable(False)
        gen_noise, gen_label = self.generator.input
        gen_output = self.generator.output
        gan_output = self.discriminator ([gen_output, gen_label])
        model = keras.Model([gen_noise, gen_label], gan_output)
        model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(lr=.00015, beta_1=.5), metrics=["accuracy"])
        self.setDiscriminatorTrainable(True)
        return model
    
    def fit(self, epochs, batchSize):  
        averageDiscriminatorRealLoss = deque([0], maxlen=250)
        averageDiscriminatorFakeLoss = deque([0], maxlen=250)
        averageGanLoss = deque([0], maxlen=250)
        print("Images loaded.")
        for epoch in range(epochs):
            print("Epoch:", epoch)
            startTime = time.time()
            # NOTE Loop over dataset
            for iBatch in range(0, 740, batchSize):
                # NOTE load real images
                realImagesX, labels = self.getSamplesFromDataset(iBatch, iBatch + batchSize)
                if len(realImagesX) == 0: break
                # NOTE generate fake images with generator
                noise = self.generateNoise(len(realImagesX))
                fakeImagesX = self.generator.predict([noise, labels])
                # NOTE save generator samples
                if epoch % 10 == 0 and iBatch == 0:
                    stepNum = str(epoch).zfill(len(str(epochs)))
                    self.saveImageBatch(fakeImagesX, str(stepNum) + "_image.png")

                # NOTE prepare data for training
                # realDataY = np.ones(len(realImagesX)) - np.random.random_sample(len(realImagesX)) * .2
                # fakeDataY = np.random.random_sample(len(realImagesX)) * .2
                # one = np.ones(len(realImagesX)) - np.random.random_sample(len(realImagesX)) * .2
                # zero = np.random.random_sample(len(realImagesX)) * .2
                realDataY = np.random.random_sample((len(realImagesX), 1 + self.n_classes)) * .2
                for i in range(len(labels)):
                    label = labels[i]
                    realDataY[i][label + 1] = 1 - np.random.random_sample(1)[0] * .2# NOTE labels start from 0 to n_labels-1
                fakeDataY = np.zeros((len(realImagesX), 3))
                fakeDataY[:,0] = .99# TODO label smoothing

                # NOTE train discriminator seperately on real and fake
                discriminatorMetricsReal = self.discriminator.train_on_batch([realImagesX, labels], realDataY)
                discriminatorMetricsFake = self.discriminator.train_on_batch([fakeImagesX, labels], fakeDataY)# TODO what for labels
                print("Discriminator: real loss: %f fake loss: %f" % (discriminatorMetricsReal[0], discriminatorMetricsFake[0]))
                averageDiscriminatorRealLoss.append(discriminatorMetricsReal[0])
                averageDiscriminatorFakeLoss.append(discriminatorMetricsFake[0])
                # NOTE train adversial model
                ganX = self.generateNoise(len(realImagesX))
                sampledLabels = list(range(self.n_classes)) * (len(realImagesX) // self.n_classes)
                sampledLabels += list(range(self.n_classes))[: len(realImagesX) - len(sampledLabels)]
                ganY = np.random.random_sample((len(realImagesX), 1 + self.n_classes)) * .2
                for i in range(len(sampledLabels)):
                    label = sampledLabels[i]
                    ganY[i][label + 1] = 1 - np.random.random_sample(1)[0] * .2
                ganMetrics = self.generateAdversial().train_on_batch([ganX, np.array(sampledLabels)], ganY)
                print("GAN loss: %f" % (ganMetrics[0]))
                averageGanLoss.append(ganMetrics[0])
                gc.collect()
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
        # fileNames = os.listdir(self.datasetDir + "/images2classes")[countStart : countEnd]
        # images = [self.loadImage(file) for file in fileNames if len(file.split(".")) == 2 and file.split(".")[1] == "jpg"]
        fileNames = os.listdir(self.datasetDir + "/images2classes")

        fileNames = [file for file in fileNames if len(file.split(".")) == 2 and file.split(".")[1] == "jpg"][countStart : countEnd]
        images = [self.loadImage(file) for file in fileNames if len(file.split(".")) == 2 and file.split(".")[1] == "jpg"]
        
        # images = [self.loadImage(file) for file in fileNames if len(file.split(".")) == 2 and file.split(".")[1] == "jpg"][countStart : countEnd]
        with open(self.datasetDir + "/labels_images2classes.txt") as file: labels = file.readlines()[countStart : countEnd]
        labels = [0 if int(label.strip()) == 4 else 1 for label in labels]
        return np.array(images), np.array(labels)

def plotLosses(losses:dict):
    for key, value in losses.items():
        plt.figure()
        plt.plot(value, label=key)
    plt.ylabel("loss")
    plt.legend()
    plt.show()

NOISE_SHAPE = (1,1,100)
EPOCHS = 50
BATCH_SIZE = 64
IMAGE_SHAPE = (64,64,3)

if __name__ == "__main__":
    gan = EmotionGAN(NOISE_SHAPE, IMAGE_SHAPE)
    # gan = EmotionGAN(NOISE_SHAPE, IMAGE_SHAPE, keras.models.load_model("generator"), keras.models.load_model("discriminator"))
    losses = gan.fit(EPOCHS, BATCH_SIZE)
    gan.generator.save("generator")
    gan.discriminator.save("discriminator")
    print("Training finished.")
    # print(gan.discriminator.summary())