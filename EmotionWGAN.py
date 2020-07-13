'''
cWGAN for generation of emotional faces
v.2.2: specific emotions
Author: Nicolas Kolbenschlag
'''

# cGAN: https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/
# WGAN: https://machinelearningmastery.com/how-to-code-a-wasserstein-generative-adversarial-network-wgan-from-scratch/

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

class ClipConstraint(keras.constraints.Constraint):
	def __init__(self, clip_value):
		self.clip_value = clip_value
 
	def __call__(self, weights):
		return keras.backend.clip(weights, -self.clip_value, self.clip_value)
 
	def get_config(self):
		return {"clip_value": self.clip_value}

class EmotionGAN():
    def __init__(self, noiseShape, imageShape, generator=None, criticer=None):
        self.n_classes = 3
        self.noiseShape = noiseShape
        self.imageShape = imageShape
        if not generator: self.generator = self.generateGenerator()
        else: self.generator = generator
        if not criticer: self.criticer = self.generateCriticer()
        else: self.criticer = criticer
        self.imageSaveDir = "generatedImages"
        self.datasetDir = "AffWild2_some_shuffled"
    
    def generateGenerator(self):
        init = keras.initializers.RandomNormal(stddev=.02)

        in_label = keras.layers.Input(shape=(1,))
        li = keras.layers.Embedding(self.n_classes, 30) (in_label)
        li = keras.layers.Dense(1 * 7) (li)
        li = keras.layers.Reshape((1, 1, 7)) (li)

        # li = keras.layers.Dropout(.3) (li)

        in_lat = keras.layers.Input(shape=self.noiseShape)
        gen = keras.layers.Dense(1 * 1 * 128) (in_lat)
        gen = keras.layers.LeakyReLU(alpha=.2) (gen)
        gen = keras.layers.Reshape((1, 1, 128)) (gen)

        # gen = keras.layers.Dropout(.3) (gen)
        
        merge = keras.layers.Concatenate() ([gen, li])

        gen = keras.layers.Conv2DTranspose(filters=512, kernel_size=(4,4), strides=(1,1), padding="valid", data_format="channels_last", kernel_initializer=init) (merge)
        gen = keras.layers.BatchNormalization(momentum=.5) (gen)
        gen = keras.layers.LeakyReLU(.2) (gen)

        # gen = keras.layers.Dropout(.25) (gen)

        gen = keras.layers.Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer=init) (gen)
        gen = keras.layers.BatchNormalization(momentum=.5) (gen)
        gen = keras.layers.LeakyReLU(.2) (gen)
        
        # gen = keras.layers.Dropout(.25) (gen)

        gen = keras.layers.Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer=init) (gen)
        gen = keras.layers.BatchNormalization(momentum=.5) (gen)
        gen = keras.layers.LeakyReLU(.2) (gen)

        # gen = keras.layers.Dropout(.25) (gen)

        gen = keras.layers.Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer=init) (gen)
        gen = keras.layers.BatchNormalization(momentum=.5) (gen)
        gen = keras.layers.LeakyReLU(.2) (gen)

        gen = keras.layers.Conv2DTranspose(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", kernel_initializer=init) (gen)
        gen = keras.layers.BatchNormalization(momentum=.5) (gen)
        gen = keras.layers.LeakyReLU(.2) (gen)


        gen = keras.layers.Conv2DTranspose(filters=3, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer=init) (gen)
        out_layer = keras.layers.Activation("tanh") (gen)
        model = keras.Model([in_lat, in_label], out_layer)
        return model
    
    def generateCriticer(self):
        const = ClipConstraint(.01)
        init = keras.initializers.RandomNormal(stddev=.02)
        
        in_label = keras.layers.Input(shape=(1,))
        # li = keras.layers.Embedding(self.n_classes, 30) (in_label)
        # li = keras.layers.Dense(64 * 64) (li)
        # li = keras.layers.Reshape((64, 64, 1)) (li)
        
        in_image = keras.layers.Input(shape=self.imageShape)
        
        # merge = keras.layers.Concatenate() ([in_image, li])

        fe = keras.layers.Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer=init, kernel_constraint=const) (in_image)#(merge)
        fe = keras.layers.LeakyReLU(.2) (fe)

        fe = keras.layers.Conv2D(filters=128, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer=init, kernel_constraint=const) (fe)
        fe = keras.layers.BatchNormalization(momentum=.5) (fe)
        fe = keras.layers.LeakyReLU(.2) (fe)

        fe = keras.layers.Dropout(.4) (fe)

        # __change_begin__
        li = keras.layers.Embedding(self.n_classes, 30) (in_label)
        li = keras.layers.Dense(16 * 16) (li)
        li = keras.layers.Reshape((16, 16, 1)) (li)

        merge = keras.layers.Concatenate() ([fe, li])
        # __change_end__
        
        fe = keras.layers.Dropout(.4) (merge)#(fe)
        
        fe = keras.layers.Conv2D(filters=256, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer=init, kernel_constraint=const) (fe)
        fe = keras.layers.BatchNormalization(momentum=.5) (fe)
        fe = keras.layers.LeakyReLU(.2) (fe)

        fe = keras.layers.Dropout(.4) (fe)

        fe = keras.layers.Conv2D(filters=512, kernel_size=(4,4), strides=(2,2), padding="same", data_format="channels_last", kernel_initializer=init, kernel_constraint=const) (fe)
        fe = keras.layers.BatchNormalization(momentum=.5) (fe)
        fe = keras.layers.LeakyReLU(.2) (fe)

        fe = keras.layers.Flatten() (fe)

        fe = keras.layers.Dropout(.2) (fe)

        out_layer = keras.layers.Dense(1, activation="linear") (fe)
        model = keras.Model([in_image, in_label], out_layer)
        model.compile(loss=wassersteinLoss, optimizer=keras.optimizers.RMSprop(lr=.00005 * 2), metrics=["accuracy"])
        return model
    
    def generateAdversial(self):
        self.criticer.trainable = False
        gen_noise, gen_label = self.generator.input
        gen_output = self.generator.output
        gan_output = self.criticer ([gen_output, gen_label])
        model = keras.Model([gen_noise, gen_label], gan_output)
        model.compile(loss=wassersteinLoss, optimizer=keras.optimizers.RMSprop(lr=.00005 * 4), metrics=["accuracy"])
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
                realDataY = - np.ones(len(realImagesX))# + np.random.random_sample(len(realImagesX)) * .2
                fakeDataY = np.ones(len(realImagesX))# - np.random.random_sample(len(realImagesX)) * .2
                # NOTE train discriminator seperately on real and fake
                discriminatorMetricsReal = self.criticer.train_on_batch([realImagesX, labels], realDataY)
                discriminatorMetricsFake = self.criticer.train_on_batch([fakeImagesX, labels], fakeDataY)
                print("Discriminator: real loss: %f fake loss: %f" % (discriminatorMetricsReal[0], discriminatorMetricsFake[0]))
                averageDiscriminatorRealLoss.append(discriminatorMetricsReal[0])
                averageDiscriminatorFakeLoss.append(discriminatorMetricsFake[0])
                # NOTE train adversial model
                ganX = self.generateNoise(len(realImagesX))
                sampledLabels = list(range(self.n_classes)) * (len(realImagesX) // self.n_classes)
                sampledLabels += list(range(self.n_classes))[: len(realImagesX) - len(sampledLabels)]
                ganY = - np.ones(len(realImagesX))# + np.random.random_sample(len(realImagesX)) * .2
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
        fileNames = os.listdir(self.datasetDir + "/images3classes")
        fileNames = [file for file in fileNames if len(file.split(".")) == 2 and file.split(".")[1] == "jpg"][countStart : countEnd]
        images = [self.loadImage(file) for file in fileNames if len(file.split(".")) == 2 and file.split(".")[1] == "jpg"]        
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

NOISE_SHAPE = (1,1,100)
EPOCHS = 30
BATCH_SIZE = 64
IMAGE_SHAPE = (64,64,3)

if __name__ == "__main__":
    gan = EmotionGAN(NOISE_SHAPE, IMAGE_SHAPE)
    # gan = EmotionGAN(NOISE_SHAPE, IMAGE_SHAPE, keras.models.load_model("generator"), keras.models.load_model("criticer"))
    losses = gan.fit(EPOCHS, BATCH_SIZE)
    # gan.generator.save("generator")
    # gan.criticer.save("criticer")
    print("Training finished.")
    # print(gan.criticer.summary())