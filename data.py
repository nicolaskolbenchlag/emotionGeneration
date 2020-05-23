import imageio
import os
import numpy as np

def loadDataset(path=""):
    labels, images = [], []
    for labelsFile in os.listdir(path + "AffWild2/annotations/EXPR_Set/Training_Set")[:1]:
        with open(path + "AffWild2/annotations/EXPR_Set/Training_Set/" + labelsFile) as file:
            lines = file.readlines()
        labels += [l.strip() for l in lines]
        imagesDir = labelsFile.split(".")[0]
        images += [imageio.imread(path + "AffWild2/cropped_aligned/cropped_aligned/" + imagesDir + "/" + file) for file in os.listdir(path + "AffWild2/cropped_aligned/cropped_aligned/" + imagesDir)]
    x, y = np.array(images), np.array(labels)
    return x, y

def writeImage(image, name, path="generatedImages/"):
    imageio.imwrite(path + name, image[:, :, 0])

if __name__ == "__main__":
    trainX, trainY = loadDataset()
    print(trainX.shape)