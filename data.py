import imageio
import os
import numpy as np

def loadDataset(path=""):
    labels, images = [], []
    for labelsFile in os.listdir(path + "MyAffWild2/annotations/EXPR_Set/Training_Set"):
        with open(path + "MyAffWild2/annotations/EXPR_Set/Training_Set/" + labelsFile) as file:
            lines = file.readlines()
        labels += [l.strip() for l in lines]
        imagesDir = labelsFile.split(".")[0]
        images += [imageio.imread(path + "MyAffWild2/cropped_aligned/cropped_aligned/" + imagesDir + "/" + file) for file in os.listdir(path + "MyAffWild2/cropped_aligned/cropped_aligned/" + imagesDir)]
    x, y = np.array(images), np.array(labels)
    return x, y

if __name__ == "__main__":
    trainX, trainY = loadDataset()
    print(trainX.shape)