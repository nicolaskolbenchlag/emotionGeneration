import imageio
import os
import numpy as np

def loadDataset(countStart=None, countEnd=None, path=""):
    labels, images = [], []

    for labelsFile in os.listdir(path + "AffWild2/annotations/EXPR_Set/Training_Set")[countStart : countEnd]:
        with open(path + "AffWild2/annotations/EXPR_Set/Training_Set/" + labelsFile) as file:
            lines = file.readlines()
        imagesDir = labelsFile.split(".")[0]
        i = 0
        for file in os.listdir(path + "AffWild2/cropped_aligned/cropped_aligned/" + imagesDir):
            try:
                image = imageio.imread(path + "AffWild2/cropped_aligned/cropped_aligned/" + imagesDir + "/" + file)
                images.append(image)
                labels.append(lines[i].strip())
            except:
                pass
            i += 1
    x, y = np.array(images), np.array(labels)
    return x, y

def writeImage(image, name, path="generatedImages/"):
    imageio.imwrite(path + name, image[:, :, 0])

if __name__ == "__main__":
    trainX, trainY = loadDataset()
    print(trainX.shape)