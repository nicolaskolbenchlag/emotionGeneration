import data

if __name__ == "__main__":
	print("Start loading images")
	imagesLoad = data.loadDataset(countStart=0, countEnd=10)[0]
	print("Shape:", imagesLoad.shape)