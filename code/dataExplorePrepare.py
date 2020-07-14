import glob
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

dataDir = '/home/natasa/share/trafficSigns/data'
germanDataDir = os.path.join(dataDir, 'germany')
belgiumDataDir = os.path.join(dataDir, 'belgium')


totalNoImages = len(glob.glob(os.path.join(germanDataDir, '*', '*'))) +  len(glob.glob(os.path.join(belgiumDataDir, '*', '*')))

noClasses = 21

imgFilesGermany = [None]*noClasses
imgFilesBelgium = [None]*noClasses

for i in range(noClasses):
	imgFilesGermany[i] = glob.glob(os.path.join(germanDataDir, str(i), '*'))
	imgFilesBelgium[i] = glob.glob(os.path.join(belgiumDataDir, str(i), '*'))
 

noImagesG = [len(x) for x in imgFilesGermany] 
noImagesB = [len(x) for x in imgFilesBelgium] 

noImages = np.array(noImagesG) + np.array(noImagesB)


plt.bar(range(noClasses), noImages)
plt.title('Number of images per class')
plt.show()

from PIL import Image


sizes = []
for imgPath in glob.glob(os.path.join(germanDataDir, '*', '*')):
	img = Image.open(imgPath)
	sizes.append(img.size)
	
for imgPath in glob.glob(os.path.join(belgiumDataDir, '*', '*')):
	img = Image.open(imgPath)
	sizes.append(img.size)

size1, size2 = zip(*sizes)	

plt.scatter(size1, size2, s = 1)
plt.title('Image size')
plt.show()

plt.hist(size1)
plt.title('Distribution of image size')
plt.show()



trainDir = os.path.join(dataDir, 'train')
validationDir = os.path.join(dataDir, 'validation')

shutil.rmtree(trainDir); shutil.rmtree(validationDir);

if not os.path.exists(trainDir): os.makedirs(trainDir)

if not os.path.exists(validationDir): os.makedirs(validationDir)

shutil.rmtree(trainClassDir); shutil.rmtree(validationClassDir);

noClasses = 21

for i in range(0,noClasses):
	trainClassDir = os.path.join(trainDir,str(i))
	validationClassDir = os.path.join(validationDir,str(i))
	if not os.path.exists(trainClassDir): os.makedirs(trainClassDir)
	if not os.path.exists(validationClassDir):os.makedirs(validationClassDir)
	imgFiles = imgFilesGermany[i] + imgFilesBelgium[i]
	alreadyCopied = []
	for j, img in enumerate(imgFiles):
		imgBaseName = img.split('_')[0]
		if imgBaseName in alreadyCopied: continue
		alreadyCopied.append(imgBaseName)
		#if np.random.binomial(1,0.25):
		if np.random.uniform()<0.25:
			copyDir = validationClassDir
		else:
			copyDir =  trainClassDir
		for imgFile in glob.glob(img.split('_')[0]+'*'):
			_ = shutil.copy(imgFile, copyDir)


imgFilesTrain = [None]*noClasses
imgFilesValidation = [None]*noClasses

for i in range(noClasses):
	imgFilesTrain[i] = glob.glob(os.path.join(germanDataDir, str(i), '*'))
	imgFilesValidation[i] = glob.glob(os.path.join(belgiumDataDir, str(i), '*'))

noImagesTrain = [len(x) for x in imgFilesTrain] 
noImagesValidation = [len(x) for x in imgFilesValidation] 
	
print(noImagesTrain)
print(noImagesValidation)


# copy all images into one dir
allDir = os.path.join(dataDir, 'all')
for i in range(1, noClasses):
    imgFiles = glob.glob(os.path.join(germanDataDir, str(i), '*')) + glob.glob(os.path.join(belgiumDataDir, str(i), '*'))
    allClassDir = os.path.join(allDir, str(i)) 
    os.makedirs(allClassDir)
    for imgFile in imgFiles:                
        _ = shutil.copy(imgFile, allClassDir)


len(glob.glob(os.path.join(allDir, '*' , '*')))
	
