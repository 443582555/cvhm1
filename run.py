import cv2
from pano import pano
import sys
import glob


if __name__ == '__main__':
	dic = sys.argv[1]
	sourceImg = []
	
	for fn in sorted(glob.glob(dic+'/*.JPG')):
		im = cv2.imread(fn)
		sourceImg.append(im)
	if len(sourceImg)==0:
		for fn in sorted(glob.glob(dic+'/*.jpg')):
			im = cv2.imread(fn)
			sourceImg.append(im)

	currentImg = sourceImg[0]
	for i in range(6):
		if i+1<len(sourceImg):
			mstich = pano(currentImg,sourceImg[i+1])
			kp1,kp2,good = mstich.xsiftFeature()
			M = mstich.get_homography(kp1,kp2,good)
			result = mstich.get_stitched_image(M)
			currentImg = result
	cv2.imwrite('mosic.jpg', result)
