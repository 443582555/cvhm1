
import cv2
import sys
import numpy as np
import glob
from matplotlib import pyplot as plt

class pano:
    def __init__(self,limg, rimg):
        self.img1 = limg
        self.img2 = rimg
        self.im3=[]

    def xsiftFeature(self):
        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(self.img1,None)
        kp2, des2 = sift.detectAndCompute(self.img2,None)
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)
        # Apply ratio test
        good=[]
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
                # cv.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(self.img1,kp1,self.img2,kp2,good,None,flags=2)
        return kp1,kp2,good


    def get_homography(self,kp1,kp2,good):
        img1_pt = []
        img2_pt = []
        for match in good:
            img1_pt.append(kp1[match[0].queryIdx].pt)
            img2_pt.append(kp2[match[0].trainIdx].pt)
        #img1_pt = np.float32(img1_pt).reshape(-1,1,2)
        #img2_pt = np.float32(img2_pt).reshape(-1,1,2)
        img1_pt = np.float32(img1_pt)
        img2_pt = np.float32(img2_pt)
        M,mask = cv2.findHomography(img2_pt,img1_pt,cv2.RANSAC,5.0)

        return M

    def get_stitched_image(self, M):

        # Get width and height of input images
        w1, h1 = self.img1.shape[:2]
        w2, h2 = self.img2.shape[:2]

        # Get the canvas dimesions
        img1_dims = np.float32([[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1,1,2)
        img2_dims_temp = np.float32([[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1,1,2)

        # Get relative perspective of second image
        img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

        # Resulting dimensions
        result_dims = np.concatenate((img1_dims, img2_dims), axis=0)

        # Getting images together
        # Calculate dimensions of match points
        [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel())
        [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel())

        # Create output array after affine transformation
        transform_dist = [-x_min, -y_min]
        transform_array = np.array([[1, 0, transform_dist[0]],
                                    [0, 1, transform_dist[1]],
                                    [0, 0, 1]])

        # Warp images to get the resulting image
        result_img = cv2.warpPerspective(self.img2, transform_array.dot(M),
                                         (x_max - x_min, y_max - y_min))
        temp_img = result_img.copy()
        temp_img[transform_dist[1]:w1 + transform_dist[1],
        transform_dist[0]:h1 + transform_dist[0]] = self.img1

        for i in range(len(temp_img)):
            for j in range(len(temp_img[0])):
                for k in range(len(temp_img[0][0])):
                    result_img[i][j][k]= max(result_img[i][j][k],temp_img[i][j][k])
        #result_img = result_img[:, 40:]
    # Return the result
        return result_img
