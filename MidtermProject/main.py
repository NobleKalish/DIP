import os
import shutil
import time

import cv2 as cv
import itertools
import csv
from collections import defaultdict

import numpy as np


class Comparison:
    def __init__(self, numMatches, img1, isDay1, img2, isDay2):
        self.numMatches = numMatches
        self.img1 = img1
        self.isDay1 = isDay1
        self.img2 = img2
        self.isDay2 = isDay2


def main():
    successes = 0
    total = 0
    entry_dict = defaultdict(
        list)  # key string {representing group} value list of ints {contains all matches to images in said group}
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "\DNIM\Image"

    groupnum = 0
    finalrestingplace = os.path.dirname(os.path.realpath(__file__)) + "/final/"
    finalrestingplace = os.path.normpath(finalrestingplace)

    createcopy = False  # CHANGE THIS TO FALSE TO PREVENT COPYING OVER!!!!!!
    if (createcopy):
        for root, subdirs, files in os.walk(dir_path):
            os.makedirs(finalrestingplace + "\group" + str(groupnum))
            print("scanning root directory" + root)
            picnum = 1
            for f in files:
                print(f)
                if f.endswith(".jpg"):
                    shutil.copyfile(root + '\\' + f,
                                    finalrestingplace + "\group" + str(groupnum) + "\pic" + str(picnum) + ".jpg")
                    picnum += 1
            groupnum += 1

    for n in range(4,7):
        dir = finalrestingplace + "/group" + str(n)
        dir = os.path.normpath(dir)
        runTestSuite(dir, n)


def runTestSuite(finalrestingplace, groupNum):
    with open("results/group"+str(groupNum)+".csv", 'w') as csv_file:

        fields = ["numMatches","img1","isDay1","img2","isDay2"]
        wr = csv.DictWriter(csv_file, delimiter=',', fieldnames=fields)
        wr.writeheader()
        relativePath = "final/group" + str(groupNum) + "/"
        for img1, img2 in itertools.combinations(os.listdir(finalrestingplace),2):

            img1 = pre_process_image(relativePath + img1)
            img2 = pre_process_image(relativePath + img2)

            compResult = Comparison(0,img1,is_photo_night_time(relativePath + img1),
                                    img2,is_photo_night_time(relativePath + img2))
            compResult.numMatches = feature_detection(relativePath + img1, relativePath + img2)
            wr.writerow(vars(compResult))
            csv_file.flush()

def is_photo_night_time(image):
    mean_blur = cv.mean(cv.blur(cv.imread(image, cv.IMREAD_GRAYSCALE), (5, 5)))[0]
    print("Image: " + image + " mean: " + str(mean_blur))
    return mean_blur > 75


def pre_process_image(image):
    # do preprocessing
    print(image)
    image = cv.imread(image)

    cv.imshow("CLAHE image", image)
    time.sleep(5)
    #image = cv.resize(image, (500, 600))
    image_bw = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=5)
    final_img = clahe.apply(image_bw) + 30

    _, ordinary_img = cv.threshold(image_bw, 155, 255, cv.THRESH_BINARY)
    cv.imshow("ordinary threshold", ordinary_img)
    cv.imshow("CLAHE image", final_img)
    time.sleep(5)

    return image


def feature_detection(photo1, photo2):
    img1 = cv.imread(photo1, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(photo2, cv.IMREAD_GRAYSCALE)
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    keypoints1, description1 = sift.detectAndCompute(img1, None)
    keypoints2, description2 = sift.detectAndCompute(img2, None)
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(description1, description2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append([m])
    return len(good)  # return number of matches between these two images


def generate_summary(dict, group):
    largestGroup = ""
    largestAvg = 0
    for k, _ in dict.items():
        sum = 0
        for elt in dict[k]:
            sum += elt
        avg = sum / len(dict[k])
        if avg > largestAvg:
            largestGroup = k
            largestAvg = avg

    return largestGroup == group


if __name__ == "__main__":
    # execute only if run as a script
    main()
