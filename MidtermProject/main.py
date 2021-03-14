import os
import shutil
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict


def main():
    print(cv.getBuildInformation())
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

    for n in range(1, 18): # Noble (10,14) Tyler (14,18)
        dir = finalrestingplace + "/group" + str(n)
        dir = os.path.normpath(dir)

        for entry in os.scandir(dir):  # iterate over every file in final
            total += 1
            print(f'Total is at {total}')
            is_entry_night_time = is_photo_night_time(entry.path)
            for comparator in os.scandir(dir):  # compare every other image in the set
                currgroup = comparator.name.split('_')[0]
                if comparator.path != entry.path:
                    is_comparator_night_time = is_photo_night_time(comparator.path)
                    # entry_dict[currgroup].append(feature_detection(entry.path, comparator.path))
            # if generate_summary(entry_dict, entry.name.split('_')[0]):
            #     successes += 1

    print(f"Total images tested: {total}")
    print(f"Images matched correctly: {successes}")
    print(f"Accuracy rate: {(successes / total) * 100} %")


def is_photo_night_time(entry):
    image = cv.imread(entry, cv.IMREAD_GRAYSCALE)
    mean_blur = cv.mean(cv.blur(image, (5, 5)))[0]
    return mean_blur < 127


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
        if m.distance < 0.75 * n.distance:
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
