from datetime import datetime
import os
import shutil
import cv2 as cv
from collections import defaultdict
from threading import Thread


def main():
    successes = 0
    total = 0

    dir_path = os.path.dirname(os.path.realpath(__file__)) + "\DNIM\Image"

    groupnum = 0
    finalrestingplace = os.path.dirname(os.path.realpath(__file__)) + "/final"
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

    threads = []
    for n in range(10, 11):  # Noble (10,14) Tyler (14,18)
        print("Spawning thread " + str(n))
        dir = finalrestingplace + "/group" + str(n)
        dir = os.path.normpath(dir)
        t = Thread(target=thread_func, args=(dir,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


def thread_func(finalrestingplace):
    total = 0
    successes = 0

    for entry in os.scandir(finalrestingplace):  # iterate over every file in final
        total += 1
        sift = cv.SIFT_create()
        img1 = entry.path
        img1 = cv.imread(img1, cv.IMREAD_GRAYSCALE)
        keypoints1, description1 = sift.detectAndCompute(img1, None)
        print("total is at: " + str(total))
        time = datetime.now()
        print(time.strftime("%H:%M:%S"))
        entry_dict = defaultdict(
            list)  # key string {representing group} value list of ints {contains all matches to images in said group}
        for group in os.scandir(os.path.dirname(finalrestingplace)):
            for comparator in os.scandir(group):  # compare every other image in the set
                currgroup = group.name
                if comparator.path != entry.path:
                    entry_dict[currgroup].append(flann(sift, description1, comparator.path))
        if generate_summary(entry_dict, os.path.basename(finalrestingplace)):
            successes += 1

    print("Report for " + os.path.basename(finalrestingplace))
    print("Total images tested: " + str(total))
    print("Images matched correctly: " + str(successes))
    print("Accuracy rate: " + str((successes / total) * 100) + "%")


def feature_detection(sift, description1, photo2):
    img2 = cv.imread(photo2, cv.IMREAD_GRAYSCALE)
    # find the keypoints and descriptors with SIFT
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


def flann(sift, description1, photo2):
    img2 = cv.imread(photo2, cv.IMREAD_GRAYSCALE)  # trainImage
    # Initiate SIFT detector
    # find the keypoints and descriptors with SIFT
    kp2, des2 = sift.detectAndCompute(img2, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(description1, des2, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
    return len(matchesMask)


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
