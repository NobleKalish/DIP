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
    for n in range(1,10): # Noble (10,14) Tyler (14,18)
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
                currgroup = comparator.name.split('_')[0]
                if comparator.path != entry.path:
                    entry_dict[currgroup].append(feature_detection(sift, description1, comparator.path))
            if generate_summary(entry_dict, entry.name.split('_')[0]):
                successes += 1
    
    print("Report for " + os.path.basename(finalrestingplace))
    print("Total images tested: " + total)
    print("Images matched correctly: " + successes)
    print("Accuracy rate: " + (successes / total) * 100 + "%")


def feature_detection(sift, description1, photo2):
    #img1 = cv.imread(photo1, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(photo2, cv.IMREAD_GRAYSCALE)
    # Initiate SIFT detector
    # sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    #keypoints1, description1 = sift.detectAndCompute(img1, None)
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
