import os
import shutil
import cv2 as cv

def main():
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "\DNIM\Image"

    groupnum = 0
    finalrestingplace = os.path.dirname(os.path.realpath(__file__)) + "\\final\\"
    finalrestingplace = os.path.normpath(finalrestingplace)

    createcopy = False  # CHANGE THIS TO FALSE TO PREVENT COPYING OVER!!!!!!
    if (createcopy):
        for root, subdirs, files in os.walk(dir_path):

            print("scanning root directory" + root)
            picnum = 1
            for f in files:
                print(f)
                if f.endswith(".jpg"):
                    shutil.copyfile(root + '\\' + f,
                                    finalrestingplace + "\group" + str(groupnum) + "_pic" + str(picnum) + ".jpg")
                    picnum += 1
            groupnum += 1

    for entry in os.scandir(finalrestingplace):  # iterate over every file in final
        for comparator in os.scandir(finalrestingplace):  # compare every other image in the set
            if comparator.path != entry.path:
                feature_detection(entry.path, comparator.path)
    generate_summary()


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
    return good

def generate_summary():
    pass


if __name__ == "__main__":
    # execute only if run as a script
    main()
