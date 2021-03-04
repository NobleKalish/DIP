import os
import shutil

def main():




    dir_path = os.path.dirname(os.path.realpath(__file__)) + "\DNIM\Image"

    groupnum = 0
    finalrestingplace = os.path.dirname(os.path.realpath(__file__)) + "\\final\\"
    finalrestingplace = os.path.normpath(finalrestingplace)

    createcopy = False  # CHANGE THIS TO FALSE TO PREVENT COPYING OVER!!!!!!
    if(createcopy):
        for root, subdirs, files in os.walk(dir_path):

            print("scanning root directory" + root)
            picnum = 1
            for f in files:
                print(f)
                if f.endswith(".jpg"):
                    shutil.copyfile(root+'\\'+f,finalrestingplace + "\group"+str(groupnum)+"_pic"+str(picnum)+".jpg")
                    picnum+=1
            groupnum+=1

    for entry in os.scandir(finalrestingplace): #iterate over every file in final
        photo1 = entry # entry is dict with key filepath and value filename
        for comparator in os.scandir(finalrestingplace): #compare every other image in the set
            if comparator[0] != entry[0]:
                feature_detection(entry, comparator)
    generate_summary()


def feature_detection(photo1, photo2):
    pass


def generate_summary():
    pass


if __name__ == "__main__":
    # execute only if run as a script
    main()
