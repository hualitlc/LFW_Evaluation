
import numpy as np

# generate the image name list which has all the names,
# and each name has N rows in the list
def generateList(lfwNamesTxt):
    srcDir = 'F:/dataset/LFW/'
    f_pathname = open(srcDir+"lfw-names-pathandname.txt", "w")

    myfile_lfwNames = open(srcDir+lfwNamesTxt)
    lines = len(myfile_lfwNames.readlines())

    myfile_lfwNames = open(srcDir+lfwNamesTxt)
    
    for i in range(lines):
        line = myfile_lfwNames.readline()
        r=line.split('\t')
        name=r[0]
        num=int(r[1])
        for j in range(num):
            filename = "{0}/{0}_{1:04d}.jpg".format(name, int(j))
            f_pathname.write(filename+'\t'+str(i)+'\n')
    f_pathname.close()

# select the names which have more than 2 pictures(>=2)
def selectMore2Pic():
    srcDir = 'F:/dataset/LFW/'

    f_more2pic = open(srcDir+"lfw-names-more2pic.txt", "w")

    imagePath = srcDir + 'lfw/'
    lfwNamesPath = srcDir + 'lfw-names.txt'

    myfile_lfwNames = open(lfwNamesPath)
    lines = len(myfile_lfwNames.readlines())
    print 'There are %d lines in %s' %(lines, lfwNamesPath)

    myfile_lfwNames = open(lfwNamesPath)
    for i in range(lines):
        line = myfile_lfwNames.readline()
        #print i,line
        r=line.split("\t")
        if int(r[1])>=2:
            #print i,line
            f_more2pic.write(line)

    f_more2pic.close()

# generate the image name list in which (where) the name has more than two pics,
# and each name has N rows in the list
def generateMore2PicList(lfwNamesMore2PicTxt):
    srcDir = 'F:/dataset/LFW/'
    f_more2pic_pathname = open(srcDir+"lfw-names-more2pic-pathandname.txt", "w")

    myfile_more2pic = open(srcDir+lfwNamesMore2PicTxt)
    lines = len(myfile_more2pic.readlines())

    myfile_more2pic = open(srcDir+lfwNamesMore2PicTxt)
    
    for i in range(lines):
        line = myfile_more2pic.readline()
        r=line.split('\t')
        name=r[0]
        num=int(r[1])
        for j in range(num):
            filename = "{0}/{0}_{1:04d}.jpg".format(name, int(j))
            f_more2pic_pathname.write(filename+'\t'+str(i)+'\n')
    f_more2pic_pathname.close()

# define the function to parse the pairsDevTrain.txt and pairsDevTest.lfwNamesMore2PicTxt


def main():
    #selectMore2Pic()
    #generateList('lfw-names.txt')
    #generateMore2PicList('lfw-names-more2pic.txt')

if __name__ == '__main__':
    main()



    


#files = ['lfw-names.txt', 'pairsDevTrain.txt', 'pairsDevTest.txt']

# def resolve_filename(format):
#     imfile = "lfw"
#     if format == "funneled":
#         imfile = "lfw-funneled"
#     elif format == "deepfunneled":
#         imfile = "lfw-deepfunneled"
#     return imfile

# def loadImage(tar, basename, name, number):
#     filename = "{0}/{1}/{1}_{2:04d}.jpg".format(basename, name, int(number))
#     return imread(tar.extractfile(filename))

# def loadImagePairFromRow(tar, basename, r):
#     if(len(r) == 3):
#         return [loadImage(tar, basename, r[0], r[1]), loadImage(tar, basename, r[0], r[2])]
#     else:
#         return [loadImage(tar, basename, r[0], r[1]), loadImage(tar, basename, r[2], r[3])]

# def loadLabelsFromRow(r):
#     if(len(r) == 3):
#         return 1
#     else:
#         return 0

# def load_images(split, tar, basename, rows):
#     image_list = []
#     progress_bar_context = progress_bar(
#         name='{} images'.fromat(split), maxval=len(rows),
#         prefix='Coverting'
#     )
#     with progress_bar_context as bar:
#         for i, row in enumerate(rows):
#             image_list.append(loadImagePairFromRow(tar, basename, row))
#             bar.update(i)
#     return np.array(image_list)


