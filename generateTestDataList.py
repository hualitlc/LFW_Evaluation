# -*- coding: utf-8 -*-
# main code from https://github.com/hqli/face_recognition/blob/master/lfw_test_deal.py, thank the author hqli again.

import os
import numpy as np
import cv2


# read "pairs.txt" and split the content into different componets
def read_paris(filelist):
    fp=open(filelist,'r')
    result=[]
    for lines in fp.readlines():
        lines=lines.replace("\n","").split("\t")
    # if(len(lines) == 3):
    #     return [loadImage(basename, lines[0], lines[1]), loadImage(basename, lines[0], lines[2])]
    # else:
    #     return [loadImage(basename, lines[0], lines[1]), loadImage(basename, lines[2], lines[3])]        
        if len(lines)==2:
            print "lenth=2:"+str(lines)
            continue
        elif len(lines)==3:
            pairs={
                'flag':1,
                'img1':lines[0],
                'img2':lines[0],
                'num1':lines[1],
                'num2':lines[2],
                }
            result.append(pairs)
            continue
        elif len(lines)==4:
            pairs={
                'flag':2,
                'img1':lines[0],
                'num1':lines[1],
                'img2':lines[2],
                'num2':lines[3],
                }
            result.append(pairs)
        else:
            print "read file Error!"
            exit()
    fp.close
    print "Read paris.txt DONE!"
    return result

# accoding to the pairs.txt to generate pairs list and then generate the left.txt, right.txt and label.txt
def split_pairs(pairslist,lfwdir,ext='bmp'):
    num=0
    sum=len(pairslist)

    #lfw图像组织形式，只有图像：0，文件夹+图像：1
    # flag=0
    # subdir=dir_subdir(lfwdir)
    # if len(subdir):
    #     flag=1

    path_left=""
    path_right=""
    label=""

    #left.txt and right.txt
    #文件夹+图像形式
    print "split pairs.txt"
    # if flag==1:
    for lines in pairslist:
        num=num+1
        if num%100==0:
            print str(num)+"/"+str(sum)

        dir_left=lfwdir+lines['img1']+'/'
        dir_right=lfwdir+lines['img2']+'/'
        
        file_left=lines['img1']+'_'+str("%04d" % int(lines["num1"]))+'.'+ext
        file_right=lines['img2']+'_'+str("%04d" % int(lines["num2"]))+'.'+ext
        
        path_left=path_left+dir_left+file_left+"\n"
        path_right=path_right+dir_right+file_right+"\n"

    #图像
    # else:
    # for lines in pairslist:
    #     num=num+1
    #     print str(num)+"/"+str(sum)
    #     path_left=path_left+lfwdir+lines['img1']+'_'+str("%04d" % int(lines["num1"]))+'.'+ext+"\n"
    #     path_right=path_right+lfwdir+lines['img2']+'_'+str("%04d" % int(lines["num2"]))+'.'+ext+"\n"
    
    #label.txt
    for lines in pairslist:    
        if int(lines["flag"])==1:
            label=label+"0\n"
        else:
            label=label+"1\n"
        
    result={
        'path_left':path_left,
        'path_right':path_right,
        'label':label}
    return result


def testdeal(pairs='pairs.txt',lfwdir='/home/tianluchao/Documents/VIPLFaceNet/aligned_lfw/',ext='bmp'): #,changdir='lfw_chang/',grey=False,resize=False,height=64,width=64
    pairslist=read_paris(pairs)
    # if grey==True or resize==True:
    #     filelist=dirdir_list(lfwdir,ext)
    #     filelist=grey_resize(lfwdir,filelist,changdir,grey,resize,height,width)
    #     lfwdir=changdir
    pairs_result=split_pairs(pairslist,lfwdir,ext)
    return pairs_result

def write_pairs(pairs_result,savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    left=savepath+'lfw_left.txt'
    right=savepath+'lfw_right.txt'
    label=savepath+'lfw_label.txt'

    fp_left=open(left,'w')
    fp_right=open(right,'w')
    fp_label=open(label,'w')

    fp_left.write(pairs_result['path_left'])
    fp_right.write(pairs_result['path_right'])
    fp_label.write(pairs_result['label'])

    fp_left.close()
    fp_right.close()
    fp_label.close()


def demo_lfw():
    caffe_dir='/home/tianluchao/ProgramFiles/caffe/'
    pairs=caffe_dir+'examples/lfwEvaluation/pairs.txt'
    lfwdir='/home/tianluchao/Documents/VIPLFaceNet/aligned_lfw/'
    ext='bmp'
    pairs_result=testdeal(pairs,lfwdir,ext)
    savepath=caffe_dir+'examples/lfwEvaluation/'
    write_pairs(pairs_result,savepath)   


# def grey_pairs():
#     DEEPID='data/deepID_grey/'
#     pairs=DEEPID+'pairs.txt'
#     lfwdir=DEEPID+'lfwcrop_grey/faces/' 
#     ext='pgm'
#     pairs_result=testdeal(pairs,lfwdir,ext)
#     return pairs_result
    
# def demo_webface_resize():
#     DEEPID='/home/ikode/caffe-master/data/deepID/'
#     pairs=DEEPID+'pairs.txt'
# #    lfwdir=DEEPID+'lfwcrop_color/faces/'
#     lfwdir='/media/ikode/Document/big_materials/document/deep_learning/caffe/face_datasets/webface/croped/'
#     ext='jpg'
#     lfw_chang='/media/ikode/Document/big_materials/document/deep_learning/caffe/face_datasets/webface/change/'

#     pairs_result=testdeal(pairs,lfwdir,ext,lfw_chang,False,True)
#     savepath=caffe_dir+'examples/deepID/webface_'
#     write_pairs(pairs_result,savepath)   

# def demo_color_resize():
#     DEEPID='/home/ikode/caffe-master/data/deepID/'
#     pairs=DEEPID+'pairs.txt'
# #    lfwdir=DEEPID+'lfwcrop_color/faces/'
#     lfwdir='/media/ikode/Document/big_materials/document/deep_learning/caffe/face_datasets/webface/croped/'
#     ext='jpg'
#     lfw_chang='/media/ikode/Document/big_materials/document/deep_learning/caffe/face_datasets/webface/change/'

#     pairs_result=testdeal(pairs,lfwdir,ext,lfw_chang,False,True)
#     return pairs_result
    

if __name__=='__main__':
    
    demo_lfw()




# import numpy as np

# # generate the image name list which has all the names,
# # and each name has N rows in the list
# def generateList(lfwNamesTxt):
#     srcDir = 'F:/dataset/LFW/'
#     f_pathname = open(srcDir+"lfw-names-pathandname.txt", "w")

#     myfile_lfwNames = open(srcDir+lfwNamesTxt)
#     lines = len(myfile_lfwNames.readlines())

#     myfile_lfwNames = open(srcDir+lfwNamesTxt)
    
#     for i in range(lines):
#         line = myfile_lfwNames.readline()
#         r=line.split('\t')
#         name=r[0]
#         num=int(r[1])
#         for j in range(num):
#             filename = "{0}/{0}_{1:04d}.jpg".format(name, int(j))
#             f_pathname.write(filename+'\t'+str(i)+'\n')
#     f_pathname.close()

# # select the names which have more than 2 pictures(>=2)
# def selectMore2Pic():
#     srcDir = 'F:/dataset/LFW/'

#     f_more2pic = open(srcDir+"lfw-names-more2pic.txt", "w")

#     imagePath = srcDir + 'lfw/'
#     lfwNamesPath = srcDir + 'lfw-names.txt'

#     myfile_lfwNames = open(lfwNamesPath)
#     lines = len(myfile_lfwNames.readlines())
#     print 'There are %d lines in %s' %(lines, lfwNamesPath)

#     myfile_lfwNames = open(lfwNamesPath)
#     for i in range(lines):
#         line = myfile_lfwNames.readline()
#         #print i,line
#         r=line.split("\t")
#         if int(r[1])>=2:
#             #print i,line
#             f_more2pic.write(line)

#     f_more2pic.close()

# # generate the image name list in which (where) the name has more than two pics,
# # and each name has N rows in the list
# def generateMore2PicList(lfwNamesMore2PicTxt):
#     srcDir = 'F:/dataset/LFW/'
#     f_more2pic_pathname = open(srcDir+"lfw-names-more2pic-pathandname.txt", "w")

#     myfile_more2pic = open(srcDir+lfwNamesMore2PicTxt)
#     lines = len(myfile_more2pic.readlines())

#     myfile_more2pic = open(srcDir+lfwNamesMore2PicTxt)
    
#     for i in range(lines):
#         line = myfile_more2pic.readline()
#         r=line.split('\t')
#         name=r[0]
#         num=int(r[1])
#         for j in range(num):
#             filename = "{0}/{0}_{1:04d}.jpg".format(name, int(j))
#             f_more2pic_pathname.write(filename+'\t'+str(i)+'\n')
#     f_more2pic_pathname.close()

# # define the function to parse the pairsDevTrain.txt and pairsDevTest.lfwNamesMore2PicTxt


# def main():
#     #selectMore2Pic()
#     #generateList('lfw-names.txt')
#     #generateMore2PicList('lfw-names-more2pic.txt')

# if __name__ == '__main__':
#     main()



    


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


