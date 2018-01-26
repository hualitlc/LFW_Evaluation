#!/usr/bin/env python
#-*- coding:utf-8 -*-

# main code from https://github.com/hqli/face_recognition/blob/master/DeepIDTest.py, thank the author hqli again.

import os
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import skimage
import sys
# _command="rm -rf caffe\n"
# _command+="ln -s "+ os.path.expanduser('~/')+"/caffe-master/python/caffe ./caffe"
# os.system(_command)

import caffe
import sklearn.metrics.pairwise as pw

# from DeepID import *
# from lfw_test_deal import *
from generateTestDataList import *

class ModelTest():#GoogLeNetModel
    pairs=''#pairs.txt的路径

    itera=189090#训练好的模型的迭代次数

    model=''#模型
    imgmean_npy=''#图像均值的npy
    deploy=''

    savepath=''#准确率与ROC曲线图像存储路径,一般放在deepID/num/test/下

 
    left=''#pairs.txt分离的左图像路径的文本,放在工程文件夹下
    right=''#pairs.txt分离的右图像路径的文本
    label=''#pairs.txt分离的标签的文本

    accuracy=''#准确率的文件
    predict=''#预测值的文件

    lfwpath=''#已经剪切好的lfw图像
    roc=''#roc图像

    def __init__(self,prj,caffepath,prjpath,datapath,num,types,pairs,itera,lfwpath):
        #GoogLeNetModel.__init__(self,prj,caffepath,prjpath,datapath,num)

        self.itera=itera

        self.pairs=pairs
        self.lfwpath=lfwpath

        self.model=prjpath+'googlenet_quick_total189090_over50_2nd'+'_iter_'+str(itera)+'.caffemodel'
        self.deploy=prjpath+'deploy.prototxt'

        self.imgmean_npy=prjpath+'mean.npy'

        if types==1:
            self.left=prjpath+'lfw_left.txt'
            self.right=prjpath+'lfw_right.txt'
            self.label=prjpath+'lfw_label.txt'
        # elif types==2:
        #     self.left=prjpath+'train_left.txt'
        #     self.right=prjpath+'train_right.txt'
        #     self.label=prjpath+'train_label.txt'
        else:
            print 'types input ERROR'
            exit()
        
        self.savepath=prjpath+'result_5120dim/'
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)
        self.accuracy=self.savepath+prj+'_'+str(num)+'_'+str(itera)+'_accuracy.txt'
        self.predict=self.savepath+prj+'_'+str(num)+'_'+str(itera)+'_predict.txt'
        self.roc=self.savepath+prj+'_'+str(num)+'_'+str(itera)+'_roc'


    # def split_pairs(self):
    #     ext='bmp'
	#     print self.lfwpath
    #     pairs_result=testdeal(self.pairs, self.lfwpath, ext)

    #     fp_left=open(self.left,'w')
    #     fp_right=open(self.right,'w')
    #     fp_label=open(self.label,'w')

    #     fp_left.write(pairs_result['path_left'])
    #     fp_right.write(pairs_result['path_right'])
    #     fp_label.write(pairs_result['label'])

    #     fp_left.close()
    #     fp_right.close()
    #     fp_label.close()

     
    # @staticmethod
    # def fileopt(filename,content):
    #     fp=open(filename,'w')
    #     fp.write(content)
    #     fp.close()

    @staticmethod
    def read_imagelist(filelist):
        '''
        @brief：从列表文件中，读取图像数据到矩阵文件中
        @param： filelist 图像列表文件
        @return ：4D 的矩阵
        '''
        fid=open(filelist)
        lines=fid.readlines()
        test_num=len(lines)
        fid.close()
        X=np.empty((test_num,3,224,224))
        i =0
        for line in lines:
            word=line.split('\n')
            filename=word[0]
            im1=skimage.io.imread(filename,as_grey=False)
            image =skimage.transform.resize(im1,(224, 224))*255
            # if image.ndim<3:
            #     print 'gray:'+filename
            #     X[i,0,:,:]=image[:,:]
            #     X[i,1,:,:]=image[:,:]
            #     X[i,2,:,:]=image[:,:]
            # else:
            X[i,0,:,:]=image[:,:,0]
            X[i,1,:,:]=image[:,:,1]
            X[i,2,:,:]=image[:,:,2]
            i=i+1
        return X

    @staticmethod
    def read_labels(label):
        '''
        读取标签列表文件
        '''
        fin=open(label)
        lines=fin.readlines()
        labels=np.empty((len(lines),))
        k=0;
        for line in lines:
            labels[k]=int(line)
            k=k+1
        fin.close()
        return labels

    @staticmethod
    def calculate_accuracy(distance,labels,num):    
        '''
        #计算识别率,
        选取阈值，计算识别率
        '''
        print "calculate_accuracy"  
        accuracy = []
        predict = np.empty((num,))
        threshold = 0.2
        while threshold <= 0.8 :
            for i in range(num):
                if distance[i] >= threshold:
                     predict[i] =1
                else:
                     predict[i] =0
            predict_right =0.0
            for i in range(num):
                if predict[i]==labels[i]:
                  predict_right = 1.0+predict_right
            current_accuracy = (predict_right/num)
            accuracy.append(current_accuracy)
            threshold=threshold+0.001
        return np.max(accuracy)


    @staticmethod
    def draw_roc_curve(fpr,tpr,title='cosine',save_name='roc_lfw'):
        '''
        画ROC曲线图
        '''
        print "draw_roc_curve"
        plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], 'k--') #'k--'是一个线型选项，用于告诉matplotlib绘制黑色虚线图。
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic using: '+title)
        plt.legend(loc="lower right") 
        '''
        UserWarning: No labelled objects f on individual plots.
        warnings.warn("No labelled objects found. "
        并不是大问题，主要是因为You have to provide a label=.. keyword 
        in the plot function for each line you want to plot, as matplotlib 
        does not automatically detect names from a numpy structured array.
        可参考# https://stackoverflow.com/questions/16488037/how-to-use-names-when-importing-csv-data-into-matplotlib
        '''
    #    plt.show()
        plt.savefig(save_name+'.png')


    def evaluate(self,metric='cosine'):
        '''
        @brief: 评测模型的性能
        @param：itera： 模型的迭代次数
        @param：metric： 度量的方法
        '''
        # 转换均值图像数据　-->npy格式文件
        # fin=self.imgmean
        # fout=self.imgmean_npy
        # blob = caffe.proto.caffe_pb2.BlobProto()
        # data = open( fin , 'rb' ).read()
        # blob.ParseFromString(data)
        # arr = np.array( caffe.io.blobproto_to_array(blob) )
        # out = arr[0]
        # np.save( fout , out )
        
        print "evaluate"
        caffe.set_mode_gpu()
        net = caffe.Classifier(self.deploy,self.model,mean=np.load(self.imgmean_npy))
        #需要对比的图像，一一对应
        filelist_left=self.left
        filelist_right=self.right
        filelist_label=self.label

        print 'network input :' , net.inputs  
        print 'network output： ', net.outputs
        #提取左半部分的特征
        #print filelist_left
        X=ModelTest.read_imagelist(filelist_left)
        test_num=np.shape(X)[0]
        
        #data 是输入层的名字
        out = net.forward_all(data=X)
        print out
        feature1 = np.float64(out['googlenetOutput'])
        feature1=np.reshape(feature1,(test_num,5120))
        #np.savetxt('feature1.txt', feature1, delimiter=',')

        #提取右半部分的特征
        X=ModelTest.read_imagelist(filelist_right)
        out = net.forward_all(data=X)
        print out
        feature2 = np.float64(out['googlenetOutput'])
        feature2=np.reshape(feature2,(test_num,5120))
        #np.savetxt('feature2.txt', feature2, delimiter=',')

        #提取标签    
        labels=ModelTest.read_labels(filelist_label)
        assert(len(labels)==test_num)
        #计算每个特征之间的距离
        mt=pw.pairwise_distances(feature1, feature2, metric=metric)
        predicts=np.empty((test_num,))
        for i in range(test_num):
              predicts[i]=mt[i][i]
        # 距离需要归一化到0--1,与标签0-1匹配
        for i in range(test_num):
                predicts[i]=(predicts[i]-np.min(predicts))/(np.max(predicts)-np.min(predicts))
        accuracy=ModelTest.calculate_accuracy(predicts,labels,test_num)
        print str(accuracy)
        fpaccu=open(self.accuracy,'w')
        fpaccu.write(str(accuracy))
        fpaccu.close()

        np.savetxt(self.predict,predicts)           
        fpr, tpr, thresholds=sklearn.metrics.roc_curve(labels,predicts)

        np.savetxt(open('/home/tianluchao/ProgramFiles/caffe/examples/lfwEvaluation/result_5120dim/'+'thresholds.txt','w'),thresholds)    
        
        ModelTest.draw_roc_curve(fpr,tpr,title=metric,save_name=self.roc)

def demo_test(num,itera):
    prj='GoogLeNet'
    home='/home/tianluchao/ProgramFiles'
    caffepath=home+'/caffe/'
    prjpath=home+'/caffe/examples/lfwEvaluation/'
    datapath=home+'/caffe/data/deepID/webface/' 
    types=1
    pairs=home+'/caffe/examples/lfwEvaluation/pairs.txt'
    lfwpath='/home/tianluchao/Documents/VIPLFaceNet/aligned_lfw'

    test=ModelTest(prj,caffepath,prjpath,datapath,num,types,pairs,itera,lfwpath)
    
    #test.split_pairs()
    test.evaluate(metric='cosine')
        
if __name__=='__main__':
    num=6000#人数 我觉得应该理解为要比对的人数对，LFW的 pairs.txt 里应为6000
    itera=189090#所选模型的迭代次数
    demo_test(num,itera)
