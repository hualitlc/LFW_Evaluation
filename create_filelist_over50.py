#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os

def GetFileList(dir,filename,filename2):	
	subDir=os.listdir(dir)
	subDir.sort()
	class_id=0
	f = open(filename,'w')
	f2 = open(filename2,'w')
	for s in subDir:
		number_id=0
		#print number_id
		path=os.path.join(dir,s)
		#print path
		number=len(os.listdir(path))
		#print number
		if number>=50:
			for file in os.listdir(path):
				if number_id<=number*0.8:
					f.write('%s %s\n' %(os.path.join(dir, s, file), class_id))
					number_id+=1
				else:
					f2.write('%s %s\n' %(os.path.join(dir, s, file), class_id))
					#number_id+=1
			class_id=class_id+1;
	
	f.flush()
	f.close()
	f2.flush()
	f2.close()
	
GetFileList('/home/tianluchao/ProgramFiles/caffe/data/VIPLFaceNet/Aligned-CASIA-WebFace-jpg', \
'/home/tianluchao/ProgramFiles/caffe/examples/CASIA-WebFace-LFW/labelTrain_over50.txt', \
'/home/tianluchao/ProgramFiles/caffe/examples/CASIA-WebFace-LFW/labelVal_over50.txt')

		
