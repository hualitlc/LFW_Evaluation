# LFW_Evaluation
Learn more about LFW dataset and parse related files. Try to change or design a CNN to get satisfactory results on LFW dataset.


### References
0. The project [face_recognition](https://github.com/hqli/face_recognition/blob/master/) of hpli's github. Thanks again for the author's contribution.
1. [CASIA-WebFace dataset](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html).
- 10,575 subjects and 494,414 images
- Thank Zhanxiong Wang for providing this dataset. If you have interest in face recognition and face attribute prediction, you can refer to this article [Multi-task Deep Neural Network for Joint Face Recognition and Facial Attribute Prediction](https://dl.acm.org/citation.cfm?id=3078973).
2. [LFW dataset](http://vis-www.cs.umass.edu/lfw/).
3. [Caffe](http://caffe.berkeleyvision.org/).
4. The implementation of GoogLeNet in Caffe.


### Experimental steps
0. We have got these two datasets (CASIA-WebFace and LFW) which have been cropped and aligned. So we did not have the image preprossing step.
1. Train model on CASIA-WebFace dataset using caffe framework. We have tried two deep models. One is GoogLeNet implemented in caffe and the other is ResNet50. Bue we found we could not make the network converged no matter what we did. We have tried to turn the BN layer on or off (use_global_status=false/true).
2. We search on the Internet and find some people said that "this dataset needs to carefully 'washed' because this dataset is a little 'dirty'". Although this dataset has so many data, but there exists a serious data imbalance. Someone have one or two pictures, but another people may have more than 200 pictures. So we only selected the people with more than 50 pictures use the script **create_filelist_over50.py** to implement this. Finally, we got 2228 people and 250392 pictures (201660 for training, 48732 for testing).
3. We train from scratch the googlenet model and excute 189090 iterations (about 90 epochs), and save the caffemodel every 5 epochs. Finally, the accuracy is about 93%. This is a rough resut because we did not try to use some tricks and attempt to use different lr_policy and some hyper-parameter related changes.
4. Test on LFW dataset.
- [1] Parse the pairs.txt provided by the official website and get lfw_left.txt, lfw_right.txt and label.txt. We totally have 6000 test pairs (3000 positive pairs and 3000 negative pairs). Each test pair, the 'left' image and its path is stored in lfw_left.txt, the 'right' image and its path is stored in lfw_right.txt, and the label ('0' represents match and '1' represents mismatch) is stored in label.txt. Use the script **generateTestDataList.py**.
- [2] Use the googlenet model *.caffemodel* and *deploy.prototxt* and the script **testModelOnLFW.py** to get the feature dimension and calculate the similarity using cosine distance. Finally we get the accuracy, predicts.txt and ROC curves.


### Results
We change the name of the *loss3/classifier* to *googlenetOutput* (if we don't excute this operation, then we use the weights learned from .caffemodel, there will have a 'number_output mismatch' error, because we train from scratch the googlenet model, the number_output is 2228, so if we want to use this trained model to extract X dimensions features, we must change the layer name and parameter 'number_output') and change the *num_output* into different dimensions, such as 160, 1024, 2048, 4096, 5120..., to extract features of different dimensions.
```
layer {
  name: "googlenetOutput"#loss3/classifier
  type: "InnerProduct"
  bottom: "pool5/7x7_s1"
  top: "googlenetOutput"#loss3/classifier
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
    num_output: 5120   # extract features of different dimensions
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
```
The accuracy of different feature dimensions is shown below:
*- 160  feature dimension: 0.920333333333\* 
 - 1024 feature dimension: 0.945\*
 - 2048 feature dimension: 0.949666666667\*
 - 4096 feature dimension: 0.951
 - 5120 feature dimension: 0.948833333333*

\*The foleder *result_160dim, result_1024dim and result_2048dim* did not have *thresholds.txt*, this file have been overwritten because of the path setting problem.

The ROC curves of different settings are almost the same, so we only display the result of 4096 feature dimensions.
![image](https://github.com/hualitlc/LFW_Evaluation/blob/master/result_4096dim/GoogLeNet_6000_189090_roc.png)

### Related discussions
0. The theory of using multi-gpu to train your network in Caffe. Ref: https://github.com/BVLC/caffe/blob/master/docs/multigpu.md. 
- We found that when the network in 'test' phase, the memory used in 'train' phase will not be released. So you should choose a appropriate batch size. 
- If use batch size 64 on one gpu, the memory used is A1; then your train stage's batch size is 64 on four gpu, the memory is also used A1.
1. If you met this problem
```
/usr/local/lib/python2.7/dist-packages/matplotlib/axes/\_axes.py:545: UserWarning: No labelled objects found. Use label='...' kwarg on is.
  warnings.warn("No labelled objects found.
```
You have to provide a label=.. keyword in the plot function for each line you want to plot, as matplotlib does not automatically detect names from a numpy structured array. Ref: https://stackoverflow.com/questions/16488037/how-to-use-names-when-importing-csv-data-into-matplotlib.


### Welcome
Welcome to discuss questions with me. Email: *hualitlc@163.com*



