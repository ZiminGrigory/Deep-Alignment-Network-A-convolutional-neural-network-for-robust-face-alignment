import dan_model
import my_dan_run_loop

import os
import sys
import glob        
import cv2
import csv

import numpy as np
import cv2
import tensorflow as tf



class VGG16Model(dan_model.Model):
    def __init__(self,num_lmark,data_format=None):
        
        img_size=112
        filter_sizes=[64,128,256,512]
        num_convs=2
        kernel_size=3

        super(VGG16Model,self).__init__(
            num_lmark=num_lmark,
            img_size=img_size,
            filter_sizes=filter_sizes,
            num_convs=num_convs,
            kernel_size=kernel_size,
            data_format=data_format
        )

# def my_input_fn(imgPath,img_size,num_lmark):
#     def _get_oneImage():
#         for i in range(1):
#             img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
#             frame = cv2.resize(img,(img_size,img_size)).astype(np.float32)
#             yield (frame,np.zeros([num_lmark,2],np.float32))

#     def input_img_fn():
#         dataset = tf.data.Dataset.from_generator(_get_oneImage,(tf.float32,tf.float32),(tf.TensorShape([img_size,img_size]),tf.TensorShape([num_lmark,2])))
#         return dataset

#     return input_img_fn


def storeFramesWithMarks(generator):
    # data_dir='./data_dir'
    data_dir=""
    model_dir='dan_tf/DAN_V2/model_dir3'
    data_format='channels_last'
    batch_size=64

    mode = tf.estimator.ModeKeys.PREDICT

    flags = {"data_dir": data_dir, "model_dir": model_dir, "data_format":data_format,
        "dan_stage": 2, "num_lmark": 68, "batch_size": batch_size, "mode":mode}

    def vgg16_model_fn(features, labels, mode, params):
        return my_dan_run_loop.dan_model_fn(features=features,
                            groundtruth=labels,
                            mode=mode,
                            stage=2,
                            num_lmark=68,
                            model_class=VGG16Model,
                            mean_shape=None,
                            imgs_mean=None,
                            imgs_std=None,
                            data_format='channels_last')

    estimator = my_dan_run_loop.dan_main(flags, vgg16_model_fn)


    predict_results = estimator.predict(generator)
    j = 0
    for x in predict_results:
        j+=1
        landmark = x['s2_ret']
        img = x['img']
        outImg = np.zeros([112,112,3])
        outImg[:,:,0] = np.squeeze(img, axis=2)
        outImg[:,:,1] = np.squeeze(img, axis=2)
        outImg[:,:,2] = np.squeeze(img, axis=2)
        print (img.shape)
        for x,y in landmark:
            outImg[111 if int(y) > 110  else int(y),111  if int(x) > 110 else int(x),:] = [0,0, 255]

        cv2.imwrite("/home/greg/dev/csc_practice_autumn2019/300W_HELEN/TST/{:05d}.png".format(j), outImg)

def showFramesWithMarksAndPredictRes(generator, predictFunc):
    # data_dir='./data_dir'
    data_dir=""
    model_dir='dan_tf/DAN_V2/model_dir3'
    data_format='channels_last'
    batch_size=64

    mode = tf.estimator.ModeKeys.PREDICT

    flags = {"data_dir": data_dir, "model_dir": model_dir, "data_format":data_format,
        "dan_stage": 2, "num_lmark": 68, "batch_size": batch_size, "mode":mode}

    def vgg16_model_fn(features, labels, mode, params):
        return my_dan_run_loop.dan_model_fn(features=features,
                            groundtruth=labels,
                            mode=mode,
                            stage=2,
                            num_lmark=68,
                            model_class=VGG16Model,
                            mean_shape=None,
                            imgs_mean=None,
                            imgs_std=None,
                            data_format='channels_last')

    estimator = my_dan_run_loop.dan_main(flags, vgg16_model_fn)


    predict_results = estimator.predict(generator)
    for x in predict_results:
        landmark = x['s2_ret']
        img = x['img']
        outImg = np.zeros([140,112,3])
        outImg[:112,:,0] = np.squeeze(img, axis=2)
        outImg[:112,:,1] = np.squeeze(img, axis=2)
        outImg[:112,:,2] = np.squeeze(img, axis=2)
        for x,y in landmark:
            outImg[111 if int(y) > 110  else int(y),111  if int(x) > 110 else int(x),:] = [0,0, 255]

        eyes = landmark[36:48]
        eyes2 = [item for sublist in eyes for item in sublist]
        prediction = predictFunc(eyes2)
        if prediction == 1:
            outImg[112:,:,1] = 127
        else:
            outImg[112:,:,0] = 127

        cv2.imshow('video with bboxes', cv2.resize(outImg,(300,300)).astype(np.uint8))
        if cv2.waitKey(33) == 27: 
            break  # esc to quit

def storeEyesData(generator, fileName):
    # data_dir='./data_dir'
    data_dir=""
    model_dir='dan_tf/DAN_V2/model_dir3'
    data_format='channels_last'
    batch_size=64
    mode = tf.estimator.ModeKeys.PREDICT

    flags = {"data_dir":data_dir, "model_dir": model_dir, "data_format":data_format,
        "dan_stage": 2, "num_lmark": 68, "batch_size": batch_size, "mode":mode}

    def vgg16_model_fn(features, labels, mode, params):
        return my_dan_run_loop.dan_model_fn(features=features,
                            groundtruth=labels,
                            mode=mode,
                            stage=2,
                            num_lmark=68,
                            model_class=VGG16Model,
                            mean_shape=None,
                            imgs_mean=None,
                            imgs_std=None,
                            data_format='channels_last')

    estimator = my_dan_run_loop.dan_main(flags, vgg16_model_fn)

    with open(fileName, "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')

        predict_results = estimator.predict(generator)
        for x in predict_results:
            landmark = x['s2_ret']
            eyes = landmark[36:48]
            eyes2 = [item for sublist in eyes for item in sublist]
            writer.writerow(eyes2)

def storeEyesDataAndImages(generator, folderForResults):
    # data_dir='./data_dir'
    data_dir=""
    model_dir='dan_tf/DAN_V2/model_dir3'
    data_format='channels_last'
    batch_size=64
    mode = tf.estimator.ModeKeys.PREDICT

    flags = {"data_dir":data_dir, "model_dir": model_dir, "data_format":data_format,
        "dan_stage": 2, "num_lmark": 68, "batch_size": batch_size, "mode":mode}

    def vgg16_model_fn(features, labels, mode, params):
        return my_dan_run_loop.dan_model_fn(features=features,
                            groundtruth=labels,
                            mode=mode,
                            stage=2,
                            num_lmark=68,
                            model_class=VGG16Model,
                            mean_shape=None,
                            imgs_mean=None,
                            imgs_std=None,
                            data_format='channels_last')

    estimator = my_dan_run_loop.dan_main(flags, vgg16_model_fn)



    resultsCSV = folderForResults + 'features.csv'
    with open(resultsCSV, "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')

        predict_results = estimator.predict(generator)
        j = 0
        for x in predict_results:
            j = j+1
            landmark = x['s2_ret']
            eyes = landmark[36:48]
            eyes2 = [item for sublist in eyes for item in sublist]
            writer.writerow(eyes2)
            predict_results = estimator.predict(generator)

            #images
            img = x['img']
            outImg = np.zeros([112,112,3])
            outImg[:,:,0] = np.squeeze(img, axis=2)
            outImg[:,:,1] = np.squeeze(img, axis=2)
            outImg[:,:,2] = np.squeeze(img, axis=2)
            for x,y in landmark:
                outImg[111 if int(y) > 110  else int(y),111  if int(x) > 110 else int(x),:] = [0,0, 255]

            cv2.imwrite("{}{:05d}.png".format(folderForResults, j), outImg)