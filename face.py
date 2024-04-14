# -*- coding:utf-8 -*-
"""
作者：知行合一
日期：2019年 04月 20日 15:36
文件名：face.py
地点：changsha
"""
'''
视频流检测人脸
1、构造HAAR人脸检测器(hog,cnn,mtcnn)
2、获取视频流
3、检测每一帧画面
4、画人脸框并显示
'''

# 导入包
import cv2
import matplotlib.pyplot as plt
import numpy as np
import dlib

'第一种'
# # 构造Haar检测器
# haar_face_detect = cv2.CascadeClassifier('./face_detection/cascades/haarcascade_frontalface_default.xml')
#
#
# cap = cv2.VideoCapture(0)
# while True:
#     ret,frame = cap.read()
#     # 镜像
#     frame = cv2.flip(frame,1)
#
#     # 转为灰度图
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # 检测人脸
#     # haar_detection = haar_face_detect.detectMultiScale(frame_gray)
#     haar_detection = haar_face_detect.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=7)
#
#     # 解析检测结果
#     for (x, y, w, h) in haar_detection:
#         # print(x,y,w,h)
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
#     # 显示画面
#     cv2.imshow('face',frame)
#     # 退出条件
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()


'第二种'
# # 构造hog人脸检测器
# hog_face_detector = dlib.get_frontal_face_detector()
#
# cap = cv2.VideoCapture(0)
# while True:
#     ret,frame = cap.read()
#     # 镜像
#     frame = cv2.flip(frame,1)
#     # 检测人脸
#     # scale 类似haar的scaleFactor方法
#     detections = hog_face_detector(frame, 1)
#
#     for face in detections:
#         x = face.left()
#         y = face.top()
#         r = face.right()
#         b = face.bottom()
#         cv2.rectangle(frame, (x, y), (r, b), (0, 255, 0), 5)
#
#     # 显示画面
#     cv2.imshow('face',frame)
#     # 退出条件
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

# '第三种'
# 'ssd'
# # 加载模型
# face_detector = cv2.dnn.readNetFromCaffe('./face_detection/weights/deploy.prototxt.txt','./face_detection/weights/res10_300x300_ssd_iter_140000.caffemodel')
#
#
# cap = cv2.VideoCapture(0)
# while True:
#     ret,frame = cap.read()
#     # 镜像
#     frame = cv2.flip(frame,1)
#     # 记录原图尺寸
#     img_width = frame.shape[1]
#     img_height = frame.shape[0]
#     # 缩放至模型输入尺寸
#     img_resize = cv2.resize(frame, (500, 300))
#     # 将图像转为二进制
#     img_blob = cv2.dnn.blobFromImage(img_resize, 1.0, (500, 300), (100.0, 177.0, 123.0))
#     # 输入
#     face_detector.setInput(img_blob)
#     # 推理
#     detections = face_detector.forward()
#     # 查看检测人脸数量
#     num_of_detections = detections.shape[2]
#     # 复制原图
#     img_copy = frame.copy()
#     for index in range(num_of_detections):
#         # 置信度
#         detection_confidence = detections[0, 0, index, 2]
#         # 挑选置信度
#         if detection_confidence > 0.75:
#             # 位置
#             location = detections[0, 0, index, 3:7] * np.array([img_width, img_height, img_width, img_height])
#             # 打印
#             print(detection_confidence * 100)
#             lx, ly, rx, ry = location.astype('int')
#             # 绘制
#             cv2.rectangle(img_copy, (lx, ly), (rx, ry), (0, 255, 0), 5)
#
#
#     # 显示画面
#     cv2.imshow('face',img_copy)
#     # 退出条件
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

'第四种cnn'
# 构造CNN人脸检测器
cnn_face_detector  = dlib.cnn_face_detection_model_v1('./face_detection/weights/mmod_human_face_detector.dat')# 预训练文件

cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    # 镜像
    frame = cv2.flip(frame, 1)
    # 检测人脸
    detections = cnn_face_detector(frame, 1)
    # 解析矩阵结果
    for face in detections:
        x = face.rect.left()
        y = face.rect.top()
        r = face.rect.right()
        b = face.rect.bottom()
        # 置信度
        c = face.confidence
        print(c)
        cv2.rectangle(frame,(x, y),(r, b),(0, 255, 0),5)

    # 显示画面
    cv2.imshow('face', frame)
    # 退出条件
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# '第五种mtcnn'
# # 导入MTCNN
# from mtcnn.mtcnn import MTCNN
# # 加载模型
# face_detector = MTCNN()
#
#
# cap = cv2.VideoCapture(0)
# while True:
#     ret,frame = cap.read()
#     # 镜像
#     frame = cv2.flip(frame,1)
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     # 检测人脸
#     detections = face_detector.detect_faces(frame)
#     for face in detections:
#         (x, y, w, h) = face['box']
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
#
#     # 显示画面
#     cv2.imshow('face',frame)
#     # 退出条件
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()