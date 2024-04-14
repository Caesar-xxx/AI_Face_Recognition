# -*- coding:utf-8 -*-
"""
作者：知行合一
日期：2020年 04月 13日 17:06
文件名：attendance.py
地点：changsha
"""

"""
人脸考勤系统
人脸注册：将人脸特征存进CSV
人脸识别：将检测的人脸特征与数据库CSV中的人脸特征作比较，如果比中，把考勤记录写入attendance.CSV
"""

# 导入包
import cv2
import matplotlib.pyplot as plt
import numpy as np
import dlib
import time
import csv

# 人脸注册的方法
def faceRegister(label_id=1,name='jason',count=3,interval=3):
    """
    :param label_id: 人脸id
    :param name: 人脸姓名
    :param count: 采集数量
    :param interval: 采集间隔时间
    :return:
    """
    # 检测人脸，获取68个关键点
    # 构造hog检测器

    cap = cv2.VideoCapture(0)

    # 获取人脸长宽
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 构造人脸检测器
    hog_face_detector = dlib.get_frontal_face_detector()
    # 关键点检测器
    shape_detector = dlib.shape_predictor('./weights/shape_predictor_68_face_landmarks.dat')
    # 特征描述符
    face_descriptor_extractor = dlib.face_recognition_model_v1('./weights/dlib_face_recognition_resnet_model_v1.dat')

    # 开始时间
    star_time = time.time()
    # 执行次数
    collect_count = 0

    # 构造CSV writer
    f  = open('./data/feature1.csv','a',newline="")
    csv_writer = csv.writer(f)

    while True:
        ret, frame = cap.read()
        # 缩放
        frame = cv2.resize(frame,(width//2,height//2))
        # 镜像
        frame = cv2.flip(frame, 1)

        # 转为灰度图
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        # haar_detection = haar_face_detect.detectMultiScale(frame_gray)
        detections = hog_face_detector(frame,1)

        # 遍历人脸
        for face in detections:
            # 人脸框坐标
            l,t,r,b = face.left(),face.top(),face.right(),face.bottom()

            # 获取人脸关键点
            points = shape_detector(frame,face)
            for point in points.parts():
                cv2.circle(frame,(point.x,point.y),2,(0,255,0),-1)

            # 矩形人脸框
            cv2.rectangle(frame,(l,t),(r,b),(0,255,0),2)

            # 采集
            if collect_count < count:
                # 获取当前时间
                now = time.time()
                # 判断时间限制
                if now-star_time > interval:
                    # 获取特征描述符
                    face_descriptor = face_descriptor_extractor.compute_face_descriptor(frame,points)
                    # print(face_descriptor)
                    # 转为列表
                    face_descriptor = [f for f in face_descriptor]

                    # 写入CSV文件
                    line = [label_id,name,face_descriptor]

                    csv_writer.writerow(line)

                    # print(face_extractor)
                    collect_count += 1
                    star_time = now
                    print('采集次数:{collect_count}'.format(collect_count=collect_count))

                else:
                    pass

            else:
                # 采集完毕
                print('采集完毕')
                return

        # 显示画面
        cv2.imshow('face attendance', frame)
        # 退出条件
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    f.close()
    cap.release()
    cv2.destroyAllWindows()

# 获取并组装CSV文件中特征
def getFeatureList():
    # 构造列表
    lable_list = []
    name_list = []
    feature_list = None
    with open('./data/feature1.csv','r') as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            label_id = line[0]
            name = line[1]
            lable_list.append(label_id)
            name_list.append(name)

            # str 转为 list
            face_descriptor = eval(line[2])

            face_descriptor = np.asarray(face_descriptor,dtype=np.float64)
            face_descriptor = np.reshape(face_descriptor,(1,-1))
            if feature_list is   None:
                feature_list = face_descriptor
            else:
                feature_list = np.concatenate((feature_list,face_descriptor),axis=0)
    return lable_list,name_list,feature_list


# 人脸识别的方法
# 1、实时获取视频流中人脸的特征描述符
# 2、将它与库中特征坐距离判断
# 3、根据最小值找到预测的ID,name
# 4、将考勤记录存入CSV文件,第一次识别到存入或者隔段时间存

def faceRecognizer(threshold=0.5):

    cap = cv2.VideoCapture(0)

    # 获取人脸长宽
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 构造人脸检测器
    hog_face_detector = dlib.get_frontal_face_detector()
    # 关键点检测器
    shape_detector = dlib.shape_predictor('./weights/shape_predictor_68_face_landmarks.dat')
    # 特征描述符
    face_descriptor_extractor = dlib.face_recognition_model_v1('./weights/dlib_face_recognition_resnet_model_v1.dat')

    # 读取特征
    lable_list, name_list, feature_list = getFeatureList()

    # 字典记录人脸识别记录
    recog_record = {}

    # 写入CSV
    f = open('./data/attendace1.csv','a',newline="")
    csv_wrater = csv.writer(f)

    # 帧率信息
    fps_time = time.time()

    while True:
        ret, frame = cap.read()
        # 缩放
        frame = cv2.resize(frame, (width, height)) # 按照自己摄像头画面进行调整
        # 镜像
        frame = cv2.flip(frame, 1)


        # 检测人脸

        detections = hog_face_detector(frame, 1)

        # 遍历人脸
        for face in detections:
            # 人脸框坐标
            l, t, r, b = face.left(), face.top(), face.right(), face.bottom()

            # 获取人脸关键点
            points = shape_detector(frame, face)

            # 矩形人脸框
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)

            # 获取特征描述符
            face_descriptor = face_descriptor_extractor.compute_face_descriptor(frame, points)
            # print(face_descriptor)
            # 转为列表
            face_descriptor = [f for f in face_descriptor]

            # 计算与库的距离
            face_descriptor = np.asarray(face_descriptor,dtype=np.float64)

            distances = np.linalg.norm((face_descriptor-feature_list),axis=1)

            # 获取最短距离索引
            min_index = np.argmin(distances)

            # 最短距离
            min_distance = distances[min_index]

            if min_distance < threshold:

                predict_id = lable_list[min_index]
                predict_name = name_list[min_index]


                cv2.putText(frame, predict_name + str(round(min_distance,2)), (l, b+40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 1)

                now = time.time()
                need_insert = False
                # 判断是否识别过
                if predict_name in recog_record:
                    # 存过

                    # 隔一段时间再存
                    if now - recog_record[predict_name] > 3:
                        # 超过阈值时间，在村一次
                        need_insert = True
                        recog_record[predict_name] = now
                    else:
                        # 还没到时间
                        pass
                        need_insert = False
                else:
                    # 没存过
                    recog_record[predict_name] = now
                    # 存入CSV文件
                    print(predict_name)

                if need_insert:
                    time_local = time.localtime(recog_record[predict_name])
                    # 转换格式
                    time_str = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
                    line = [predict_id,predict_name,min_distance,time_str]
                    csv_wrater.writerow(line)

                    print('{time}:写入成功:{name}'.format(name = predict_name,time = time_str))
            else:
                print('未识别到')



        # 计算帧率
        now = time.time()
        fps = 1/(now - fps_time)
        fps_time = now

        cv2.putText(frame,"FPS:"+str(round(fps,2)),(20,40),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,255,0),1)

        # 显示画面
        cv2.imshow('face attendance', frame)

        # 退出条件
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    f.close()
    cap.release()
    cv2.destroyAllWindows()


# faceRegister(label_id=1,name='jason',count=3,interval=3) # 注册人脸

faceRecognizer(threshold=0.5) # 对比人脸