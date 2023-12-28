import cv2
import mediapipe as mp
import numpy as np

from cv2 import getTickCount, getTickFrequency
import math
import argparse
import autopy
from ultralytics import YOLO
from ultralytics import FastSAM

# 加载 YOLOv8 模型
model_det = YOLO("yolov8l.pt")
model_pose = YOLO("yolov8l-pose.pt")
model_seg = YOLO("yolov8l-seg.pt")
# 加载 FastSAM 模型
model_sam = FastSAM('FastSAM-x.pt')

# 全局变量 获取电脑屏幕宽高
wScr, hScr = autopy.screen.size()

# 全局变量 画笔拿起flag==0 画笔落下flag==1
flag = [0, 0]

"""毕业设计

学校：桂林理工大学
班级：通信工程20-3班
作者姓名：王中天
学号：3202052052130
指导老师：李新

1. 代码完全由作者构建，仅借鉴mediapipe以及yolov8官方网站示例代码
2. 任何使用方式请遵循开源开发者使用协议
"""


def get_args():
    parser = argparse.ArgumentParser(description='桂林理工大学 通信工程20-3班 王中天 学号3202052052130')

    parser.add_argument("--device", type=int, default=0, help='使用摄像头的编号，默认为0')
    parser.add_argument("--huaquan", type=bool, default=True, help='是否画出手部的bbox')
    parser.add_argument("--save", type=str, default='mp4', help='是否保存视频，可选择mp4和avi两种格式')
    parser.add_argument("--mark_finger", type=bool, default=True, help='是否显示每个手指的数字')
    parser.add_argument("--smoothening", type=int, default=3, help='消除虚拟键盘抖动，最小为1')
    parser.add_argument("--width", type=int, default=1960, help='检测图窗宽度')  # 1960
    parser.add_argument("--height", type=int, default=1080, help='检测图窗高度')  # 1080
    parser.add_argument("--niehe", type=int, default=0.15,
                        help='手指捏合判定为画笔落下的距离，太小则不流畅 可能断开，太大则不精准')

    parser.add_argument("--min_detection_confidence", type=float, default=0.5, help='手的目标检测置信度')
    parser.add_argument("--min_tracking_confidence", type=int, default=0.5, help='手的追踪检测置信度')

    args = parser.parse_args()
    return args


# 打开摄像头进行实时视频流的捕获，并实时显示出来
def main():
    # 设置摄像头参数
    args = get_args()
    save_video = args.save
    cap_width = args.width
    cap_height = args.height
    cap_device = args.device
    plocX, plocY = 0, 0

    # 调用摄像头
    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    if save_video:
        if save_video == 'avi':
            # 保存为较为稳定的avi格式
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('output.avi', fourcc, 20.0, (cap_width, cap_height))
        elif save_video == 'mp4':
            # 保存为较为常用的mp4格式
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (cap_width, cap_height))

    # 主循环
    while True:
        # 记录时间计算fps
        loop_start = getTickCount()
        # 读取视频帧
        ret, frame = cap.read()

        # flip对图像进行翻转，0为竖直 1为水平
        img = cv2.flip(frame, 1)
        frame = img
        # 调用detect_and_draw_hand_landmarks函数 返回每帧的21个关键点信息(x,y,z)
        # hand_landmarks, lmlist, bbox = detect_and_draw_hand_landmarks(frame)

        # 调用draw_palm_points函数 返回绘制的图形坐标点信息
        # huahua = draw_palm_points(frame, hand_landmarks, flag)

        # results_det = model_det.predict(source=frame)  # 对当前帧进行目标检测并显示结果
        # frame = results_det[0].plot()
        # print(results_det[0].verbose())

        # results_pose = model_pose.predict(source=frame) # 姿态估计
        # frame = results_pose[0].plot(boxes=False)
        # print(results_pose[0].verbose())

        # 在图像上运行推断
        results_seg = model_sam(source=frame, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
        frame = results_seg[0].plot(boxes=False)
        # results_seg = model_seg.predict(source=frame)  # 语义分割
        # frame = results_seg[0].plot(boxes=False)

        # 有啥东西
        # print(results_seg[0].verbose())

        # 虚拟鼠标
        # frame, plocX, plocY = mouse(lmlist, frame, plocX, plocY)

        # 计算并显示fps数
        frame = fps(loop_start, frame)

        # 保存视频
        frame = cv2.resize(frame, (cap_width, cap_height))
        if save_video:
            out.write(frame)
        # 在窗口frame中显示图像
        cv2.imshow('frame', frame)

        # 键控设置
        # 按下q或者Esc(ascii值27)则退出
        if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == 27:
            break
        # 按下w则保存当前一帧的图片

        if cv2.waitKey(1) == 119: # 键入'w'
            cv2.imwrite(f'./seg/seg.jpg', img)
            for index, mask in enumerate(results_seg[0].masks.data):
                cv2.imwrite(f'./seg/seg{index}.jpg', mask.cpu().numpy() * 255)
            print("保存OK了")

    cap.release()
    if save_video:
        out.release()
    cv2.destroyAllwindows()


# 操作虚拟鼠标
def mouse(lmlist, frame, plocX, plocY):
    # 设置摄像头参数
    args = get_args()
    cap_width = args.width
    cap_height = args.height
    smoothening = args.smoothening
    # 判断食指和中指是否伸出
    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:]
        fingers = fingersUp(lmlist)
        print(fingers)

        # 若只有食指伸出 则进入移动模式
        if fingers[1] == 1 and fingers[2] == 0:
            # 4. 坐标转换： 将食指在窗口坐标转换为鼠标在桌面的坐标
            # 鼠标坐标
            x3 = np.interp(x1, (0, cap_width), (0, wScr))
            y3 = np.interp(y1, (0, cap_height), (0, hScr))
            # 反转x轴
            x3 = wScr - x3 * 2

            # 平滑处理
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 控制鼠标
            autopy.mouse.move(min(wScr - clocX, wScr - 1), min(clocY * 2.4, hScr - 1))
            # 突出显示食指
            cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # 5. 若是食指和中指都伸出 则检测指头距离 距离够短则对应鼠标点击
        if fingers[1] and fingers[2]:
            length, frame, pointInfo = findDistance(8, 12, frame, lmlist)
            if length < 58:
                cv2.circle(frame, (pointInfo[4], pointInfo[5]),
                           15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()
    return frame, plocX, plocY


# 测量手部区域上的关键点和边缘，并呈现关键点
def detect_and_draw_hand_landmarks(frame):
    # 获取参数
    args = get_args()
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    mark = args.mark_finger
    huaquan = args.huaquan
    # 运用手的检测模型，并定义为mp_hands
    mp_hands = mp.solutions.hands
    lmlist, bbox, xlist, ylist = [], [], [], []

    # 使用.Hands方法，并命名为hands
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=2, model_complexity=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # 读取到的BGR图片转换成RGB
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 使用hands.process对该图片进行检测
    results = hands.process(rgb_image)

    hand_landmarks = None
    # 定义画手的关键点的函数
    mp_drawing = mp.solutions.drawing_utils
    # 定义画手的样式函数
    # 点的
    hand_landmarks_style = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
    # 线的
    hand_con_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3)

    # 使用视窗大小
    imgheigt = frame.shape[0]
    imgwidth = frame.shape[1]
    # 如果有检测到手
    if results.multi_hand_landmarks:
        # 对于每只手的21个关键点，将他们画出来mp_hands.HAND_CONNECTIONS表示关键点之间需要连接
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, )
            # hand_landmarks_style,hand_con_style)

            cz0 = hand_landmarks.landmark[0].z
            # 对获取的hand_marks做处理
            for i, lm in enumerate(hand_landmarks.landmark):
                # 获取3D坐标
                cx, cy = int(lm.x * imgwidth), int(lm.y * imgheigt)
                cz = lm.z
                depth_z = cz0 - cz

                # 标记好每一点的数字 大小为0.4 粗度为2
                if mark:
                    cv2.putText(frame, str(i), (cx - 25, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

                # 用圆的半径反映深度大小
                radius = max(int(6 * (1 + depth_z * 5)), 0)

                if i == 0:  # 手腕
                    cv2.circle(frame, (cx, cy), radius, (0, 0, 255), -1)
                # if i == 8:  # 食指指尖
                #         cv2.circle(frame, (cx, cy), radius, (193, 182, 255), -1)
                if i in [1, 5, 9, 13, 17]:  # 指根
                    cv2.circle(frame, (cx, cy), radius, (16, 144, 247), -1)
                if i in [2, 6, 10, 14, 18]:  # 第一指节
                    cv2.circle(frame, (cx, cy), radius, (1, 240, 255), -1)
                if i in [3, 7, 11, 15, 19]:  # 第二指节
                    cv2.circle(frame, (cx, cy), radius, (140, 47, 240), -1)
                if i in [4, 8, 12, 16, 20]:  # 指尖（(除食指指尖)
                    cv2.circle(frame, (cx, cy), radius, (223, 155, 60), -1)

                # 输出每一个关键点的位置
                lmlist.append([i, cx, cy])
                xlist.append(cx)
                ylist.append(cy)

            xmin, xmax = min(xlist), max(xlist)
            ymin, ymax = min(ylist), max(ylist)
            bbox = xmin, ymin, xmax, ymax
            # print(bbox)
            # print("一次")
            if huaquan:
                cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

    hands.close()
    return hand_landmarks, lmlist, bbox


# 描绘手指捏合时的轨迹点跟踪与显示
def draw_palm_points(frame, hand_landmarks, flag, history_palm_points=[]):
    # 获取参数
    args = get_args()
    niehe = args.niehe

    # 如果没有手，就把画的线给杀了
    if not hand_landmarks:
        history_palm_points.clear()
        flag.clear()
        return
    # thumb_index和indexfinger_index为捏合的两个手指
    # 4 8 12 16 20分别为大拇指 食指 中指 无名指 小拇指
    thumb_index = 4
    indexfinger_index = 8
    # 取关键点坐标
    indexfinger_point = hand_landmarks.landmark[indexfinger_index]
    thumb_point = hand_landmarks.landmark[thumb_index]
    # 计算关键点距离
    finger_distance = np.sqrt(
        np.square(thumb_point.x - indexfinger_point.x)
        + np.square(thumb_point.y - indexfinger_point.y))
    # 如果距离小于0.1 就将当前的视窗位置append到history里
    if finger_distance < niehe:
        # 画笔落下
        flag.append(1)
        palm_point = [
            int((thumb_point.x + indexfinger_point.x) * frame.shape[1] // 2),
            int((thumb_point.y + indexfinger_point.y) * frame.shape[0] // 2), ]
        history_palm_points.append(palm_point)
    else:
        # 画笔拿起
        flag[-2:] = [0, 0]

    if len(history_palm_points) < 1:
        pass
    else:
        for i in range(1, len(history_palm_points)):
            # 若画笔落下则开画
            if flag[i]:
                cv2.line(frame, tuple(history_palm_points[i - 1][:2]), tuple(history_palm_points[i][:2]), (0, 255, 255),
                         5)
                # print(history_palm_points)
    if not flag[-1]:
        return history_palm_points


# 返回五根手指的伸出和握回情况
def fingersUp(lmList):
    fingers = []
    tipIds = [4, 8, 12, 16, 20]
    # 大拇指
    if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
        fingers.append(1)
    else:
        fingers.append(0)

    # 其余手指
    for id in range(1, 5):
        if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)

    # totalFingers = fingers.count(1)
    return fingers


# 计算两点距离
def findDistance(p1, p2, img, lmList, draw=True, r=15, t=3):
    x1, y1 = lmList[p1][1:]
    x2, y2 = lmList[p2][1:]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    if draw:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
        cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

    return length, img, [x1, y1, x2, y2, cx, cy]


# 计算并给frame标上fps
def fps(loop_start, frame):
    loop_time = getTickCount() - loop_start
    total_time = loop_time / (getTickFrequency())
    FPS = int(1 / total_time)
    # 在图像左上角添加FPS文本
    fps_text = f"FPS: {FPS:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 0, 255)  # 红色
    text_position = (10, 30)  # 左上角位置
    # 计算并显示fps数

    cv2.putText(frame, fps_text, text_position, font, font_scale, text_color, font_thickness)
    return frame


if __name__ == '__main__':
    main()
