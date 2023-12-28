import cv2
import numpy as np
import pyautogui

# 获取屏幕分辨率
screen_width, screen_height = pyautogui.size()

# 设置视频捕获窗口的大小，可以根据需要进行调整
capture_width, capture_height = 1960, 1080

# 创建视频捕获对象
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('screen_capture.avi', fourcc, 20.0, (capture_width, capture_height))

while True:
    # 获取屏幕截图
    screenshot = pyautogui.screenshot()

    # 将截图转换为OpenCV图像
    frame = np.array(screenshot)
    # 转换色彩
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 调整捕获窗口的大小
    frame = cv2.resize(frame, (capture_width, capture_height))

    # 将屏幕截图写入视频文件
    # out.write(frame, )

    # 显示屏幕截图
    cv2.imshow('Screen Capture', frame)

    # 按'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕获对象和关闭窗口
# out.release()
cv2.destroyAllWindows()