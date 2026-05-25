#!/usr/bin/env python3
"""
人脸替换摄像头 - 摄像头视频人脸替换
使用方法: python face_swap.py <人脸图片路径>
"""

import cv2
import numpy as np
import sys
import os


class 人脸替换器:
    def __init__(self, 覆盖图片路径: str = None):
        # 加载哈尔级联分类器用于人脸检测
        级联路径 = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.人脸级联 = cv2.CascadeClassifier(级联路径)
        
        # 加载要覆盖的图片
        self.覆盖图片 = self.加载覆盖图片(覆盖图片路径)
        
    def 加载覆盖图片(self, 路径: str) -> np.ndarray:
        """加载要覆盖在人脸上的图片"""
        if 路径 is None or not os.path.exists(路径):
            if 路径:
                print(f"⚠️  图片未找到: {路径}")
            else:
                print("⚠️  未指定图片")
            print("使用默认笑脸图片")
            # 创建简单的笑脸图片
            图片 = np.zeros((200, 200, 4), dtype=np.uint8)
            cv2.circle(图片, (100, 100), 90, (255, 255, 0, 255), -1)  # 黄色圆圈
            cv2.circle(图片, (65, 70), 15, (0, 0, 0, 255), -1)  # 左眼
            cv2.circle(图片, (135, 70), 15, (0, 0, 0, 255), -1)  # 右眼
            cv2.ellipse(图片, (100, 110), (40, 30), 0, 0, 180, (0, 0, 0, 255), 5)  # 笑容
            return 图片
        
        图片 = cv2.imread(路径, cv2.IMREAD_UNCHANGED)
        if 图片 is None:
            raise ValueError(f"无法加载图片: {路径}")
        
        # 如果没有透明通道,添加它
        if 图片.shape[2] == 3:
            图片 = cv2.cvtColor(图片, cv2.COLOR_BGR2BGRA)
        
        return 图片
    
    def 调整覆盖图片大小(self, 宽度: int, 高度: int) -> np.ndarray:
        """调整覆盖图片的大小"""
        return cv2.resize(self.覆盖图片, (宽度, 高度))
    
    def 覆盖透明图片(self, 背景: np.ndarray, 覆盖层: np.ndarray, 
                     x坐标: int, y坐标: int) -> np.ndarray:
        """将带透明通道的图片覆盖到背景上"""
        # 检查边界
        高, 宽 = 覆盖层.shape[:2]
        背景高, 背景宽 = 背景.shape[:2]
        
        if x坐标 < 0:
            覆盖层 = 覆盖层[:, -x坐标:]
            宽 = 覆盖层.shape[1]
            x坐标 = 0
        if y坐标 < 0:
            覆盖层 = 覆盖层[-y坐标:, :]
            高 = 覆盖层.shape[0]
            y坐标 = 0
        
        if x坐标 + 宽 > 背景宽:
            覆盖层 = 覆盖层[:, :背景宽 - x坐标]
            宽 = 覆盖层.shape[1]
        if y坐标 + 高 > 背景高:
            覆盖层 = 覆盖层[:背景高 - y坐标, :]
            高 = 覆盖层.shape[0]
        
        if 覆盖层.shape[0] == 0 or 覆盖层.shape[1] == 0:
            return 背景
        
        # 提取透明通道
        透明度 = 覆盖层[:, :, 3] / 255.0
        透明度 = cv2.merge([透明度, 透明度, 透明度])
        
        # 获取RGB部分
        覆盖层RGB = 覆盖层[:, :, :3]
        
        # 覆盖
        背景区域 = 背景[y坐标:y坐标+高, x坐标:x坐标+宽]
        混合 = (覆盖层RGB * 透明度 + 背景区域 * (1 - 透明度)).astype(np.uint8)
        背景[y坐标:y坐标+高, x坐标:x坐标+宽] = 混合
        
        return 背景
    
    def 处理帧(self, 帧: np.ndarray) -> np.ndarray:
        """处理帧并替换人脸"""
        灰度图 = cv2.cvtColor(帧, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        人脸列表 = self.人脸级联.detectMultiScale(
            灰度图,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        for (x, y, w, h) in 人脸列表:
            # 在人脸周围添加边距(20%)
            边距 = 0.2
            x新 = int(x - w * 边距)
            y新 = int(y - h * 边距)
            w新 = int(w * (1 + 2 * 边距))
            h新 = int(h * (1 + 2 * 边距))
            
            # 调整覆盖层大小
            覆盖层调整后 = self.调整覆盖图片大小(w新, h新)
            
            # 覆盖图片
            帧 = self.覆盖透明图片(帧, 覆盖层调整后, x新, y新)
        
        return 帧


def 主函数():
    # 检查参数
    if len(sys.argv) > 1:
        覆盖路径 = sys.argv[1]
    else:
        print("=" * 50)
        print("📸 人脸替换摄像头")
        print("=" * 50)
        print("\n使用方法: python face_swap.py <图片路径>")
        print("\n示例:")
        print("  python face_swap.py tuntunsahur.png")
        print("\n使用默认笑脸启动中...")
        print("-" * 50)
        覆盖路径 = None
    
    # 初始化摄像头
    print("\n🎥 初始化摄像头...")
    摄像头 = cv2.VideoCapture(0)
    
    if not 摄像头.isOpened():
        print("❌ 错误: 无法打开摄像头")
        print("请确保摄像头已连接且未被其他应用程序使用")
        return
    
    # 设置窗口大小
    摄像头.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    摄像头.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✅ 摄像头初始化成功")
    print("\n⌨️  控制:")
    print("   Q 或 ESC - 退出")
    print("   S - 保存截图")
    print("-" * 50)
    
    # 创建人脸替换器
    try:
        替换器 = 人脸替换器(覆盖路径)
    except Exception as e:
        print(f"❌ 加载图片时出错: {e}")
        return
    
    截图计数 = 0
    
    while True:
        读取成功, 帧 = 摄像头.read()
        if not 读取成功:
            print("❌ 读取帧时出错")
            break
        
        # 镜像翻转图像以自然显示
        帧 = cv2.flip(帧, 1)
        
        # 处理帧
        try:
            帧 = 替换器.处理帧(帧)
        except Exception as e:
            pass  # 即使处理出错也继续
        
        # 显示结果
        cv2.imshow('人脸替换 - 按 Q 退出', 帧)
        
        # 按键处理
        按键 = cv2.waitKey(1) & 0xFF
        if 按键 == ord('q') or 按键 == 27:  # Q 或 ESC
            break
        elif 按键 == ord('s'):  # 保存截图
            截图计数 += 1
            文件名 = f'人脸替换截图_{截图计数}.png'
            cv2.imwrite(文件名, 帧)
            print(f"📸 截图已保存: {文件名}")
    
    # 释放资源
    摄像头.release()
    cv2.destroyAllWindows()
    print("\n👋 再见!")


if __name__ == "__main__":
    主函数()
