import os
import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
from pynput import keyboard
from PIL import ImageGrab, Image  # 添加 Image 导入
import numpy as np
from datetime import datetime

class FaceDetector:
    def __init__(self, model_path=None):
        """初始化人脸检测器"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        if model_path is None:
            self.model = YOLO('yolov11s-face.pt')
        else:
            self.model = YOLO(model_path)
        
        self.model.to(self.device)

    def detect_and_save_faces(self, img, output_dir, conf_threshold=0.5, scale_factor=2.2):
        """检测并保存人脸"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保持原始图像格式，不进行颜色空间转换
        # YOLO模型可以处理RGB格式的输入
        results = self.model(img, conf=conf_threshold)
        
        saved_paths = []
        for i, det in enumerate(results[0].boxes.xyxy):
            x1, y1, x2, y2 = map(int, det)
            
            # 计算中心点和尺寸
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1
            
            # 使用最大边长作为基准并应用缩放因子
            size = int(max(width, height) * scale_factor)
            
            # 计算扩展后的边界
            new_x1 = max(0, center_x - size // 2)
            new_y1 = max(0, center_y - size // 2)
            new_x2 = min(img.shape[1], new_x1 + size)
            new_y2 = min(img.shape[0], new_y1 + size)
            
            # 保持正方形
            if new_x2 - new_x1 != new_y2 - new_y1:
                size = min(new_x2 - new_x1, new_y2 - new_y1)
                new_x2 = new_x1 + size
                new_y2 = new_y1 + size
            
            # 裁剪人脸
            face = img[new_y1:new_y2, new_x1:new_x2]
            
            # 转换为PIL Image进行resize，保持颜色正确
            face_pil = Image.fromarray(face)
            face_pil = face_pil.resize((600, 600), Image.Resampling.LANCZOS)
            
            # 使用时间戳作为文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f'face_{timestamp}_{i}.png')
            
            # 直接保存PIL图像，避免OpenCV的颜色转换
            face_pil.save(output_path, 'PNG', quality=99)
            saved_paths.append(output_path)
            
        return saved_paths

class ScreenFaceCapture:
    def __init__(self, output_dir):
        self.detector = FaceDetector()
        self.output_dir = output_dir
        self.is_running = True
        self.alt_pressed = False

    def capture_screen(self):
        """捕获当前屏幕并返回numpy数组"""
        # 使用PIL的ImageGrab，返回的是RGB格式
        screenshot = ImageGrab.grab()
        # 转换为numpy数组，保持RGB格式
        return np.array(screenshot)

    def process_screen(self):
        """处理当前屏幕内容"""
        print("Capturing screen...")
        screen = self.capture_screen()
        faces = self.detector.detect_and_save_faces(screen, self.output_dir)
        if faces:
            print(f"Found and saved {len(faces)} faces")
            print(f"Faces saved to: {self.output_dir}")
        else:
            print("No faces detected")

    def on_press(self, key):
        """按键按下时的回调函数"""
        try:
            # 检查是否按下Alt键
            if key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
                self.alt_pressed = True
            # 如果按下Q键且Alt被按下
            elif self.alt_pressed and key.char == 'q':
                self.process_screen()
            # 检查是否按下Esc键
            elif key == keyboard.Key.esc:
                self.is_running = False
                print("Application terminated")
                return False
        except AttributeError:
            pass

    def on_release(self, key):
        """按键释放时的回调函数"""
        if key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
            self.alt_pressed = False

    def run(self):
        """运行主循环，监听快捷键"""
        print("Screen Face Capture is running...")
        print("Press Alt+Q to capture faces")
        print("Press Esc to exit")
        print(f"Faces will be saved to: {self.output_dir}")

        # 创建监听器
        with keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release) as listener:
            listener.join()

def main():
    # 设置输出目录
    output_dir = os.path.join(os.path.expanduser("~"), "Desktop", "captured_faces")
    
    # 创建并运行应用
    app = ScreenFaceCapture(output_dir)
    try:
        app.run()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
