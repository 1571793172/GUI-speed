import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import torch
from PIL import Image, ImageTk
from ultralytics import YOLO
import collections
from collections import deque


class ObjectDetectionApp:
    def __init__(self, root):
        self.display_width = 800
        self.display_height = 450
        self.root = root
        self.root.title("YOLO检测可视化系统")
        self.root.state('zoomed')
        self.root.geometry("1600x800")
        self.root.config(bg='#2e2e2e')
        self.model_path = None
        self.model = None
        self.image_path = None
        self.video_path = None
        self.track_history = collections.defaultdict(
            lambda: {
                'positions': deque(maxlen=10),
                'times': deque(maxlen=10)
            }
        )
        self.PIXELS_PER_METER = 50  # 默认标定值(像素尺寸pix/实际尺寸m)
        self.setup_ui()

    def setup_ui(self):
        # 顶部工具栏
        top_frame = tk.Frame(self.root, bg='#2e2e2e')
        top_frame.pack(fill="x", pady=10)

        # 标题
        self.title_label = tk.Label(top_frame, text="水下机器人-可视化检测系统",
                                    font=("Arial", 24), bg='#2e2e2e', fg='white')
        self.title_label.pack(side="left", padx=10)

        # 标定参数设置
        self.calib_frame = tk.Frame(top_frame, bg='#2e2e2e')
        self.calib_frame.pack(side="left", padx=20)

        self.calib_label = tk.Label(self.calib_frame, text="标定 (pix/m):",
                                    font=("Arial", 12), bg='#2e2e2e', fg='white')
        self.calib_label.pack(side="left")

        self.calib_entry = tk.Entry(self.calib_frame, width=8,
                                    font=("Arial", 12), bg='#444444', fg='white')
        self.calib_entry.insert(0, "50")
        self.calib_entry.pack(side="left", padx=5)

        self.update_calib_btn = tk.Button(self.calib_frame, text="更新",
                                          command=self.update_calibration,
                                          font=("Arial", 10), bg='#555555', fg='white')
        self.update_calib_btn.pack(side="left")

        # 模型路径
        self.model_label = tk.Label(top_frame, text="模型路径:",
                                    font=("Arial", 12), bg='#2e2e2e', fg='white')
        self.model_label.pack(side="left", padx=5)

        self.model_entry = tk.Entry(top_frame, width=30,
                                    font=("Arial", 12), bg='#444444', fg='white')
        self.model_entry.pack(side="left", padx=5)

        # 功能按钮
        buttons = [
            ("选择模型", self.select_model),
            ("加载媒体", self.load_media),
            ("开始检测", self.detect_objects),
            ("实时检测", self.detect_from_camera),
            ("保存结果", self.save_result)
        ]

        for text, cmd in buttons:
            btn = tk.Button(top_frame, text=text, command=cmd,
                            font=("Arial", 12), bg='#555555', fg='white')
            btn.pack(side="left", padx=5)

        # 视频显示区域
        middle_frame = tk.Frame(self.root, bg='#2e2e2e')
        middle_frame.pack(fill="both", expand=True, padx=20, pady=20)

        self.left_frame = tk.Frame(middle_frame, bg='#2e2e2e')
        self.left_frame.pack(side="left", fill="both", expand=True)

        self.right_frame = tk.Frame(middle_frame, bg='#2e2e2e')
        self.right_frame.pack(side="left", fill="both", expand=True)

        self.original_label = tk.Label(self.left_frame, bg='#2e2e2e')
        self.original_label.pack(fill="both", expand=True)

        self.detected_label = tk.Label(self.right_frame, bg='#2e2e2e')
        self.detected_label.pack(fill="both", expand=True)

        # 结果文本框
        self.result_text = tk.Text(self.root, height=10, wrap=tk.WORD,
                                   font=("Arial", 12), bg='#444444', fg='white')
        self.result_text.pack(fill="x", padx=20, pady=10)

        scroll_y = tk.Scrollbar(self.result_text, orient="vertical",
                                command=self.result_text.yview)
        scroll_y.pack(side="right", fill="y")
        self.result_text.config(yscrollcommand=scroll_y.set)

    def update_calibration(self):
        try:
            self.PIXELS_PER_METER = float(self.calib_entry.get())
            messagebox.showinfo("成功", "标定参数已更新！")
        except ValueError:
            messagebox.showerror("错误", "请输入有效数字")

    def load_model(self, model_path):
        if model_path is None:
            messagebox.showerror("错误", "请选择模型文件！")
            return None
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = YOLO(model_path)
            model.to(device)
            return model
        except Exception as e:
            messagebox.showerror("模型错误", f"加载模型失败: {str(e)}")
            return None

    def select_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("YOLO模型文件", "*.pt")])
        if file_path:
            self.model_path = file_path
            self.model_entry.delete(0, tk.END)
            self.model_entry.insert(0, file_path)
            self.model = self.load_model(file_path)

    def load_media(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("媒体文件", "*.jpg;*.jpeg;*.png;*.mp4;*.avi")])
        if file_path:
            if file_path.lower().endswith(('.mp4', '.avi')):
                self.video_path = file_path
                self.image_path = None
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"已加载视频: {file_path}\n")
            else:
                self.image_path = file_path
                self.video_path = None
                img = cv2.imread(file_path)
                self.display_image(img, self.original_label)

    def display_image(self, image, label):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((self.display_width, self.display_height))
        img_tk = ImageTk.PhotoImage(img)
        label.config(image=img_tk, width=self.display_width, height=self.display_height)
        label.image = img_tk

    def calculate_speed(self, track_id, position, timestamp):
        self.track_history[track_id]['positions'].append(position)
        self.track_history[track_id]['times'].append(timestamp)

        if len(self.track_history[track_id]['positions']) >= 2:
            first_pos = self.track_history[track_id]['positions'][0]
            first_time = self.track_history[track_id]['times'][0]

            dx = position[0] - first_pos[0]
            dy = position[1] - first_pos[1]
            dt = timestamp - first_time

            if dt > 0:
                distance_pixel = (dx ** 2 + dy ** 2) ** 0.5
                speed_mps = distance_pixel / self.PIXELS_PER_METER / dt
                return speed_mps * 3.6  # 转换为km/h
        return 0.0

    def detect_objects(self):
        if not self.model:
            messagebox.showerror("错误", "请先加载模型！")
            return

        if self.video_path:
            cap = cv2.VideoCapture(self.video_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 处理帧
                frame_resized = cv2.resize(frame, (self.display_width, self.display_height))
                results = self.model.track(
                    frame_resized,
                    persist=True,
                    tracker="botsort.yaml"
                )

                # 获取带追踪结果的图像
                result_img = results[0].plot()

                # 处理追踪信息
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    confs = results[0].boxes.conf.cpu().tolist()
                    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

                    # 更新结果文本
                    self.result_text.delete(1.0, tk.END)

                    for box, tid, conf in zip(boxes, track_ids, confs):
                        x1, y1, x2, y2 = map(int, box)
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)

                        # 计算速度
                        speed = self.calculate_speed(tid, center, current_time)

                        # 在图像上绘制速度
                        speed_text = f"{speed:.1f} km/h"
                        text_size = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.putText(
                            result_img, speed_text,
                            (x2 - text_size[0], y2 + text_size[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                        )

                        # 记录到文本框
                        self.result_text.insert(tk.END,
                                                f"ID {tid} | 速度: {speed:.1f} km/h | 置信度: {conf:.2f}\n")
                        self.result_text.insert(tk.END,
                                                f"位置: ({x1}, {y1}) - ({x2}, {y2})\n\n")

                # 更新显示
                self.display_image(frame, self.original_label)
                self.display_image(result_img, self.detected_label)
                self.root.update()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()

    def detect_from_camera(self):
        if not self.model:
            messagebox.showerror("错误", "请先加载模型！")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头！")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_resized = cv2.resize(frame, (self.display_width, self.display_height))
            results = self.model.track(
                frame_resized,
                persist=True,
                tracker="botsort.yaml"
            )

            result_img = results[0].plot()
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                confs = results[0].boxes.conf.cpu().tolist()

                self.result_text.delete(1.0, tk.END)

                for box, tid, conf in zip(boxes, track_ids, confs):
                    x1, y1, x2, y2 = map(int, box)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    speed = self.calculate_speed(tid, center, current_time)

                    speed_text = f"{speed:.1f} km/h"
                    text_size = cv2.getTextSize(speed_text,
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.putText(
                        result_img, speed_text,
                        (x2 - text_size[0], y2 + text_size[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                    )

                    self.result_text.insert(tk.END,
                                            f"ID {tid} | 速度: {speed:.1f} km/h | 置信度: {conf:.2f}\n")
                    self.result_text.insert(tk.END,
                                            f"位置: ({x1}, {y1}) - ({x2}, {y2})\n\n")

            self.display_image(frame, self.original_label)
            self.display_image(result_img, self.detected_label)
            self.root.update()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    def save_result(self):
        if not self.video_path:
            messagebox.showerror("错误", "没有可保存的结果！")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt")]
        )
        if file_path:
            with open(file_path, 'w') as f:
                f.write(self.result_text.get(1.0, tk.END))
            messagebox.showinfo("保存成功", f"结果已保存到: {file_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()