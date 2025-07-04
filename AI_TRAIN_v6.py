import sys, os, subprocess, ctypes, re, cv2, time, shutil, glob, copy
from PIL import Image as PILImage, ImageEnhance, ImageChops, ImageFilter
from PIL.Image import Transpose     # 버전이슈 _06.05 추가

from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QMetaObject, QTimer, QTime, QFileSystemWatcher
from PyQt6.QtGui import QMovie      # 5.22
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm

import platform

import threading

from collections import deque       # 05.13
from tkinter import *           # 05.23 
from datetime import datetime   # 05.23

import tensorflow as tf     # 06.10
from tensorboard.backend.event_processing import event_accumulator      # 06.10

from PyQt6.QtWidgets import QSizePolicy     # 06.13

## total_step 고정값 X, configs의 num_steps 에서 읽어옴 
# EarlyStopping 방식이라 loss 개선이 안 되면 강제 종료 (wait >= 99)
# Extract model → Ckpt 입력 시 숫자 저장 + Run Training File 버튼 비활성화 

# ProgressBar 멈춤 → [Step] 로그가 멈췄을 때 타임아웃 처리 (15분 경과시 강제 종료) _05.27추가
# shutdown_after_* 값 로그로 확인 + 관리자 권한 포함 강제 종료 처리 _05.27추가
# flush = True : 자꾸 CONSOLE 완료가 안 됨, 모든 print문에 강제로 출력 _05.28추가

# 06.09_오류발생 가능성 방지 코드 삽입

# Gpu(Boost+Normal)_06.04
def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

if not is_admin():
    # 관리자 권한으로 재실행
    ctypes.windll.shell32.ShellExecuteW(
        None, "runas", sys.executable, " ".join(sys.argv), None, 1)
    sys.exit()

# GPU BOOST 실행 (AI_TRAIN.exe 켜지자마자 한 번만 실행)
# try:
# except Exception as e:
#     print(f"[ERROR] GPU Boost 실패: {e}", flush=True)


# Qt6 버전때문에 생길 오류 방지코드 _06.09
# 1. 호환되지 않는 Qt 스타일 DLL 자동 제거
def remove_incompatible_qt_styles():
    try:
        base_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))  # PyInstaller 대응
        style_path = os.path.join(base_dir, "PyQt6", "Qt6", "plugins", "styles")
        style_file = os.path.join(style_path, "qmodernwindowsstyle.dll")
        if os.path.exists(style_file):
            os.remove(style_file)
            print("[INFO] qmodernwindowsstyle.dll 자동 제거 완료", flush=True)
    except Exception as e:
        print(f"[ERROR] DLL 제거 실패: {e}", flush=True)

# 2. DLL 메타데이터 유효성 확인
def check_qt_plugin_compatibility():
    try:
        base_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
        plugin_path = os.path.join(base_dir, "PyQt6", "Qt6", "plugins", "styles")
        if os.path.exists(plugin_path):
            for file in os.listdir(plugin_path):
                if file.endswith(".dll") and "qmodernwindowsstyle" in file.lower():
                    dll_path = os.path.join(plugin_path, file)
                    with open(dll_path, "rb") as f:
                        data = f.read(64)
                        if b"QTMETADATA" not in data:
                            print(f"[경고] {file} 메타데이터 이상 감지 (실행 불안정 가능성)", flush=True)
    except Exception as e:
        print(f"[ERROR] Qt 스타일 DLL 검사 실패: {e}", flush=True)

def check_qt_platform_plugin():
    base_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    qt_platform_path = os.path.join(base_dir, "PyQt6", "Qt6", "plugins", "platforms", "qwindows.dll")
    if not os.path.exists(qt_platform_path):
        print("[오류] Qt 플랫폼 플러그인 (qwindows.dll) 누락됨", flush=True)

def check_path_exists(path, description="경로"):
    if not os.path.exists(path):
        print(f"[경고] {description} 경로 존재하지 않음: {path}", flush=True)

# 3. 스타일 안전 적용
def apply_safe_qt_style():
    try:
        from PyQt6.QtWidgets import QApplication
        QApplication.setStyle("Fusion")  # 안정적인 기본 스타일
    except Exception as e:
        print(f"[경고] Qt 스타일 적용 실패: {e}", flush=True)

# 4. 전체 통합 실행 (main 함수 실행 전 호출)
def safe_qt_init():
    remove_incompatible_qt_styles()
    check_qt_plugin_compatibility()
    check_path_exists(r"D:\\AI_SVT_Training_mk\\annotations\\annos", "XML 폴더")
    check_path_exists(r"D:\\AI_SVT_Training_mk\\configs", "Config 파일 경로")
    apply_safe_qt_style()

# 기존 exe 코드
class TrainingThread(QThread):
    progress_signal = pyqtSignal(float)
    done_signal = pyqtSignal(tuple)     # 06.05 수정, tuple형태로 best_loss전달
    loss_signal = pyqtSignal(float, int)    # float : Loss, int : Step 전달

    # 시간 걸어서 shutdown 추가 | 사용자 선택 시각_250613
    def __init__(self, max_steps, shutdown_enable = False, shutdown_hour=0, shutdown_minute=0, parect=None):
        super().__init__(parect)        # 06.25 parect 추가, 위에도 동일
        self.shutdown_enable = shutdown_enable
        self.shutdown_hour = shutdown_hour
        self.shutdown_minute = shutdown_minute
        self.max_steps = max_steps      # GUI 가 넘겨준 train_steps 저장

    # 무조건 종료_05.27
    def run(self):
        # total_steps = None
        current_step = 0
        # loss율 _06.05
        best_loss = float('inf')
        best_step = 0
        done_flag = False
        final_step = None

        cmd_command = 'D:\\AI_SVT_Training_mk\\1)train_FRCNN_res50.bat'
        # print(f"[LOG] 훈련 스레드 시작 (무조건 종료 모드)", flush=True)
        process = subprocess.Popen(cmd_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True)
        # last_step_time = time.time()      # 06.25 주석
        # 06.25 추가_ 이제 total_steps 는 로그에서 읽지 않고 self.max_steps 고정 
        # total_steps = self.max_steps
        
        prev_line = ""  # 이전 줄 저장

        for line in iter(process.stdout.readline, ''):
            print(line.strip(), flush=True)

            total_match = re.search(r'\[TOTAL_STEPS\]\s*(\d+)', line)
            if total_match:
                final_step = int(total_match.group(1))
                prev_line = line
                continue

            step_match = re.search(r'\[Step\]\s*(\d+)', line)
            if step_match:
                current_step = int(step_match.group(1))
                if final_step is not None and current_step >= self.max_steps:       
                    done_flag = True

            loss_match = re.search(r'\[LOSS\]\s*([0-9.]+)', prev_line + line)
            if loss_match:
                loss = float(loss_match.group(1))
                if loss < best_loss:
                    best_loss = loss
                    best_step = current_step
                self.loss_signal.emit(loss, current_step)

                if done_flag:
                    # 1. 훈련 완료 신호 _07.03
                    self.progress_signal.emit(final_step or current_step)
                    self.done_signal.emit((best_loss, best_step))

                    # 2. GPU 설정 복원
                    print("→ GpuNormal.bat 실행", flush=True)
                    subprocess.run(['nvidia-smi', '-rgc'], check=True)
                    subprocess.run(['nvidia-smi', '-rmc'], check=True)
                    time.sleep(2)

                    # # 07.03 _3. 배치 스크립트 끝까지 출력
                    # remaining,_ = process.communicate()
                    # if remaining:
                    #     print(remaining.strip(), flush=True)

                    # 4. 훈련 종료 후 설정된 시각에만 PC 종료
                    if self.shutdown_enable:
                        now = datetime.now()
                        threshold = now.replace(
                            hour=self.shutdown_hour,
                            minute=self.shutdown_minute,
                            second=0,
                            microsecond=0
                        )
                        if now >= threshold:
                            delay = 60
                        else:
                            delay = int((threshold - now).total_seconds())
                        print(f"PC를 {delay}초 후에 종료합니다", flush=True)
                        subprocess.run(f"shutdown /s /t {delay} /f", shell=True)
                    # 07.03
                    try:
                        remaining,_ = process.communicate()
                        if remaining:
                            print(remaining.strip(), flush=True)
                    except subprocess.TimeoutExpired:
                        process.kill()

                    break   # 반복 X

            # 정상 훈련 중: 스텝 값 그대로 emit
            self.progress_signal.emit(current_step)
            prev_line = line       # 현재 줄을 다음 반복에서 이전 줄로 저장
        
        process.stdout.close()
        process.wait()

# # Choose Image Augmentation Options 관련 ProgressBar
# xml_dir = r"D:\AI_SVT_Training_mk\annotations\annos"
# for file in os.listdir(xml_dir):
#     if file.endswith(".xml"):
#         path = os.path.join(xml_dir, file)
#         with open(path, "r", encoding="utf-8") as f:
#             lines = f.readlines()
#         if lines and lines[0].strip().startswith("<?xml"):
#             lines = lines[1:]  # 선언문 제거
#         with open(path, "w", encoding="utf-8") as f:
#             f.writelines(lines)


class ClassAugChanger(QMainWindow):
    progress_signal = pyqtSignal(int)       # 추가
    done_signal = pyqtSignal()              # 추가

    def __init__(self):
        super().__init__()
        # self.progress_bar = QProgressBar(self)      # 추가 했다가 다시 뺌
        self.progress_signal.connect(self.progress_bar_update)
        self.initUI()
        self.aug_list = []
        self.folder_name_clahe = str("aug_clahe")
        self.folder_name_hs = str("aug_H_Shift")
        self.folder_name_vs = str("aug_V_Shift")
        self.folder_name_hf = str("aug_H_Flip")
        self.folder_name_vf = str("aug_V_Flip")
        self.folder_name_rt = str("aug_Rotation")
        self.folder_name_clahe = str("aug_clahe")
        self.folder_rbc = str("aug_RowByColumn")
        self.num_shift = [*range(1, 10, 2)]
        self.num_rot = [*range(45, 360, 45)]
        self.output_folder_rgb = "output_rgb_images"
        self.output_folder_bgr = "output_bgr_images"
        self.folder_gray = "gray_scale_images"
        self.folder_color = "color_images"

    # 입력커서(클래스명)
    def on_text_edited(self, text):
        if not text.startswith(self.prefix):
            cursor_pos = self.input_delete_classsname.cursorPosition()
            new_text = self.prefix + text[len(self.prefix):] if len(text) > len(self.prefix) else self.prefix
            self.input_delete_classsname.setText(new_text)
            self.input_delete_classsname.setCursorPosition(max(cursor_pos, len(self.prefix)))
        elif len(text) < len(self.prefix):
            self.input_delete_classsname.setText(self.prefix)
            self.input_delete_classsname.setCursorPosition(len(self.prefix))

    def on_text_edited_1(self, text):
        if not text.startswith(self.prefix_1):
            cursor_pos = self.input_new_classsname.cursorPosition()
            new_text = self.prefix_1 + text[len(self.prefix_1):] if len(text) > len(self.prefix_1) else self.prefix_1
            self.input_new_classsname.setText(new_text)
            self.input_new_classsname.setCursorPosition(max(cursor_pos, len(self.prefix_1)))
        elif len(text) < len(self.prefix_1):
            self.input_new_classsname.setText(self.prefix_1)
            self.input_new_classsname.setCursorPosition(len(self.prefix_1))

    # 입력커서(행렬 변환)_06.18
    def on_row_edited(self, text):
        if not text.startswith(self.prefix_row):
            cursor_pos = self.row_input.cursorPosition()
            new_text = self.prefix_row + text[len(self.prefix_row):] if len(text) > len(self.prefix_row) else self.prefix_row
            self.row_input.setText(new_text)
            self.row_input.setCursorPosition(max(cursor_pos, len(self.prefix_row)))
        elif len(text) < len(self.prefix_row):
            self.row_input.setText(self.prefix_row)
            self.row_input.setCursorPosition(len(self.prefix_row))

    def on_col_edited(self, text):
        if not text.startswith(self.prefix_col):
            cursor_pos = self.column_input.cursorPosition()
            new_text = self.prefix_col + text[len(self.prefix_col):] if len(text) > len(self.prefix_col) else self.prefix_col
            self.column_input.setText(new_text)
            self.column_input.setCursorPosition(max(cursor_pos, len(self.prefix_col)))
        elif len(text) < len(self.prefix_col):
            self.column_input.setText(self.prefix_col)
            self.column_input.setCursorPosition(len(self.prefix_col))

    # 06.19 (ckpt)
    def on_ckpt_edited(self, text):
        if not text.startswith(self.prefix_ckpt):
            cursor_pos = self.enter_chkp.cursorPosition()
            new_text = self.prefix_ckpt + text[len(self.prefix_ckpt):] if len(text) > len(self.prefix_ckpt) else self.prefix_ckpt
            self.enter_chkp.setText(new_text)
            self.enter_chkp.setCursorPosition(max(cursor_pos, len(self.prefix_ckpt)))
        elif len(text) < len(self.prefix_ckpt):
            self.enter_chkp.setText(self.prefix_ckpt)
            self.enter_chkp.setCursorPosition(len(self.prefix_ckpt))

    # 입력란에서 실제 클래스명만 추출
    def get_real_classname(self):
        text = self.input_delete_classsname.text()
        if text.startswith(self.prefix):
            return text[len(self.prefix):]
        return ""

    def get_real_classname_1(self):
        text = self.input_new_classsname.text()
        if text.startswith(self.prefix_1):
            return text[len(self.prefix_1):]
        return ""

    # 입력란에서 실제 행렬 변환 행, 열 숫자만 추출
    def get_row_number(self):
        text = self.row_input.text()
        if text.startswith(self.prefix_row):
            return text[len(self.prefix_row):]
        return ""

    def get_col_number(self):
        text = self.column_input.text()
        if text.startswith(self.prefix_col):
            return text[len(self.prefix_col):]
        return ""

    # 06.19 (ckpt)
    def get_ckpt_number(self):
        text = self.enter_chkp.text()
        if text.startswith(self.prefix_ckpt):
            return text[len(self.prefix_ckpt):]
        return ""


    def initUI(self):
        self.setWindowTitle('SVT model prep')
        self.setGeometry(100, 100, 800, 400)
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        self.setStyleSheet("""
            QWidget {
                background-color: #FFFBE6;
            }

            QPushButton {
                background-color: #DDECCC;
                color: #4E7D5B;
                border: 1px solid #BBD4B3;
                border-radius: 8px;
                padding: 6px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #CFE3B9;
            }

            QPushButton:pressed {
                background-color: #BBD4B3;
            }

            QProgressBar {
                border: 2px solid #D6D6D6;
                border-radius: 5px;
                background-color: #FAFDEB;
            }

            QProgressBar::chunk {
                background-color: #C5DDB3;
            }

            QLineEdit {
                background-color: #FFFBE6;
                color: #4E7D5B;
                border: 1px solid #BBD4B3;
                border-radius: 6px;
                padding: 4px;
            }

            QLineEdit::placeholder {
                color: #556B2F;
            }
        """)

        # 06.25 추가
        # (1) annos 폴더 경로 설정
        annos_dir = r"D:\AI_SVT_Training_mk\annotations\annos"
        
        # (2) 이미지 파일 개수 세기
        try:
            num_images = len([
                f for f in os.listdir(annos_dir)
                if os.path.isfile(os.path.join(annos_dir, f)) and f.lower().endswith(('.jpg','.jpeg','.png','.bmp')) 
            ])
        except (OSError, FileNotFoundError):
            num_images = 0
            self.num_images = num_images

        self.num_images = num_images
        self.update_epochno_file()

        # (3) EpochNo.txt 초기 생성(기본 x20)
        annotation_dir = r"D:\AI_SVT_Training_mk\annotations"
        os.makedirs(annotation_dir, exist_ok=True)
        epoch_file = os.path.join(annotation_dir, "EpochNo.txt")
        # 06.26 한줄 추가
        if not os.path.exists(epoch_file):
            with open(epoch_file, "w", encoding="utf-8") as f:
                f.write(str(self.num_images * 20))

        # (4) EpochNo.txt 읽어서 클램핑
        try:
            with open(epoch_file, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            iMaxTrainStep = int(raw)
        except (ValueError, OSError):
            # 파일 이상 시 기본 mul = 20
            iMaxTrainStep = self.num_images * 20
            
        iMaxTrainStep = min(iMaxTrainStep, 100000)
        iMaxTrainStep = max(iMaxTrainStep, 10000)
        train_steps = iMaxTrainStep     # 06.25
        self.train_steps = train_steps      # 06.25
        # self.progress_bar.setMaximum(self.train_steps)      # 06.26

        print(f"[TRAIN_STEPS] {train_steps}", flush=True)       # 06.25

        layout = QVBoxLayout(central_widget)
        class_buttons_layout = QHBoxLayout()
        aug_layout = QGridLayout()
        input_layout = QHBoxLayout()
        rowNcol_layout = QHBoxLayout()
        img_conversion_layout = QHBoxLayout()
        train_layout = QHBoxLayout()
        inference_layout = QHBoxLayout()

        # (1) 06.25 _annos 읽어오기
        self.annos_dir = r"D:\AI_SVT_Training_mk\annotations\annos"
        # (2) QFileSystemWatcher 생성 및 연결
        self.watcher = QFileSystemWatcher([self.annos_dir], parent=self)
        self.watcher.directoryChanged.connect(self.on_annos_changed)

        train_result_path = r"D:\AI_SVT_Training_mk\train_result\checkpoint"
        self.train_result_path = train_result_path.replace("\\", "/")
        config_path = r"D:\AI_SVT_Training_mk\configs\faster_rcnn_resnet50_v1_800x1333_batch1.config"
        self.config_file_path = config_path.replace("\\", "/")
        path = r"D:\AI_SVT_Training_mk\annotations\annos"
        labelmap_path = os.path.join(r"D:\AI_SVT_Training_mk\annotations\label_map.pbtxt")
        #labelmap_path = r"D:\AI_SVT_Training_mk\annotations\label_map.pbtxt"
        self.labelmap_path = labelmap_path.replace("\\", "/")

        self.label_check = QLabel("Check Training Files")
        font = self.label_check.font()
        font.setPointSize(15)
        self.label_check.setFont(font)
        self.label_check.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.label_check)
        # 👻🐸🐥🐣🦄🍀🌸🥑
        # 버전 변경한거 확인용
        # 오른쪽 정렬
        self.label_check_version = QLabel("🐸2025.06.19.version🐸")
        font_small = self.label_check_version.font()
        font_small.setPointSize(9)
        self.label_check_version.setFont(font_small)
        self.label_check_version.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.label_check_version.setStyleSheet("color: 	#006400;")       # #ff3399 핑크색 
        layout.addWidget(self.label_check_version)

        self.set_location = QPushButton('폴더 선택', self)
        self.set_location.clicked.connect(self.directory)
        layout.addWidget(self.set_location)

        self.bmp2jpg_button = QPushButton('.bmp to .jpg', self)
        self.bmp2jpg_button.clicked.connect(self.bmptojpg)
        layout.addWidget(self.bmp2jpg_button)

        self.check_all_button = QPushButton('파일 확인', self)
        self.check_all_button.clicked.connect(self.checkAllFiles)
        class_buttons_layout.addWidget(self.check_all_button)

        self.label_map_button = QPushButton('라벨 맵 생성', self)
        self.label_map_button.clicked.connect(self.label_map)
        class_buttons_layout.addWidget(self.label_map_button)

        self.change_button = QPushButton('클래스 번호 변경', self)
        self.change_button.clicked.connect(self.changeClassNum)
        class_buttons_layout.addWidget(self.change_button)
        layout.addLayout(class_buttons_layout)

        self.info_label = QLabel(self)
        self.info_label.setStyleSheet("background-color: rgba(255, 255, 255, 150); padding: 5px")
        layout.addWidget(self.info_label)

        self.input_delete_classsname = QLineEdit(self)
        self.prefix = "기존 클래스명: "
        self.input_delete_classsname.setText(self.prefix)
        input_layout.addWidget(self.input_delete_classsname)

        self.input_delete_classsname.textEdited.connect(self.on_text_edited)
        
        self.input_new_classsname = QLineEdit(self)
        self.prefix_1 = "변경할 클래스명: "
        self.input_new_classsname.setText(self.prefix_1)
        input_layout.addWidget(self.input_new_classsname)
        layout.addLayout(input_layout)

        self.input_new_classsname.textEdited.connect(self.on_text_edited_1)

        self.change_class_name_button = QPushButton("클래스명 변경")
        self.change_class_name_button.clicked.connect(self.changeClassName)
        layout.addWidget(self.change_class_name_button)

        # 라벨 입력창과 버튼 추가
        # self.label_input_old = QLineEdit()
        # self.label_input_old.setPlaceholderText("기존 라벨명 입력")
        # layout.addWidget(self.label_input_old)

        # self.label_input_new = QLineEdit()
        # self.label_input_new.setPlaceholderText("변경할 라벨명 입력")
        # layout.addWidget(self.label_input_new)

        # self.label_change_button = QPushButton("라벨명 변경")
        # self.label_change_button.clicked.connect(self.change_label_run)
        # layout.addWidget(self.label_change_button)

        self.aug_check = QLabel("Choose Image Augmentation Options")
        font = self.aug_check.font()
        font.setPointSize(15)
        self.aug_check.setFont(font)
        self.aug_check.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.aug_check)

        self.clear_aug_label = QLabel("Augmentation Info")
        layout.addWidget(self.clear_aug_label)

        self.hf_button = QPushButton('수평 뒤집기(H Flip)')
        self.hf_button.clicked.connect(self.imgHflip)
        self.hf_button.clicked.connect(self.xmlHflip)
        aug_layout.addWidget(self.hf_button, 0, 0)

        self.hs_button = QPushButton('수평 이동(H Shift)')
        self.hs_button.clicked.connect(self.imgHshift)
        self.hs_button.clicked.connect(self.xmlHshift_all)
        aug_layout.addWidget(self.hs_button, 0, 1)

        self.rot_button = QPushButton('회전(Rotation)')
        self.rot_button.clicked.connect(self.imgRotation)
        self.rot_button.clicked.connect(self.xmlRotation_all)
        aug_layout.addWidget(self.rot_button, 0, 2)

        self.vf_button = QPushButton('수직 뒤집기(V Flip)')
        self.vf_button.clicked.connect(self.imgVflip)
        self.vf_button.clicked.connect(self.xmlVflip)
        aug_layout.addWidget(self.vf_button, 1, 0)

        self.vs_button = QPushButton('수직 이동(V Shift)')
        self.vs_button.clicked.connect(self.imgVshift)
        self.vs_button.clicked.connect(self.xmlVshift_all)
        aug_layout.addWidget(self.vs_button, 1, 1)

        self.clahe_button = QPushButton('CLAHE')
        self.clahe_button.clicked.connect(self.clahe_aug)
        aug_layout.addWidget(self.clahe_button, 1, 2)
        layout.addLayout(aug_layout)

        # xml 변경 버튼 추가 (하나의 xml 파일에 object 추가) _PJS
        self.xml_transform_button = QPushButton("XML 객체 추가(단일 이미지에 복수 객체 추가)", self)
        self.xml_transform_button.clicked.connect(self.transform_xml_boxes)
        layout.addWidget(self.xml_transform_button)   

        # Auto Label XML 버튼 추가 (JPG, XML 따로따로 생성) _PJS (실제 훈련에 들어갈 jpg, xml 파일 생성 버튼)
        self.mm003_button = QPushButton('Auto Label XML 조정(복수 이미지에 복수 파일 추가)', self)
        self.mm003_button.clicked.connect(self.process_xml_variants)
        layout.addWidget(self.mm003_button)

        self.prefix_row = "행 번호: "
        self.prefix_col = "열 번호: "


        self.row_input = QLineEdit(self)
        self.row_input.setText(self.prefix_row)
        rowNcol_layout.addWidget(self.row_input)
        self.row_input.textEdited.connect(self.on_row_edited)

        self.column_input = QLineEdit(self)
        self.column_input.setText(self.prefix_col)
        rowNcol_layout.addWidget(self.column_input)
        layout.addLayout(rowNcol_layout)
        self.column_input.textEdited.connect(self.on_col_edited)

        self.RowByColumn_button = QPushButton('행렬 변환(Row by Column)')
        self.RowByColumn_button.clicked.connect(self.RowColumn_all)
        layout.addWidget(self.RowByColumn_button)

        self.colortogray_button = QPushButton('Color to Gray')
        self.colortogray_button.clicked.connect(self.ColorToGray)
        img_conversion_layout.addWidget(self.colortogray_button)

        self.rgb_button = QPushButton('BGR to RGB')
        self.rgb_button.clicked.connect(self.BGRtoRGB)
        img_conversion_layout.addWidget(self.rgb_button)

        self.bgr_button = QPushButton('RGB to BGR')
        self.bgr_button.clicked.connect(self.RGBtoBGR)
        img_conversion_layout.addWidget(self.bgr_button)
        layout.addLayout(img_conversion_layout)

        # self.save_button = QPushButton('저장')
        # self.save_button.clicked.connect(self.save_code)
        # layout.addWidget(self.save_button)

        self.train_label = QLabel("Proceed Training")
        font = self.train_label.font()
        font.setPointSize(15)
        self.train_label.setFont(font)
        self.train_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.train_label)

        # ProgressBar 추가_PJS
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat('%p%')     # % 포멧
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)    # 숫자 가운데정렬
        layout.addWidget(self.progress_bar)
        self.progress_bar.setMaximum(self.train_steps)  # 06.26

        # # 샷다운추가_250523 # 👻🐸🐥🐣🦄🍀🌸🥑
        # self.shutdown_checkbox = QCheckBox("🥑훈련 종료 후 18시 이후면 PC 종료🥑")
        # layout.addWidget(self.shutdown_checkbox)

        # 250613_18시 말고 선택 시각 이후에 종료
        shutdown_layout = QHBoxLayout()
        self.shutdown_checkbox = QCheckBox("🥑훈련 종료 후 설정된 시각 이후 PC 종료🥑")
        shutdown_layout.addWidget(self.shutdown_checkbox)

        self.shutdown_time_edit = QTimeEdit(self)
        self.shutdown_time_edit.setDisplayFormat("HH:mm")
        self.shutdown_time_edit.setTime(QTime(00, 0))   # 기본값 00:00
        # self.shutdown_time_edit 크기 조정
        self.shutdown_time_edit.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred))
        self.shutdown_time_edit.setMaximumWidth(100)  # 최대 너비 설정
        self.shutdown_time_edit.setMinimumWidth(80)   # 최소 너비 설정
        # StyleSheet로 디자인 변경
        self.shutdown_time_edit.setStyleSheet("""
            QTimeEdit {
                border: 2px solid #BBD4B3;
                border-radius: 6px;
                padding: 4px;
                background-color: #FFFFFF;
                font-size: 14px;
            }
            QTimeEdit::up-button, QTimeEdit::down-button {
                width: 16px; height: 16px;
                subcontrol-origin: border;
            }
            QTimeEdit:hover {
                background-color: #F1FFF0;
            }
            QTimeEdit:focus {
                border-color: #66BB6A;
            }
        """)   
        shutdown_layout.addWidget(self.shutdown_time_edit)
        layout.addLayout(shutdown_layout) 

        # ProgressBar 색상 변경 << background 여기서 변경 해주면 됨
        self.progress_bar.setStyleSheet("""
        QProgressBar {
            border: 2px solid grey;
            border-radius: 5px;
        }
        QProgressBar::chunk {
            background-color: 	#E6E6FA;   
            width: 20px;
        }
        """)
        
        self.run_train = QPushButton('훈련 시작', self)
        self.run_train.clicked.connect(self.runCommand)
        train_layout.addWidget(self.run_train)
        
        # 버튼 추가 _PJS
        self.check_train_result_button = QPushButton('훈련 파일 결과 확인', self)
        self.check_train_result_button.clicked.connect(self.checkTrainingComplete)
        train_layout.addWidget(self.check_train_result_button)

        self.pbmodel_button = QPushButton('모델 추출', self)
        self.pbmodel_button.clicked.connect(self.pbmodel)
        train_layout.addWidget(self.pbmodel_button)
        layout.addLayout(train_layout)

        self.prefix_ckpt = "체크포인트 입력: "       # 06.19

        self.enter_chkp = QLineEdit(self)
        self.enter_chkp.setText(self.prefix_ckpt)           # 이 부분도 글자 고정시킬거임
        layout.addWidget(self.enter_chkp)

        self.enter_chkp.textEdited.connect(self.on_ckpt_edited)

        self.label_ckpt_status = QLabel("")
        self.label_ckpt_status.setStyleSheet("color: green; font-weight: bold;")    
        layout.addWidget(self.label_ckpt_status)

        # 06.24 _Radiobutton 추가
        radio_layout = QHBoxLayout()

        # 버튼 간격 설정
        radio_layout.setSpacing(20)
        # 레이아웃 바깥 여백 설정
        radio_layout.setContentsMargins(10, 5, 10, 5)  

        # 1) 라디오 버튼 생성
        self.radio10 = QRadioButton("× 10")
        self.radio20 = QRadioButton("× 20")
        self.radio30 = QRadioButton("× 30")
        self.radio20.setChecked(True)  # 기본 20

        # 2) 버튼 그룹에 ID 연결
        self.bg_mul = QButtonGroup(self)
        self.bg_mul.addButton(self.radio10, 10)
        self.bg_mul.addButton(self.radio20, 20)
        self.bg_mul.addButton(self.radio30, 30)

        # 3) 레이아웃에 추가
        radio_layout.addStretch(1)          # 왼쪽 여백
        radio_layout.addWidget(self.radio10)
        radio_layout.addWidget(self.radio20)
        radio_layout.addWidget(self.radio30)
        radio_layout.addStretch(1)          # 오른쪽 여백
        layout.addLayout(radio_layout)  # layout은 initUI에서 정의된 메인 레이아웃

        # 4) 시그널 연결
        self.bg_mul.buttonClicked.connect(self.on_mul_changed)

        # 06.05 _loss율
        self.label_loss_status = QLabel("실시간 Loss: -")
        layout.addWidget(self.label_loss_status)

    # 06.05 _loss율
    def update_loss_label(self, loss, step):
        # print(f"[GUI 업데이트] Step {step}, Loss {loss:.4f}", flush=True)
        QTimer.singleShot(0, lambda: self.label_loss_status.setText(f"Step {step} → Loss: {loss:.4f}"))

    # 06.25 _Radiobutton + EpochNo.txt 추가
    def on_mul_changed(self, button):
        mul = self.bg_mul.id(button)    # 0, 1, 2 -> 10, 20, 30
        
        # 매번 annos 이미지 개수 다시 계산
        annos_dir = r"D:\AI_SVT_Training_mk\annotations\annos"
        try:
            num_images = len([
                f for f in os.listdir(annos_dir)
                if os.path.isfile(os.path.join(annos_dir, f)) and
                    f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))
            ])
        except (OSError, FileNotFoundError):
            num_images = 0

        # EpocNo.txt 업데이트
        annotations_dir = r"D:\AI_SVT_Training_mk\annotations"
        os.makedirs(annotations_dir, exist_ok=True)
        epoch_file = os.path.join(annotations_dir, "EpochNo.txt")
        with open(epoch_file, "w", encoding="utf-8") as f:
            f.write(str(self.num_images * mul))
        print(f"[EpochNo.txt] {self.num_images}×{mul} → {self.num_images*mul}", flush=True)     # 팝업창 추가
        
        # 06.26 변경된 EpochNo.txt 읽어서 train_steps 에 반영
        try:
            with open(epoch_file, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            iMaxTrainStep = int(raw)
        except (ValueError, OSError):
            iMaxTrainStep = self.num_images * mul
        iMaxTrainStep = min(iMaxTrainStep, 100000)
        iMaxTrainStep = max(iMaxTrainStep, 10000)
        self.train_steps = iMaxTrainStep
        self.progress_bar.setMaximum(self.train_steps)
            
        # 팝업창 추가
        msg = QMessageBox(self)
        msg.setWindowTitle("EpochNo 업데이트")
        msg.setText(f"훈련 스텝이 이미지개수 {self.num_images}개 × {mul} = {self.num_images*mul} 로 설정되었습니다.")
        msg.setIcon(QMessageBox.Icon.Information)
        msg.addButton("확인", QMessageBox.ButtonRole.AcceptRole)
        msg.exec()  # 모달로 실행

        # self.num_images 갱신
        self.num_images  = num_images 


    def pbmodel(self):
        checkpoint_number = self.get_ckpt_number().strip()
        if not checkpoint_number.isdigit():
            QMessageBox.warning(self, "입력 오류", "유효한 숫자를 입력하세요.")
            return
        checkpoint_number = int(int(checkpoint_number)/200)
        path_chkp = self.train_result_path
        if path_chkp:
            try:
                with open(path_chkp, 'r') as file:
                    lines = file.readlines()
                if lines:
                    lines[0] = f'model_checkpoint_path: "ckpt-{checkpoint_number}"\n'
                with open(path_chkp, 'w') as file:
                    file.writelines(lines)
                self.label_ckpt_status.setText(f"체크포인트가 업데이트 되었습니다.: ckpt-{checkpoint_number}")
                self.label_ckpt_status.setStyleSheet("color: green; font-weight: bold;")
            except Exception as e:
                QMessageBox.critical(self, "오류", f"파일 업데이트 중 오류 발생: {e}")
        try:
            cmd_pb_directory = 'D:\\AI_SVT_Training_mk\\2)model_pb.bat'
            if sys.platform.startswith('win'):
                self.cmd_pb = subprocess.Popen(['cmd', '/c', cmd_pb_directory],
                                                    creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                self.cmd_pb = subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', cmd_pb_directory])
        except Exception as e:
            print(f"오류가 발생했습니다.: {e}", flush=True)

        # 버튼 비활성화_05.21 추가
        self.run_train.setEnabled(False)


    def runCommand(self):
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(self.train_steps)      # 06.26 (100 에서 self.train_steps로 변경)
        self.progress_bar.setFormat("0.00%")  # 초기 포맷 설정
       
        # Run Training File 버튼 누르면 GpuBoost 실행
        try:
            print("[LOG] GPU Boost 실행", flush=True)
            subprocess.run(['nvidia-smi', '-lgc', '2500,2580'], check=True)
        except Exception as e:
            print(f"[ERROR] GPU Boost 실패: {e}", flush=True)


        # 샷다운 추가_250523 , 설정된 시각으로 추가_250613
        shutdown_flag = self.shutdown_checkbox.isChecked()
        qt = self.shutdown_time_edit.time()
        shutdown_hour = qt.hour()
        shutdown_minute = qt.minute()

        # 변경 : TrainingTread에 시, 분 인자 전달 _250613 
        self.train_thread = TrainingThread(
            shutdown_enable= shutdown_flag,
            shutdown_hour=shutdown_hour,
            shutdown_minute=shutdown_minute,
            # 06.25 추가
            max_steps=self.train_steps
        )

        self.train_thread.progress_signal.connect(self.progress_bar_update)  # 연결
        self.train_thread.done_signal.connect(self.onTrainingDone)  # 06.05 _loss
        self.train_thread.loss_signal.connect(self.update_loss_label)   # 06.05 _실시간 loss

        # 훈련 시작
        self.train_thread.start()
        # 훈련시작 버튼 비활성화
        self.run_train.setEnabled(False)
        # 라디오 버튼 비활성화      # 06.25
        self.radio10.setEnabled(False)
        self.radio20.setEnabled(False)
        self.radio30.setEnabled(False)


    # 06.25 추가
    def progress_bar_update(self, step):
        # ProgBar 값 설정
        # self.progress_bar.setValue(step)
        # percent = step / self.train_steps * 100
        # 06.30 1. raw step 이 최대 스텝 넘지 않게 자르기
        raw = min(step, self.train_steps)
        self.progress_bar.setValue(raw)
        # 2. 퍼센트가 100% 이상으로 올라가지 않게 _ 1,2번만 06.30 추가
        percent = raw / self.train_steps * 100
        self.progress_bar.setFormat(f"{percent:.2f}%")

    # 06.05 _loss율
    def onTrainingDone(self, best_info):
        best_loss, best_step = best_info
        QMessageBox.information(self, "훈련 완료", f"모델 훈련이 완료됐습니다.\n\n"
                                                f"최적 Loss : {best_loss:.4f}\n"
                                                f"해당 Step : {best_step}")
        # --- 팝업 OK 누른 직후 프로그레스바를 0%로 리셋 _07.01
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0.00%")

        self.run_train.setEnabled(True)
        # 06.25 라디오버튼 비활성화 추가
        self.radio10.setEnabled(True)
        self.radio20.setEnabled(True)
        self.radio30.setEnabled(True)

    def checkTrainingComplete(self):            # 버튼 추가에 관한 내용_PJS
        checkpoint_path = os.path.join(r"D:\\AI_SVT_Training_mk\\train_result")
        if os.path.isdir(checkpoint_path) and os.listdir(checkpoint_path):
            QMessageBox.information(self, "훈련 완료", "훈련이 완료됐습니다.\ntrain_result 폴더가 존재합니다.")
        else:QMessageBox.warning(self, "훈련 완료X", "아직 훈련이 완료되지 않았습니다.")

    def directory(self):
        options = QFileDialog.Option(QFileDialog.Option.ReadOnly)
        self.chosen_directory = QFileDialog.getExistingDirectory(self, "Open Directory", "", options=options)
        if self.chosen_directory:
            self.folder_path = self.chosen_directory.replace("\\", '/')
            self.img_folder_list = [os.path.join(self.chosen_directory, filename).replace("\\", '/') for
                                    filename in os.listdir(self.chosen_directory) if filename.lower().endswith(('.jpg', '.jpeg', '.JPG', '.JPEG'))]
            self.xml_folder_list = [os.path.join(self.chosen_directory, filename).replace("\\", '/') for filename in
                                os.listdir(self.chosen_directory) if filename.lower().endswith(('.xml', '.XML'))]

    def bmptojpg(self):
        src_path = self.folder_path
        dst_path = os.path.join(src_path, "jpg_images/")  # jpg images path
        if not os.path.isdir(dst_path):  # make dst dir if it's not existed
            os.mkdir(dst_path)
        bmp_path = os.path.join(src_path, 'bmp_images/')
        os.makedirs(bmp_path, exist_ok=True)
        src_path_bmp = (img for img in set(glob.glob(src_path + "/*.bmp" or src_path + "/*.BMP")))
        # bmpfiles=[img for img in set(glob.glob(src_path+"/*.bmp" or src_path+"/*.BMP"))]

        for img in tqdm(src_path_bmp, desc='iterate list'):
            time.sleep(0.1)
            images = PILImage.open(img)
            name = img.split("\\")[-1]
            name = name.split(".")[0] + ".jpg"
            new_name = os.path.join(dst_path, name)
            new_name = new_name.replace("\\", "/")
            images.save(new_name)
            images.close()
        for img in set(glob.glob(src_path + "/*.bmp" or src_path + "/*.BMP")):
            img = img.replace("\\", "/")
            print(f"file:{img}", flush=True)
            shutil.move(img, bmp_path)

    def changeClassName(self):
        original_fileName = str(self.input_delete_classsname.text())
        modi_fileName = str(self.input_new_classsname.text())
        prefix = "기존 클래스명: "
        if original_fileName.startswith(prefix):
            original_fileName = original_fileName[len(prefix):]
        else:
            original_fileName = original_fileName
        prefix_1 = "변경할 클래스명: "
        if modi_fileName.startswith(prefix_1):
            modi_fileName = modi_fileName[len(prefix_1):]
        else:
            modi_fileName = modi_fileName
        for xml_file in self.xml_folder_list:
            targetXML = open(xml_file, 'rt', encoding='utf-8')
            tree = ET.parse(targetXML)
            root = tree.getroot()
            for obj in root.iter('object'):
                class_name = obj.find('name')
                original = class_name.text
                modified = original.replace(original_fileName, modi_fileName)
                class_name.text = modified
            tree.write(xml_file)
        self.info_label.setText(f'기존 클래스 {original_fileName} 이(가) {modi_fileName}로 변경되었습니다.')

    def checkAllFiles(self):
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.xml'):
                xml_path = os.path.join(self.folder_path, filename)
                image_path = os.path.join(self.folder_path, os.path.splitext(filename)[
                    0] + '.jpg')
                if not os.path.exists(image_path):
                    os.remove(xml_path)
            elif filename.endswith('.jpg'):
                image_path = os.path.join(self.folder_path, filename)
                xml_path = os.path.join(self.folder_path, os.path.splitext(filename)[0] + '.xml')
                if not os.path.exists(xml_path):
                    os.remove(image_path)
        for xml_file in self.xml_folder_list:
            targetXML = open(xml_file, 'rt', encoding='utf-8')
            tree = ET.parse(targetXML)
            root = tree.getroot()
            target_tag = root.find("object")
            if target_tag is None:
                targetXML.close()
                os.remove(xml_file)
        self.info_label.setText('비어있는 jpg, xml 파일이 삭제되었습니다.')

    def label_map(self):
        self.num_classes, self.class_names = self.count_classes_in_folder(self.xml_folder_list)
        self.info_label.setText(f'Number of classes: {self.num_classes}. Class names: {self.class_names}')
        with open(self.labelmap_path, "w", encoding='utf-8') as a:
            for i, class_name in enumerate(self.class_names, 1):
                self.pbtxt_content = f"item {{ \n id: {i}\n name:'{class_name}'\n display_name:'{class_name}'\n}}\n"
                a.write(self.pbtxt_content)

    def count_classes_in_folder(self, xml_folder_list):
        class_set = set()
        for xml_file in xml_folder_list:
            classes = self.extract_classes_from_xml(xml_file)
            class_set.update(classes)
        return len(class_set), class_set

    def extract_classes_from_xml(self, xml_file):
        classes = set()
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.findall('.//object'):
            class_name = obj.find('name').text
            classes.add(class_name)
        return classes

    def changeClassNum(self):
        maxid = self.maxid()
        with open(self.config_file_path, "r") as config_file:
            config_content = config_file.read()
        pattern = r"num_classes:\d+"
        new_config_content = re.sub(pattern, f"num_classes:{maxid}", config_content)
        with open(self.config_file_path, "w") as config_file:
            config_file.write(new_config_content)
        self.info_label.setText(f"최대ID로 config 파일 구성이 업데이트 되었습니다.:{maxid}")

    def maxid(self):
        self.max_id = -1
        id_pattern = re.compile(r"id:\s*(\d+)")
        with open(self.labelmap_path, 'r') as labelmap:
            for line in labelmap:
                match = id_pattern.search(line)
                if match:
                    id_value = int(match.group(1))
                    if id_value > self.max_id:
                        self.max_id = id_value
        return self.max_id

    # def save_code(self):
    #     start_marker_text = "#start"
    #     end_marker_text = "#end"
    #     existing_content = []
    #     with open(self.config_file_path, "r") as config_file:
    #         existing_content = config_file.readlines()
    #     start_index = None
    #     end_index = None
    #     for i, line in enumerate(existing_content):
    #         if start_marker_text in line:
    #             start_index = i
    #         if end_marker_text in line:
    #             end_index = i
    #     # 예외 처리 추가_06.18
    #     if start_index is None or end_index is None:
    #         QMessageBox.warning(self, "Error", "#start 또는 #end 마커가 config 파일에 없습니다.")
    #         return
    #     updated_content = (
    #             existing_content[:start_index + 1]
    #             + [f"{aug_line}\n" for aug_line in self.aug_list]
    #             + existing_content[end_index:]
    #     )
    #     with open(self.config_file_path, "w") as config_file:
    #         config_file.writelines(updated_content)
    #     self.clear_aug_label.setText('File is updated!')
    #     self.aug_list = []
# ############################# Change Label ###########################_05.22추가
#     def change_label_run(self):
#         old_label = self.label_input_old.text().strip()
#         new_label = self.label_input_new.text().strip()

#         if not old_label or not new_label:
#             QMessageBox.warning(self, "입력 오류", "기존 라벨명과 변경할 라벨명을 모두 입력해주세요.")
#             return

#         xml_dir = r"D:\\AI_SVT_Training_mk\\annotations\\annos"
#         changed = 0
#         for file_name in os.listdir(xml_dir):
#             if file_name.endswith(".xml"):
#                 file_path = os.path.join(xml_dir, file_name)
#                 try:
#                     tree = ET.parse(file_path)
#                     root = tree.getroot()

#                     modified = False
#                     for obj in root.findall("object"):
#                         label = obj.find("name").text
#                         if label == old_label:
#                             obj.find("name").text = new_label
#                             modified = True
#                             changed += 1

#                     if modified:
#                         tree.write(file_path)

#                 except Exception as e:
#                     print(f"오류 발생: {file_name} - {e}", flush=True)

#         QMessageBox.information(self, "완료", f"라벨 변경 완료 ({changed}개 수정됨)")

############################# CLAHE ###########################
    def clahe_aug(self):
        # 05.07 추가
        total_files = len(self.img_folder_list)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        
        count = 0
        for i, img in enumerate(self.img_folder_list):
            image = cv2.imread(img, 0)
            base_name = os.path.basename(img).split('.')[0]
            clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
            cll = clahe.apply(image)
            os.makedirs(os.path.join(self.folder_path, self.folder_name_clahe), exist_ok=True)
            output_path = os.path.join(self.folder_path, self.folder_name_clahe, f"{base_name}.jpg")
            cv2.imwrite(output_path, cll)
            count += 1

            percent = int((i + 1) / total_files * 100)
            self.progress_bar.setValue(percent)
            QApplication.processEvents()

        self.progress_bar.setValue(100)
        self.clear_aug_label.setText('CLAHE 작업이 완료됐습니다.')
        QMessageBox.information(self, "처리 완료", f"총 {count}개의 CLAHE 작업이 완료됐습니다.")

        self.progress_bar.setValue(0)

        self.clear_aug_label.setText('CLAHE 폴더가 준비됐습니다.')

############################# Horizontal Flip ###########################
    def imgHflip(self):
        # 05.07 추가(ProgressBar)
        total_files = len(self.img_folder_list)
        self.progress_bar.setMaximum(100)       # 전체 개수 설정
        self.progress_bar.setValue(0)       # 초기화

        count = 0
        for idx, file in enumerate(self.img_folder_list):
            image = PILImage.open(file)
            base_name = os.path.basename(file).split('.')[0]
            flip_image = image.transpose(Transpose.FLIP_LEFT_RIGHT)
            os.makedirs(os.path.join(self.folder_path, self.folder_name_hf), exist_ok=True)
            flip_image.save(os.path.join(self.folder_path, self.folder_name_hf, f"{base_name}_HF.jpg"))
            count += 1
            percent = int((idx + 1) / total_files * 100)
            self.progress_bar.setValue(percent)  # 진행률 업데이트
            QApplication.processEvents()  # UI 업데이트
            # # 05.07 추가(ProgressBar)
            # self.progress_bar.setValue(idx + 1)
            # QApplication.processEvents()        

        self.progress_bar.setValue(100)
        self.clear_aug_label.setText('수평 뒤집기(H Flip) 작업이 완료됐습니다.')
        QMessageBox.information(self, "처리 완료", f"총 {count}개의 수평 뒤집기(H Flip) 작업이 완료됐습니다.")

        self.progress_bar.setValue(0)       # 완료 후 0으로 리셋

    def HFlip_bbox_coordinates(self, image_width, xmin, xmax):
        return int(image_width - xmin), int(image_width - xmax)

    def xmlHflip(self):
        for idx, xml_file in enumerate(self.xml_folder_list):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            image_width = int(root.find('size/width').text)
            for fln in root.iter('filename'):
                original_filename = fln.text
                new_name = f"_HF.jpg"
                modified_filename = original_filename.replace('.jpg', new_name)
                fln.text = modified_filename
            for path in root.iter('path'):
                original_path = path.text
                new_name = f"_HF.jpg"
                modified_path = original_path.replace('.jpg', new_name)
                path.text = modified_path
            for obj in root.iter('object'):
                for bbox in obj.iter('bndbox'):
                    xmin = int(bbox.find('xmin').text)
                    xmax = int(bbox.find('xmax').text)
                    new_xmin, new_xmax = self.HFlip_bbox_coordinates(image_width, xmin, xmax)
                    bbox.find('xmin').text = str(new_xmax)
                    bbox.find('xmax').text = str(new_xmin)

            base_name, extension = os.path.splitext(os.path.basename(xml_file))
            new_xml_name = f"{base_name}_HF.xml"
            output_xml_path = os.path.join(self.folder_path, self.folder_name_hf, new_xml_name)
            tree.write(output_xml_path)
        self.clear_aug_label.setText('수평 뒤집기(H Flip) 폴더가 준비됐습니다.')

############################# Vertical Flip ##########################
    def imgVflip(self):
        # 05.07 추가(ProgressBar)
        total_files = len(self.img_folder_list)
        self.progress_bar.setMaximum(100)   
        self.progress_bar.setValue(0)

        count = 0
        for idx, file in enumerate(self.img_folder_list):
            image = PILImage.open(file)
            base_name = os.path.basename(file).split('.')[0]
            flip_image = image.transpose(Transpose.FLIP_TOP_BOTTOM)
            os.makedirs(os.path.join(self.folder_path, self.folder_name_vf), exist_ok=True)
            flip_image.save(os.path.join(self.folder_path, self.folder_name_vf, f"{base_name}_VF.jpg"))
            count += 1
            percent = int((idx + 1) / total_files * 100)
            self.progress_bar.setValue(percent)  # 진행률 업데이트
            QApplication.processEvents()  # UI 업데이트

        self.progress_bar.setValue(100)
        self.clear_aug_label.setText('수직 뒤집기(V Flip) 작업이 완료됐습니다.')
        QMessageBox.information(self, "처리 완료", f"총 {count}개의 수직 뒤집기(V Flip) 작업이 완료됐습니다.")

        self.progress_bar.setValue(0)

    def VFlip_bbox_coordinates(self, image_height, ymin, ymax):
        return int(image_height - ymin), int(image_height - ymax)

    def xmlVflip(self):
        for idx, xml_file in enumerate(self.xml_folder_list):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            image_height = int(root.find('size/height').text)
            for fln in root.iter('filename'):
                original_filename = fln.text
                new_name = "_VF.jpg"
                modified_filename = original_filename.replace('.jpg', new_name)
                fln.text = modified_filename
            for path in root.iter('path'):
                original_path = path.text
                new_name = "_VF.jpg"
                modified_path = original_path.replace('.jpg', new_name)
                path.text = modified_path
            for obj in root.iter('object'):
                for bbox in obj.iter('bndbox'):
                    ymin = int(bbox.find('ymin').text)
                    ymax = int(bbox.find('ymax').text)
                    new_ymin, new_ymax = self.VFlip_bbox_coordinates(image_height, ymin, ymax)
                    bbox.find('ymin').text = str(new_ymax)
                    bbox.find('ymax').text = str(new_ymin)

            base_name, extension = os.path.splitext(os.path.basename(xml_file))
            new_xml_name = f"{base_name}_VF.xml"
            output_xml_path = os.path.join(self.folder_path, self.folder_name_vf, new_xml_name)
            tree.write(output_xml_path)
        self.clear_aug_label.setText('수직 뒤집기(V Flip) 폴더가 준비됐습니다.')

############################# Horizontal Shift ############################
    def imgHshift(self):
        # 05.07 추가
        total_files = len(self.img_folder_list) * len(self.num_shift)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        
        count = 0
        current = 0

        for i in self.num_shift:
            for j in range(len(self.img_folder_list)):
                img = self.img_folder_list[j]
                image = cv2.imread(img)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                width, height, _ = image.shape
                base_name = os.path.basename(img).split('.')[0]
                M = np.float32([[1, 0, i], [0, 1, 0]])
                shifted = cv2.warpAffine(image, M, (height, width))
                os.makedirs(os.path.join(self.folder_path, self.folder_name_hs), exist_ok=True)
                cv2.imwrite(os.path.join(self.folder_path, self.folder_name_hs, f"{base_name}_HS{i}.jpg"), shifted)
                count += 1

                current += 1
                percent = int(current / total_files * 100)

            # 05.07 추가
            self.progress_bar.setValue(percent)
            QApplication.processEvents()

        self.progress_bar.setValue(100)
        self.clear_aug_label.setText('수평 이동(H Shift) 작업이 완료됐습니다.')
        QMessageBox.information(self, "처리 완료", f"총 {count}개의 수평 이동(H Shift) 작업이 완료됐습니다.")

        self.progress_bar.setValue(0)

    def Hshift_bbox_coordinates(self, num_shift, xmin, xmax):
        return int(xmin + num_shift), int(xmax + num_shift)
    def xmlHshift(self, xml_file, i):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for fln in root.iter('filename'):
            original_filename = fln.text
            new_name = f"_HS{i}.jpg"
            modified_filename = original_filename.replace('.jpg', new_name)
            fln.text = modified_filename
        for path in root.iter('path'):
            original_path = path.text
            new_name = f"_HS{i}.jpg"
            modified_path = original_path.replace('.jpg', new_name)
            path.text = modified_path
        for obj in root.iter('object'):
            for bbox in obj.iter('bndbox'):
                xmin = int(bbox.find('xmin').text)
                xmax = int(bbox.find('xmax').text)
                new_xmin, new_xmax = self.Hshift_bbox_coordinates(i, xmin, xmax)
                bbox.find('xmin').text = str(new_xmin)
                bbox.find('xmax').text = str(new_xmax)

        base_name, extension = os.path.splitext(os.path.basename(xml_file))
        new_xml_name = f"{base_name}_HS{i}.xml"
        output_xml_path = os.path.join(self.folder_path, self.folder_name_hs, new_xml_name)
        tree.write(output_xml_path)
        self.clear_aug_label.setText('수평 이동(H Shift) 폴더가 준비됐습니다.')

    def xmlHshift_all(self):
        for xml_file in self.xml_folder_list:
            for i in self.num_shift:
                self.xmlHshift(xml_file, i)

############################# Vertical Shift ############################
    def imgVshift(self):
        # 05.07 추가
        total_files = len(self.img_folder_list) * len(self.num_shift)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)

        count = 0
        current = 0
        for i in self.num_shift:
            for j in range(len(self.img_folder_list)):
                img = self.img_folder_list[j]
                image = cv2.imread(img)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                width, height, _ = image.shape
                base_name = os.path.basename(img).split('.')[0]
                M = np.float32([[1, 0, 0], [0, 1, i]])
                shifted = cv2.warpAffine(image, M, (height, width))
                os.makedirs(os.path.join(self.folder_path, self.folder_name_vs), exist_ok=True)
                cv2.imwrite(os.path.join(self.folder_path, self.folder_name_vs, f"{base_name}_VS{i}.jpg"), shifted)
                count += 1

                current += 1
                percent = int(current / total_files * 100)
            # 05.07 추가
            self.progress_bar.setValue(percent)
            QApplication.processEvents()

        self.progress_bar.setValue(100)
        self.clear_aug_label.setText('수직 이동(V Shift) 작업이 완료됐습니다.')
        QMessageBox.information(self, "처리 완료", f"총 {count}개의 수직 이동(V Shift) 작업이 완료됐습니다.")

        self.progress_bar.setValue(0)

    def Vshift_bbox_coordinates(self, num_shift, ymin, ymax):
        return int(ymin + num_shift), int(ymax + num_shift)

    def xmlVshift(self, xml_file, i):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for fln in root.iter('filename'):
            original_filename = fln.text
            new_name = f"_VS{i}.jpg"
            modified_filename = original_filename.replace('.jpg', new_name)
            fln.text = modified_filename
        for path in root.iter('path'):
            original_path = path.text
            new_name = f"_VS{i}.jpg"
            modified_path = original_path.replace('.jpg', new_name)
            path.text = modified_path
        for obj in root.iter('object'):
            for bbox in obj.iter('bndbox'):
                ymin = int(bbox.find('ymin').text)
                ymax = int(bbox.find('ymax').text)
                new_ymin, new_ymax = self.Vshift_bbox_coordinates(i, ymin, ymax)
                bbox.find('ymin').text = str(new_ymin)
                bbox.find('ymax').text = str(new_ymax)

        base_name, extension = os.path.splitext(os.path.basename(xml_file))
        new_xml_name = f"{base_name}_VS{i}.xml"
        output_xml_path = os.path.join(self.folder_path, self.folder_name_vs, new_xml_name)
        tree.write(output_xml_path)
        self.clear_aug_label.setText('수직 이동(V Shift) 폴더가 준비됐습니다.')

    def xmlVshift_all(self):
        for xml_file in self.xml_folder_list:
            for i in self.num_shift:
                self.xmlVshift(xml_file, i)

############################# Row by Column ############################
    def imgRowColumn(self, img):
        self.rows = int(self.get_row_number())
        self.cols = int(self.get_col_number())
        image = cv2.imread(img)
        height, width, _ = image.shape
        sub_img_width = width // self.cols
        sub_img_height = height // self.rows
        cropped_images = []
        all = self.rows * self.cols
        for row in range(self.rows):
            for col in range(self.cols):
                x1 = col * sub_img_width
                y1 = row * sub_img_height
                x2 = (col + 1) * sub_img_width
                y2 = (row + 1) * sub_img_height
                sub_img = image[y1:y2, x1:x2]
                cropped_images.append(sub_img)
        rearranged_images = [cropped_images[i % len(cropped_images)] for i in range(1, all + 1, 1)]
        new_img = np.zeros_like(image)
        for row in range(self.rows):
            for col in range(self.cols):
                x1 = col * sub_img_width
                y1 = row * sub_img_height
                x2 = (col + 1) * sub_img_width
                y2 = (row + 1) * sub_img_height
                new_img[y1:y2, x1:x2] = rearranged_images.pop(0)
        base_name = os.path.basename(img).split('.')[0]
        os.makedirs(os.path.join(self.folder_path, self.folder_rbc), exist_ok=True)
        cv2.imwrite(os.path.join(self.folder_path, self.folder_rbc, f"{base_name}_arr.jpg"), new_img)

    def xmlRowColumn(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        image_height = int(root.find('size/height').text)
        image_width = int(root.find('size/width').text)
        for fln in root.iter('filename'):
            original_filename = fln.text
            new_name = f"_arr.jpg"
            modified_filename = original_filename.replace('.jpg', new_name)
            fln.text = modified_filename
        for path in root.iter('path'):
            original_path = path.text
            new_name = f"_arr.jpg"
            modified_path = original_path.replace('.jpg', new_name)
            path.text = modified_path
        for obj in root.iter('object'):
            for bbox in obj.iter('bndbox'):
                ymin = int(bbox.find('ymin').text)
                ymax = int(bbox.find('ymax').text)
                xmin = int(bbox.find('xmin').text)
                xmax = int(bbox.find('xmax').text)
                new_xmin, new_xmax, new_ymin, new_ymax = self.rowNcolumn_coordinates(image_width, image_height, xmin, xmax,
                                                                                ymin, ymax)
                bbox.find('ymin').text = str(new_ymin)
                bbox.find('ymax').text = str(new_ymax)
                bbox.find('xmin').text = str(new_xmin)
                bbox.find('xmax').text = str(new_xmax)
        self.clear_aug_label.setText('행렬 변환(Row by Column) 폴더가 준비됐습니다.')

        base_name, extension = os.path.splitext(os.path.basename(xml_file))
        new_xml_name = f"{base_name}_arr.xml"
        output_xml_path = os.path.join(self.folder_path, self.folder_rbc, new_xml_name)
        tree.write(output_xml_path)
        self.clear_aug_label.setText('회전(Rotation) 폴더가 준비됐습니다.')

    def rowNcolumn_coordinates(self, image_width, image_height, xmin, xmax, ymin, ymax):
        if xmin > int(image_width / self.cols):
            new_xmin = xmin - int(image_width / self.cols)
            new_xmax = xmax - int(image_width / self.cols)
            new_ymin = ymin
            new_ymax = ymax
            return int(new_xmin), int(new_xmax), int(new_ymin), int(new_ymax)
        elif ymin > int(image_height / self.rows) and xmin < int(image_width / self.cols):
            new_xmin = xmin + int(image_width / self.cols)
            new_xmax = xmax + int(image_width / self.cols)
            new_ymin = ymin - int(image_height / self.rows)
            new_ymax = ymax - int(image_height / self.rows)
            return int(new_xmin), int(new_xmax), int(new_ymin), int(new_ymax)
        elif ymin < int(image_height / self.rows) and xmin < int(image_width / self.cols):
            new_xmin = xmin + int((image_width / self.cols)*(self.cols-1))
            new_xmax = xmax + int((image_width / self.cols)*(self.cols-1))
            new_ymin = ymin + int((self.rows-1) * (image_height / self.rows))
            new_ymax = ymax + int((self.rows-1) * (image_height / self.rows))
            return int(new_xmin), int(new_xmax), int(new_ymin), int(new_ymax)

    def RowColumn_all(self):
        row_text = self.row_input.text()
        col_text = self.column_input.text()
        if not row_text or not col_text:
            QMessageBox.warning(self, "입력 오류", "행과 열의 숫자를 입력해주세요.")
            return
        total = len(self.img_folder_list) + len(self.xml_folder_list)
        current = 0
        for img in self.img_folder_list:
            self.imgRowColumn(img)
            current += 1
            progress = int(current / total * 100)

            self.progress_bar.setValue(progress)
            QApplication.processEvents()

        for xml_file in self.xml_folder_list:
            self.xmlRowColumn(xml_file)
            current += 1
            progress = int(current / total * 100)

            self.progress_bar.setValue(progress)
            QApplication.processEvents()

        # 06.18 추가
        self.progress_bar.setValue(100)     
        self.clear_aug_label.setText('행렬 변환(Row by Column) 작업이 완료됐습니다.')
        QMessageBox.information(self, "처리 완료", "행렬 변환(Row by Column) 작업이 완료됐습니다.")

        self.progress_bar.setValue(0)

############################# Rotation ############################
    def imgRotation(self):
        # 05.07 추가
        total_files = len(self.img_folder_list) * len(self.num_rot)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)

        count = 0
        current = 0

        for i in self.num_rot:
            for j in range(len(self.img_folder_list)):
                replace_path = self.img_folder_list[j]
                image = cv2.imread(replace_path)
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        # 이거 하면 R G B 중에서 R 이랑 B 랑 뒤집혀서 나옴(컬러일때)
                height, width = image.shape[:2]
                center_img = (width // 2, height // 2)
                base_name = os.path.basename(replace_path).split('.')[0]
                os.makedirs(os.path.join(self.folder_path, self.folder_name_rt), exist_ok=True)
                rotate_matrix = cv2.getRotationMatrix2D(center=center_img, angle=i, scale=1)
                rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
                cv2.imwrite(os.path.join(self.folder_path, self.folder_name_rt, f"{base_name}_RT{i}.jpg"), rotated_image)
                count += 1

                current += 1
                percent = int(current / total_files * 100)

            # 05.07 추가
            self.progress_bar.setValue(percent)
            QApplication.processEvents()

        self.progress_bar.setValue(100)
        self.clear_aug_label.setText('회전(Rotation) 작업이 완료됐습니다.')
        QMessageBox.information(self, "처리 완료", f"{count}개의 회전(Rotation) 작업이 완료됐습니다.")

        self.progress_bar.setValue(0)

    def xmlRotation_all(self):
        for xml_file in self.xml_folder_list:
            for i in self.num_rot:
                self.xmlRotation(xml_file, i)

    def rotate_vertices(self, vertices, angle, bbox_center, image_height, image_width):
        angle_rad = np.radians(- angle)
        x, y = vertices
        cx, cy = bbox_center
        new_x = (((x - cx) * np.cos(angle_rad)) - ((y - cy) * np.sin(angle_rad))) + cx
        new_y = (((x - cx) * np.sin(angle_rad)) + ((y - cy) * np.cos(angle_rad))) + cy
        if new_y < 0:
            new_y = 0
        elif new_y > image_height:
            new_y = image_height
        elif new_x > image_width:
            new_x = image_width
        elif new_x < 0:
            new_x = 0
        return new_x, new_y

    def maxNminDistance(self, rotated_vertices, bbox_center):
        max_x = max(rotated_vertices, key=lambda point: point[0])[0]
        max_y = max(rotated_vertices, key=lambda point: point[1])[1]
        dis_x = max_x - bbox_center[0]
        dis_y = max_y - bbox_center[1]
        return dis_x, dis_y

    def rotate_central_point(self, bbox_center, angle, image_center):
        x, y = bbox_center
        angle_rad = np.radians(360 - angle)
        translated_point = np.array([x - image_center[0], y - image_center[1]])
        rotated_x = (translated_point[0] * np.cos(angle_rad)) - (translated_point[1] * np.sin(angle_rad))
        rotated_y = (translated_point[0] * np.sin(angle_rad)) + (translated_point[1] * np.cos(angle_rad))
        new_x = rotated_x + image_center[0]
        new_y = rotated_y + image_center[1]
        return new_x, new_y

    def final_vertices(self, new_bbox_center, distance_from_center):
        new_xmin = new_bbox_center[0] - distance_from_center[0]
        new_ymin = new_bbox_center[1] - distance_from_center[1]
        new_xmax = new_bbox_center[0] + distance_from_center[0]
        new_ymax = new_bbox_center[1] + distance_from_center[1]
        return (int(new_xmin), int(new_ymin), int(new_xmax), int(new_ymax))

    def move_and_rotate_box(self, box_vertices, i, image_center, bbox_center, image_height, image_width):
        rotated_vertices = [self.rotate_vertices(vertex, i, bbox_center, image_height, image_width) for vertex in
                            box_vertices]
        distance_from_center = self.maxNminDistance(rotated_vertices, bbox_center)
        new_bbox_center = self.rotate_central_point(bbox_center, i, image_center)
        final_vertices_new_bbox = self.final_vertices(new_bbox_center, distance_from_center)
        return final_vertices_new_bbox

    def xmlRotation(self, xml_file, i):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        image_height = int(root.find('size/height').text)
        image_width = int(root.find('size/width').text)
        image_center = np.array([image_width / 2, image_height / 2])
        for fln in root.iter('filename'):
            original_filename = fln.text
            new_name = f"_RT{i}.jpg"
            modified_filename = original_filename.replace('.jpg', new_name)
            fln.text = modified_filename
        for path in root.iter('path'):
            original_path = path.text
            new_name = f"_RT{i}.jpg"
            modified_path = original_path.replace('.jpg', new_name)
            path.text = modified_path
        for obj in root.iter('object'):
            for bbox in obj.iter('bndbox'):
                xmin = bbox.find('xmin').text
                xmin = eval(xmin)
                ymin = bbox.find('ymin').text
                ymin = eval(ymin)
                xmax = bbox.find('xmax').text
                xmax = eval(xmax)
                ymax = bbox.find('ymax').text
                ymax = eval(ymax)
                box_vertices = np.array([[xmin, ymin], [xmax, ymin],
                                         [xmax, ymax], [xmin, ymax]
                                         ])
                bbox_center = np.mean(box_vertices, axis=0)
                result_vertices = self.move_and_rotate_box(box_vertices, i, image_center, bbox_center, image_height,
                                                      image_width)
                min_x = result_vertices[0]
                min_y = result_vertices[1]
                max_x = result_vertices[2]
                max_y = result_vertices[3]
                bbox.find('xmin').text = str(min_x)
                bbox.find('ymin').text = str(min_y)
                bbox.find('xmax').text = str(max_x)
                bbox.find('ymax').text = str(max_y)
        base_name, extension = os.path.splitext(os.path.basename(xml_file))
        new_xml_name = f"{base_name}_RT{i}.xml"
        output_xml_path = os.path.join(self.folder_path, self.folder_name_rt, new_xml_name)
        tree.write(output_xml_path)
############################## Color to Gray ###############################################
    def ColorToGray(self):
        os.makedirs(os.path.join(self.folder_path, self.folder_gray), exist_ok=True)
        os.makedirs(os.path.join(self.folder_path, self.folder_color), exist_ok=True)
        for filename in os.listdir(self.folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_path = os.path.join(self.folder_path, filename)
                img_color = cv2.imread(image_path)
                if img_color is not None:
                    img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
                    output_path_gray = os.path.join(self.folder_path, self.folder_gray, filename)
                    output_path_color = os.path.join(self.folder_path, self.folder_color, filename)
                    cv2.imwrite(output_path_gray, img_gray)
                    cv2.imwrite(output_path_color, img_color)
        self.clear_aug_label.setText('흑백 파일이 생성됩니다.')

############################## BGR to RGB ###############################################
    def BGRtoRGB(self):
        os.makedirs(os.path.join(self.folder_path, self.output_folder_rgb), exist_ok=True)
        os.makedirs(os.path.join(self.folder_path, self.output_folder_bgr), exist_ok=True)
        for filename in os.listdir(self.folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_path = os.path.join(self.folder_path, filename)
                img_bgr = cv2.imread(image_path)
                if img_bgr is not None:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    output_path_rgb = os.path.join(self.folder_path, self.output_folder_rgb, filename)
                    output_path_bgr = os.path.join(self.folder_path, self.output_folder_bgr, filename)
                    cv2.imwrite(output_path_rgb, img_rgb)
                    cv2.imwrite(output_path_bgr, img_bgr)
        self.clear_aug_label.setText('RGB 파일이 생성됩니다.')

############################## RGB to BGR ###############################################
    def RGBtoBGR(self):
        os.makedirs(os.path.join(self.folder_path, self.output_folder_rgb), exist_ok=True)
        os.makedirs(os.path.join(self.folder_path, self.output_folder_bgr), exist_ok=True)
        for filename in os.listdir(self.folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_path = os.path.join(self.folder_path, filename)
                img_rgb = cv2.imread(image_path)
                if img_rgb is not None:
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    output_path_rgb = os.path.join(self.folder_path, self.output_folder_rgb, filename)
                    output_path_bgr = os.path.join(self.folder_path, self.output_folder_bgr, filename)
                    cv2.imwrite(output_path_rgb, img_rgb)
                    cv2.imwrite(output_path_bgr, img_bgr)
        self.clear_aug_label.setText('BGR 파일이 생성됩니다.')

# ######################## XML 변경 (기존 XML 파일에 object 추가) _PJS #######################
    def transform_xml_boxes(self):
        folder_path = r"D:\AI_SVT_Training_mk\annotations\annos"
        mm_per_pixel = 0.03
        offset_mm = 0.15
        offset_px = int(offset_mm / mm_per_pixel)

        # 좌표 수정 함수 
        def modify_expand(bndbox, px):          # 확장
            bndbox.find('xmin').text = str(int(bndbox.find('xmin').text) - px)
            bndbox.find('xmax').text = str(int(bndbox.find('xmax').text) + px)
            bndbox.find('ymin').text = str(int(bndbox.find('ymin').text) - px)
            bndbox.find('ymax').text = str(int(bndbox.find('ymax').text) + px)

        def modify_shrink(bndbox, px):          # 축소
            bndbox.find('xmin').text = str(int(bndbox.find('xmin').text) + px)
            bndbox.find('xmax').text = str(int(bndbox.find('xmax').text) - px)
            bndbox.find('ymin').text = str(int(bndbox.find('ymin').text) + px)
            bndbox.find('ymax').text = str(int(bndbox.find('ymax').text) - px)

        def modify_left(bndbox, px):            # 좌로 이동
            bndbox.find('xmin').text = str(int(bndbox.find('xmin').text) - px)
            bndbox.find('xmax').text = str(int(bndbox.find('xmax').text) - px)

        def modify_right(bndbox, px):           # 우로 이동
            bndbox.find('xmin').text = str(int(bndbox.find('xmin').text) + px)
            bndbox.find('xmax').text = str(int(bndbox.find('xmax').text) + px)

        def modify_up(bndbox, px):           # 위로 이동_04.30추가
            bndbox.find('ymin').text = str(int(bndbox.find('ymin').text) - px)
            bndbox.find('ymax').text = str(int(bndbox.find('ymax').text) - px)

        def modify_down(bndbox, px):           # 아래로 이동_04.30추가
            bndbox.find('ymin').text = str(int(bndbox.find('ymin').text) + px)
            bndbox.find('ymax').text = str(int(bndbox.find('ymax').text) + px)

        # 함수 리스트
        transformations = [
            modify_expand,
            modify_left,
            modify_right,
            modify_shrink,
            modify_up,
            modify_down
        ]

         # ProgressBar
        xml_files = [file for file in os.listdir(folder_path) if file.endswith(".xml")]
        total_files = len(xml_files)
        self.progress_bar.setMaximum(total_files)   # 전체 개수 설정
        self.progress_bar.setValue(0)   # 초기화

        # 전체 XML 처리
        count = 0
        for idx, file in enumerate(xml_files):
            xml_path = os.path.join(folder_path, file)

            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                # 필터링 추가
                image_width = int(root.find("size/width").text)
                image_height = int(root.find("size/height").text)
                
                original_objects = root.findall('object')
                if not original_objects:
                    continue

                for obj in original_objects:
                    for transform_func in transformations:
                        new_obj = copy.deepcopy(obj)
                        bndbox = new_obj.find('bndbox')
                        transform_func(bndbox, offset_px)

                        # 유효성 검사 추가
                        xmin = int(bndbox.find('xmin').text)
                        xmax = int(bndbox.find('xmax').text)
                        ymin = int(bndbox.find('ymin').text)
                        ymax = int(bndbox.find('ymax').text)

                        if (xmin < 0 or xmax > image_width or ymin < 0 or ymax > image_height or
                            xmin >= xmax or ymin >= ymax):
                            continue

                        # if len(root.findall('object')) >= 10:
                        #     print(f"X {file} - object 10개 초과, 생략됨")
                        #     continue

                        root.append(new_obj)

                tree.write(xml_path, encoding='utf-8')
                count += 1
            except Exception as e:
                print(f"오류 발생: {file} - {e}", flush=True)

            # ProgressBar 갱신
            self.progress_bar.setValue(idx + 1)
            QApplication.processEvents()    # 이벤트 루프 업뎃 (프리징 방지)

        QMessageBox.information(self, "처리 완료", f"{count}개의 XML파일이 수정되었습니다.")

        self.progress_bar.setValue(0)   # 완료 후 0으로 리셋
        # self.progress_bar.setFormat("완료")     # 완료 표시

##################### Auto Label 조정(JPG, XML 따로따로 생성) _PJS ######################
    def adjust_bbox_xml(self, input_xml, output_xml, shift_x=0, shift_y=0, expand_x=0, expand_y=0):
        import shutil
        
        tree = ET.parse(input_xml)
        root = tree.getroot()

        new_filename = os.path.basename(output_xml).replace(".xml", ".jpg")

        # xml path ↠ annos 로 고정
        fixed_path = os.path.join("D:\\AI_SVT_Training_mk\\annotations\\annos", new_filename)

        # <filename> 태그 수정
        for fln in root.iter("filename"):
            fln.text = new_filename
        for path_tag in root.iter("path"):
            path_tag.text = fixed_path

        # <path> 태그 수정
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            xmax = int(bndbox.find('xmax').text)
            ymin = int(bndbox.find('ymin').text)    # 상, 하 추가
            ymax = int(bndbox.find('ymax').text)

            # 크기 확장/축소
            xmin -= expand_x
            xmax += expand_x
            ymin -= expand_y
            ymax += expand_y

            # 위치 이동
            xmin += shift_x
            xmax += shift_x
            ymin += shift_y
            ymax += shift_y

            xmin = max(0, xmin)
            xmax = max(xmin + 1, xmax)
            ymin = max(0, ymin)
            ymax = max(ymin + 1, ymax)

            bndbox.find('xmin').text = str(xmin)
            bndbox.find('xmax').text = str(xmax)
            bndbox.find('ymin').text = str(ymin)
            bndbox.find('ymax').text = str(ymax)

        # XML 저장
        tree.write(output_xml, encoding='utf-8')

        # 이미지 복사
        base_name = os.path.basename(input_xml).replace(".xml", "")
        original_img_path = os.path.join(self.folder_path, f"{base_name}.jpg")

        if os.path.exists(original_img_path):
            new_img_path = output_xml.replace(".xml", ".jpg")
            shutil.copyfile(original_img_path, new_img_path)

    def process_xml_variants(self):
        input_dir = self.folder_path
        output_base = os.path.join(input_dir, "augmented_mm")
        folder_path = r"D:\AI_SVT_Training_mk\annotations\annos"

        # 조정 설정: 폴더 이름 ↠ (이동값, 크기값)
        subfolders = {
            "expand":  (0, 0, 3, 3),     # shift_x, shift_y, expand_x, expand_y
            "shrink":  (0, 0, -3, -3),
            "left":    (-3, 0, 0, 0),
            "right":   (3, 0, 0, 0),
            "up":      (0, -3, 0, 0),
            "down":    (0, 3, 0, 0)
        }

        # ProgressBar
        xml_files = [file for file in os.listdir(folder_path) if file.endswith(".xml")]
        total_files = len(xml_files)
        self.progress_bar.setMaximum(total_files)   # 전체 개수 설정
        self.progress_bar.setValue(0)   # 초기화

        # 각 형태별 폴더 생성
        for sub in subfolders:
            os.makedirs(os.path.join(output_base, sub), exist_ok=True)

        count = 0
        for idx, image_file in enumerate(os.listdir(input_dir)):
            if image_file.lower().endswith(".jpg"):
                base_name = os.path.splitext(image_file)[0]
                xml_name = f"{base_name}.xml"
                xml_path = os.path.join(input_dir, xml_name)

                if not os.path.exists(xml_path):
                    continue    # XML 없으면 스킵

                for sub, (sx, sy, ex, ey) in subfolders.items():
                    output_xml_name = f"{base_name}_{sub}.xml"
                    output_path = os.path.join(output_base, sub, output_xml_name)
                    self.adjust_bbox_xml(xml_path, output_path,shift_x=sx, shift_y=sy, expand_x=ex, expand_y=ey)
                    count += 1
        
                    # ProgressBar 갱신
            self.progress_bar.setValue(idx + 1)
            QApplication.processEvents()    # 이벤트 루프 업뎃 (프리징 방지)

        self.clear_aug_label.setText(f"총 {count}개의 XML이 저장 완료되었습니다")
        QMessageBox.information(self, "처리 완료", f"{count}개의 XML파일이 수정되었습니다.")

        self.progress_bar.setValue(0)   # 완료 후 0으로 리셋
        # self.progress_bar.setFormat("완료")     # 완료 표시

    # 06.25 _annos 변경 건 자동갱신
    def on_annos_changed(self, path):
        # (1) 최신 이미지 개수 재계산
        try:
            num_images = len([
                f for f in os.listdir(self.annos_dir)
                if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))
            ])
        except Exception:
            num_images = 0
        self.num_images = num_images

        # (2) EpochNo.txt 업데이트
        self.update_epochno_file()

        # (3) UI 갱신
        if hasattr(self, 'info_label'):
            mul = self.bg_mul.id(self.bg_mul.checkedButton())
            # self.info_label.setText(f"현재 이미지: {num_images}, step = {num_images}×{mul}")
        # (4) 콘솔 로그
        # print(f"[annos_changed] num_images={num_images}", flush=True)        

    # 06.25 _EpochNo.txt 생성/덮어쓰기 로직 분리
    def update_epochno_file(self):
        """
        self.num_images 와 현재 선택된 mul 값을 이용해
        annotations/EpochNo.txt 를 쓰거나 덮어씁니다.
        """
        # 현재 mul 값 읽기 (10,20,30)
        if hasattr(self, 'bg_mul') and self.bg_mul.checkedButton():
            mul = self.bg_mul.id(self.bg_mul.checkedButton())
        else:
            mul = 20  # 기본값

        # 쓰기
        annotation_dir = r"D:\AI_SVT_Training_mk\annotations"
        os.makedirs(annotation_dir, exist_ok=True)
        epoch_file = os.path.join(annotation_dir, "EpochNo.txt")
        with open(epoch_file, "w", encoding="utf-8") as f:
            f.write(str(self.num_images * mul))
        # print(f"[EpochNo.txt] 업데이트 → {self.num_images}×{mul} = {self.num_images*mul}", flush=True)



def main():
    app = QApplication(sys.argv)
    viewer = ClassAugChanger()
    viewer.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    safe_qt_init()      # 06.09 추가
    main()

