import os
# os.chmod('D:\\AI_SVT_Training_mk\\annotations\\annos',0o777)    # 관리자 권한 실행

import shutil
import sys
import xml.etree.ElementTree as ET
import copy
from pathlib import Path

import re
from PIL import Image, ImageEnhance, ImageChops, ImageFilter
import cv2
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
import time
import shutil
import glob
import subprocess

# XML 파일들이 있는 폴더 경로
folder_path = r"D:\AI_SVT_Training_mk\annotations\annos"

# mm -> pixel 변환
mm_per_pixel = 0.03
offset_mm = 0.15
offset_px = int(offset_mm / mm_per_pixel)

# 좌표 수정 함수
def modify_expand(bndbox, px):
    bndbox.find('xmin').text = str(int(bndbox.find('xmin').text) - px)
    bndbox.find('xmax').text = str(int(bndbox.find('xmax').text) + px)
    bndbox.find('ymin').text = str(int(bndbox.find('ymin').text) - px)
    bndbox.find('ymax').text = str(int(bndbox.find('ymax').text) + px)

def modify_shrink(bndbox, px):
    bndbox.find('xmin').text = str(int(bndbox.find('xmin').text) + px)
    bndbox.find('xmax').text = str(int(bndbox.find('xmax').text) - px)
    bndbox.find('ymin').text = str(int(bndbox.find('ymin').text) + px)
    bndbox.find('ymax').text = str(int(bndbox.find('ymax').text) - px)

def modify_left(bndbox, px):
    bndbox.find('xmin').text = str(int(bndbox.find('xmin').text) - px)
    bndbox.find('xmax').text = str(int(bndbox.find('xmax').text) - px)

def modify_right(bndbox, px):
    bndbox.find('xmin').text = str(int(bndbox.find('xmin').text) + px)
    bndbox.find('xmax').text = str(int(bndbox.find('xmax').text) + px)
   
# 함수 리스트
transformations = [
    modify_expand,
    modify_left,
    modify_right,
    modify_shrink
]

# 전체 XML 처리
count = 0
for file in os.listdir(folder_path):
    if file.endswith(".xml"):
        xml_path = os.path.join(folder_path, file)

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            original_object = root.find('object')
            if original_object is None:
                continue
            
            for transform_func in transformations:
                new_obj = copy.deepcopy(original_object)
                bndbox = new_obj.find('bndbox')
                transform_func(bndbox, offset_px)
                root.append(new_obj)

            tree.write(xml_path, encoding='utf-8', xml_declaration=True)
            count += 1
        except Exception as e:
            print(f"오류 발생: {file} - {e}")

print(f"총 {count}개의 XML파일 처리 완료")






# 1. ClassAugChanger 클래스에 아래 메서드 추가:

# def transform_xml_boxes(self):
#     folder_path = r"D:\AI_SVT_Training_mk\annotations\annos"
#     mm_per_pixel = 0.03
#     offset_mm = 0.15
#     offset_px = int(offset_mm / mm_per_pixel)

#     def modify_expand(bndbox, px):
#         bndbox.find('xmin').text = str(int(bndbox.find('xmin').text) - px)
#         bndbox.find('xmax').text = str(int(bndbox.find('xmax').text) + px)
#         bndbox.find('ymin').text = str(int(bndbox.find('ymin').text) - px)
#         bndbox.find('ymax').text = str(int(bndbox.find('ymax').text) + px)

#     def modify_shrink(bndbox, px):
#         bndbox.find('xmin').text = str(int(bndbox.find('xmin').text) + px)
#         bndbox.find('xmax').text = str(int(bndbox.find('xmax').text) - px)
#         bndbox.find('ymin').text = str(int(bndbox.find('ymin').text) + px)
#         bndbox.find('ymax').text = str(int(bndbox.find('ymax').text) - px)

#     def modify_left(bndbox, px):
#         bndbox.find('xmin').text = str(int(bndbox.find('xmin').text) - px)
#         bndbox.find('xmax').text = str(int(bndbox.find('xmax').text) - px)

#     def modify_right(bndbox, px):
#         bndbox.find('xmin').text = str(int(bndbox.find('xmin').text) + px)
#         bndbox.find('xmax').text = str(int(bndbox.find('xmax').text) + px)

#     transformations = [
#         modify_expand,
#         modify_left,
#         modify_right,
#         modify_shrink
#     ]

#     count = 0
#     for file in os.listdir(folder_path):
#         if file.endswith(".xml"):
#             xml_path = os.path.join(folder_path, file)

#             try:
#                 tree = ET.parse(xml_path)
#                 root = tree.getroot()
#                 original_object = root.find('object')
#                 if original_object is None:
#                     continue

#                 for transform_func in transformations:
#                     new_obj = copy.deepcopy(original_object)
#                     bndbox = new_obj.find('bndbox')
#                     transform_func(bndbox, offset_px)
#                     root.append(new_obj)

#                 tree.write(xml_path, encoding='utf-8', xml_declaration=True)
#                 count += 1
#             except Exception as e:
#                 print(f"오류 발생: {file} - {e}")

#     QMessageBox.information(self, "처리 완료", f"{count}개의 XML 파일이 수정되었습니다.")

# 2. initUI() 메서드의 적절한 위치 (예: layout.addLayout(train_layout) 아래)에 버튼 추가:

# self.xml_transform_button = QPushButton("XML Box 변형", self)
# self.xml_transform_button.clicked.connect(self.transform_xml_boxes)
# layout.addWidget(self.xml_transform_button)
