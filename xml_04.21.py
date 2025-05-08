### 01 ###

import os
import shutil

# 원본 xml 경로 설정
original_xml_path = r'D:\AI_SVT_Training_mk\annotations\annos\CAM1_test_001.xml'

# 복사할 개수
num_copies = 4

# 파일명 분리
file_dir = os.path.dirname(original_xml_path)
file_name = os.path.basename(original_xml_path)
base_name, ext = os.path.splitext(file_name)

# 복사
for i in range(1, num_copies + 1):
    new_name = f"{base_name}_copy{i}{ext}"
    new_path = os.path.join(file_dir, new_name)
    shutil.copyfile(original_xml_path, new_path)

print(f"{num_copies}개의 복사본이 생성되었습니다.")

# 폴더 내에 .xml 파일이 여러 개 있을 경우

import os

folder = r"D:\AI_SVT_Training_mk\annotations\annos"
for file in os.listdir(folder):
    if file.endswith(".xml"):
        xml_path = os.path.join(folder, file)
        tree = ET.parse(xml_path)
        ...

################################
import os
import xml.etree.ElementTree as ET

folder = r"D:\AI_SVT_Training_mk\annotations\annos"

for file in os.listdir(folder):
    xml_path = os.path.join(folder, file)
    
    # 폴더가 아닌 파일인지, .xml 확장자인지 확인
    if os.path.isfile(xml_path) and file.endswith(".xml"):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            print(f"✅ 처리 중: {file}")
        except Exception as e:
            print(f"❌ 오류 발생: {file} - {e}")



### 02 _ 동일 크기로 복사
import xml.etree.ElementTree as ET

# XML 파일 경로
xml_path = r"D:\AI_SVT_Training_mk\annotations\annos\000_test_Bubble.xml"

# XML 파싱
tree = ET.parse(xml_path)
root = tree.getroot()

# 기존 object 복제할 개수
copies = 3

# 기존 object 찾기 (맨 처음 것)
original_object = root.find('object')

# 복사 및 추가
for _ in range(copies):
    root.append(original_object)

# 파일 저장
tree.write(xml_path, encoding='utf-8', xml_declaration=True)

print("동일 객체 4개로 XML 수정 완료!")

### 002__경로 변경

import os
import xml.etree.ElementTree as ET
import copy

# XML 파일들이 있는 폴더 경로
folder_path = r"D:\AI_SVT_Training_mk\annotations\annos"

# 복제할 횟수
copies = 3

# 폴더 내 .xml 파일 순회
for file in os.listdir(folder_path):
    if file.endswith(".xml"):
        xml_path = os.path.join(folder_path, file)

        try:
            # XML 파싱
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 원본 object 찾기
            original_object = root.find('object')
            if original_object is None:
                print(f"❗ object 없음: {file}")
                continue

            # 복사해서 추가
            for _ in range(copies):
                new_obj = copy.deepcopy(original_object)
                root.append(new_obj)

            # 저장
            tree.write(xml_path, encoding='utf-8', xml_declaration=True)
            print(f"✅ 복제 완료: {file}")

        except Exception as e:
            print(f"❌ 오류 발생: {file} - {e}")

print("\n🎉 모든 XML 파일에 object 복제 완료!")


## 03_ 좌,우 크기 살짝 이동

import xml.etree.ElementTree as ET
import copy

# XML 파일 경로
xml_path = r"D:\AI_SVT_Training_mk\annotations\annos\000_test_Bubble.xml"

# XML 파싱
tree = ET.parse(xml_path)
root = tree.getroot()

# 원본 object 찾기
original_object = root.find('object')

# 이동할 픽셀 변화량 리스트 (복제 수만큼)
# 양수면 오른쪽 이동, 음수면 왼쪽 이동
offsets = [5, -5, 10]  # 3개 복제할 거니까 3개 지정

for offset in offsets:
    # deepcopy로 object 복제
    new_obj = copy.deepcopy(original_object)

    # bndbox 값 수정
    bndbox = new_obj.find('bndbox')
    xmin = int(bndbox.find('xmin').text)
    xmax = int(bndbox.find('xmax').text)

    # 좌우 이동
    bndbox.find('xmin').text = str(xmin + offset)
    bndbox.find('xmax').text = str(xmax + offset)

    # XML에 추가
    root.append(new_obj)

# 저장
tree.write(xml_path, encoding='utf-8', xml_declaration=True)
print("좌표 이동한 객체 3개 추가 완료!")


##04 상,하 이동

import xml.etree.ElementTree as ET
import copy

# XML 파일 경로
xml_path = r"D:\AI_SVT_Training_mk\annotations\annos\000_test_Bubble.xml"

# XML 파싱
tree = ET.parse(xml_path)
root = tree.getroot()

# 원본 object 찾기
original_object = root.find('object')

# 이동할 y 방향 픽셀 변화량 리스트
# 양수면 아래로, 음수면 위로 이동
offsets = [5, -5, 10]

for offset in offsets:
    # object 복제
    new_obj = copy.deepcopy(original_object)

    # bndbox 찾기
    bndbox = new_obj.find('bndbox')
    ymin = int(bndbox.find('ymin').text)
    ymax = int(bndbox.find('ymax').text)

    # 상하 이동
    bndbox.find('ymin').text = str(ymin + offset)
    bndbox.find('ymax').text = str(ymax + offset)

    # XML에 추가
    root.append(new_obj)

# 덮어쓰기 저장
tree.write(xml_path, encoding='utf-8', xml_declaration=True)
print("ymin/ymax 상하 이동된 object 3개 추가 완료!")


## 05 좌,우,상,하 동시 이동 (1pixel=0.01mm) 

import xml.etree.ElementTree as ET
import copy

# XML 경로
xml_path = r"D:\AI_SVT_Training_mk\annotations\annos\000_test_Bubble.xml"

# 픽셀 변환 기준
mm_per_pixel = 0.01  # 1픽셀 = 0.01mm

# mm 단위로 지정된 이동값들 (x_offset_mm, y_offset_mm)
offsets_mm = [
    (0.05, 0.05),    # → ↓
    (-0.05, -0.05),  # ← ↑
    (0.10, -0.05)    # → ↑
]

# XML 파싱
tree = ET.parse(xml_path)
root = tree.getroot()

# 원본 object
original_object = root.find('object')

# 이동
for x_mm, y_mm in offsets_mm:
    # mm → 픽셀 변환
    x_offset = int(x_mm / mm_per_pixel)
    y_offset = int(y_mm / mm_per_pixel)

    # 복제
    new_obj = copy.deepcopy(original_object)
    bndbox = new_obj.find('bndbox')

    # 좌표 추출
    xmin = int(bndbox.find('xmin').text)
    xmax = int(bndbox.find('xmax').text)
    ymin = int(bndbox.find('ymin').text)
    ymax = int(bndbox.find('ymax').text)

    # 변경
    bndbox.find('xmin').text = str(xmin + x_offset)
    bndbox.find('xmax').text = str(xmax + x_offset)
    bndbox.find('ymin').text = str(ymin + y_offset)
    bndbox.find('ymax').text = str(ymax + y_offset)

    # 추가
    root.append(new_obj)

# 저장
tree.write(xml_path, encoding='utf-8', xml_declaration=True)
print("mm 단위로 좌우/상하 이동된 객체 3개 추가 완료!")

### 06_ expand, left, right, shrink 라벨링박스 이동한거 (얘는 xml 에 이름 변경됨)

import xml.etree.ElementTree as ET
import copy

# XML 파일 경로
xml_path = r"D:\AI_SVT_Training_mk\annotations\annos\000_test_Bubble.xml"

# mm 기준 → 픽셀 변환
mm_per_pixel = 0.03

# 변형 정의: (변형 이름, 수정 함수)
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

# 변형 리스트: (라벨 이름 변경, 적용 함수)
transformations = [
    ("Bubble_expand", modify_expand),
    ("Bubble_shrink", modify_shrink),
    ("Bubble_left", modify_left),
    ("Bubble_right", modify_right)
]

# XML 파싱
tree = ET.parse(xml_path)
root = tree.getroot()
original_object = root.find('object')

# 픽셀 단위 이동량 (예: 0.15mm → 5픽셀)
offset_mm = 0.15
offset_px = int(offset_mm / mm_per_pixel)

# 변형 적용
for label_name, transform_func in transformations:
    new_obj = copy.deepcopy(original_object)
    
    # 라벨 이름 변경
    new_obj.find('name').text = label_name
    
    # 박스 조정
    bndbox = new_obj.find('bndbox')
    transform_func(bndbox, offset_px)

    # 추가
    root.append(new_obj)

# 저장
tree.write(xml_path, encoding='utf-8', xml_declaration=True)
print("✅ expand / shrink / left / right 버전 object 생성 완료!")

### 07_ expand, left, right, shrink 라벨링박스 이동한거(xml 에 이름 변경 X)

import xml.etree.ElementTree as ET
import copy

# XML 파일 경로
xml_path = r"D:\AI_SVT_Training_mk\annotations\annos\000_test_Bubble.xml"

# mm → 픽셀 변환 기준
mm_per_pixel = 0.03
offset_mm = 0.15
offset_px = int(offset_mm / mm_per_pixel)  # 예: 0.15mm → 5픽셀

# 변형 정의 함수들
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

# 변형 리스트: 함수만!
transformations = [
    modify_expand,
    modify_shrink,
    modify_left,
    modify_right
]

# XML 파싱
tree = ET.parse(xml_path)
root = tree.getroot()
original_object = root.find('object')

# 변형 적용
for transform_func in transformations:
    new_obj = copy.deepcopy(original_object)
    bndbox = new_obj.find('bndbox')
    transform_func(bndbox, offset_px)  # 이동
    root.append(new_obj)

# 저장
tree.write(xml_path, encoding='utf-8', xml_declaration=True)
print("✅ Bubble 이름 유지 + expand/shrink/left/right 좌표 수정 완료!")


### 08_ 시각화 visualize_objects.py
import os
import cv2
import xml.etree.ElementTree as ET

def draw_bboxes(image_path, xml_path, output_path, box_color=(0, 255, 0), text_color=(0, 0, 255)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for idx, obj in enumerate(root.findall('object')):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), box_color, 2)
        cv2.putText(image, f"{name}_{idx+1}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    cv2.imwrite(output_path, image)
    print(f"✅ 시각화 완료: {output_path}")

def visualize_all_labels(image_dir, xml_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(xml_dir):
        if filename.endswith(".xml"):
            name = os.path.splitext(filename)[0]
            xml_path = os.path.join(xml_dir, filename)
            image_path = os.path.join(image_dir, f"{name}.jpg")

            if not os.path.exists(image_path):
                print(f"이미지가 없습니다: {image_path}")
                continue

            output_path = os.path.join(output_dir, f"{name}_labeled.jpg")
            draw_bboxes(image_path, xml_path, output_path)

# 실행 예시
if __name__ == "__main__":
    image_folder = r"D:\AI_SVT_Training_mk\annotations\adjust_label"
    xml_folder = r"D:\AI_SVT_Training_mk\annotations\adjust_label\augmented_001mm"
    output_folder = r"D:\AI_SVT_Training_mk\annotations\visualized_001mm"

    visualize_all_labels(image_folder, xml_folder, output_folder)

## 09 << 05의 시각화

import cv2
import xml.etree.ElementTree as ET
import os

def draw_bboxes(image_path, xml_path, output_path, box_color=(0, 255, 0), text_color=(0, 0, 255)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for idx, obj in enumerate(root.findall('object')):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))

        # 사각형 그리기
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), box_color, 2)
        # 텍스트 이름 옆에 번호 붙이기
        cv2.putText(image, f"{name}_{idx+1}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    # 저장
    cv2.imwrite(output_path, image)
    print(f"✅ 시각화 이미지 저장 완료: {output_path}")

# 사용 예시
if __name__ == "__main__":
    base_dir = r"D:\AI_SVT_Training_mk\annotations\annos"
    filename = "000_test_Bubble"

    image_path = os.path.join(base_dir, f"{filename}.jpg")
    xml_path = os.path.join(base_dir, f"{filename}.xml")
    output_path = os.path.join(base_dir, f"{filename}_labeled.jpg")

    draw_bboxes(image_path, xml_path, output_path)

## 10 << 05의 컬러박스 시각화
import cv2
import xml.etree.ElementTree as ET
import os

def draw_bboxes(image_path, xml_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for idx, obj in enumerate(root.findall('object')):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))

        # 색상 설정
        if idx == 0:
            box_color = (255, 0, 0)   # 파랑 (Blue) - 원본
        elif idx % 2 == 1:
            box_color = (0, 255, 0)   # 초록 (Green) - 복제
        else:
            box_color = (0, 0, 255)   # 빨강 (Red) - 복제

        # 사각형 및 텍스트 표시
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), box_color, 2)
        cv2.putText(image, f"{name}_{idx+1}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

    # 이미지 저장
    cv2.imwrite(output_path, image)
    print(f"✅ 시각화 완료: {output_path}")

# 사용 예시
if __name__ == "__main__":
    base_dir = r"D:\AI_SVT_Training_mk\annotations\annos"
    filename = "000_test_Bubble"

    image_path = os.path.join(base_dir, f"{filename}.jpg")
    xml_path = os.path.join(base_dir, f"{filename}.xml")
    output_path = os.path.join(base_dir, f"{filename}_labeled_color.jpg")

    draw_bboxes(image_path, xml_path, output_path)

# TRAIN.py 에 추가
def auto_label_variants_inplace(self):
    xml_path = QFileDialog.getOpenFileName(self, "Select XML File", "", "XML Files (*.xml)")[0]
    if not xml_path:
        return

    mm_per_pixel = 0.03
    offset_mm = 0.15
    offset_px = int(offset_mm / mm_per_pixel)

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

    transformations = [
        modify_expand,
        modify_shrink,
        modify_left,
        modify_right
    ]

    tree = ET.parse(xml_path)
    root = tree.getroot()
    original_object = root.find('object')

    for transform_func in transformations:
        new_obj = copy.deepcopy(original_object)
        bndbox = new_obj.find('bndbox')
        transform_func(bndbox, offset_px)
        root.append(new_obj)

    tree.write(xml_path, encoding='utf-8', xml_declaration=True)
    QMessageBox.information(self, "완료", "라벨 복제 및 위치/크기 수정 완료!\n(XML에 object 4개 추가됨)")


# self.mm001_button = QPushButton('Auto Label XML 조정') 아래에 다음 라인을 추가
self.mm001_button.clicked.connect(self.auto_label_variants_inplace)

## 경로 변경
import os
import xml.etree.ElementTree as ET
import copy

# XML 있는 폴더 경로
folder_path = r"D:\AI_SVT_Training_mk\annotations\annos"

# mm → pixel 변환
mm_per_pixel = 0.03
offset_mm = 0.15
offset_px = int(offset_mm / mm_per_pixel)

# 좌표 수정 함수들
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
    modify_shrink,
    modify_left,
    modify_right
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
            print(f" 오류 발생: {file} - {e}")

print(f"총 {count}개의 XML 파일 처리 완료 (각 파일에 object 4개씩 추가됨)")



# 방법 2 예시
import xml.etree.ElementTree as ET

# 저장 방식
xml_str = ET.tostring(root, encoding='utf-8')
with open(xml_path, 'wb') as f:
    f.write(b"<?xml version='1.0' encoding='utf-8'?>\n")
    f.write(xml_str)



import os
import xml.etree.ElementTree as ET
import copy

folder_path = r"D:\AI_SVT_Training_mk\annotations\annos"
copies = 3

for file in os.listdir(folder_path):
    if file.endswith(".xml"):
        xml_path = os.path.join(folder_path, file)

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            original_object = root.find('object')
            if original_object is None:
                print(f"❗ object 없음: {file}")
                continue

            for _ in range(copies):
                new_obj = copy.deepcopy(original_object)
                root.append(new_obj)

            # 안전한 저장 방식
            xml_str = ET.tostring(root, encoding='utf-8')
            with open(xml_path, 'wb') as f:
                f.write(b"<?xml version='1.0' encoding='utf-8'?>\n")
                f.write(xml_str)

            print(f"✅ 복제 + 저장 완료: {file}")

        except Exception as e:
            print(f"❌ 오류 발생: {file} - {e}")

print("\n🎉 모든 XML 파일에 object 복제 + 안전 저장 완료!")



# class ClassAugChanger(QMainWindow): 추가 건

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




