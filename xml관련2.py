import os
import cv2
import xml.etree.ElementTree as ET

def draw_bboxes(image_path, xml_path, output_path, box_color=(0, 255, 0), text_color=(0, 0, 255)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지 불러올 수 없음: {image_path}")
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

        # # 색상 (얘는 굳이같음 깔끔해 보이지 않음)
        # if idx == 0:
        #     box_color = (255, 0, 0)     # 파랑 (라벨링 원본)
        # elif idx % 2 == 1:
        #     box_color = (0, 255, 0)     # 초록 (라벨링 복사)
        # else:
        #     box_color = (0, 0, 255)     # 빨강 (라벨링 복사)


        # 사각형 및 텍스트
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), box_color, 2)
        cv2.putText(image, f"{name}_{idx+1}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, text_color, 1)
    cv2.imwrite(output_path, image)
    print(f"시각화 완료: {output_path}")

def visualize_all_labels(image_dir, xml_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(xml_dir):
        if filename.endswith(".xml"):
            name = os.path.splitext(filename)[0]
            xml_path = os. path.join(xml_dir, filename)
            image_path = os.path.join(image_dir, f"{name}.jpg")

            if not os.path.exists(image_path):
                print(f"이미지가 없음: {image_path}")
                continue

            output_path = os.path.join(output_dir, f"{name}_labeled.jpg")
            draw_bboxes(image_path, xml_path, output_path)

# 실행
if __name__ == "__main__":
    image_folder = r"D:\AI_SVT_Training_mk\annotations\adjust_label"
    xml_folder = r"D:\AI_SVT_Training_mk\annotations\adjust_label\augmented_001mm"
    output_folder = r"D:\AI_SVT_Training_mk\annotations\visualized_001mm"

    visualize_all_labels(image_folder, xml_folder, output_folder)

