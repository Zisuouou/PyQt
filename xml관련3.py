import os
import xml.etree.ElementTree as ET
import copy

folder_path = r"D:\AI_SVT_Training_mk\annotations\annos"

# 복제 횟수
copies = 6      # 원본 + 복사 = 7개 라벨링 자동 완성

for file in os.listdir(folder_path):
    if file.endswith(".xml"):
        xml_path = os.path.join(folder_path, file)

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 원본 object 
            original_object = root.find('object')
            if original_object is None:
                print(f"object 없음: {file}")
                continue

            # 복사
            for _ in range(copies):
                new_obj = copy.deepcopy(original_object)
                root.append(new_obj)

            # 저장
            tree.write(xml_path, encoding='utf-8', xml_declaration=True)
            print(f"복제 완료: {file}")

        except Exception as e:
            print(f"오류발생: {file} - {e}")

print(f"모든 XML파일 처리 완료")
