from flask import Flask, jsonify
import os
import cv2
import numpy as np
from ultralytics import YOLO
import logging
import psycopg2
from psycopg2 import sql
from shapely import wkb

# Flask 애플리케이션 생성
app = Flask(__name__)

# 로그 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# YOLO 모델 로드
model = YOLO('mshamrai/yolov8n-visdrone')
logging.info("YOLO 모델 로드 완료.")

# PostgreSQL 연결 설정
conn_src = psycopg2.connect(
    host="localhost",
    database="nsu_db",
    user="postgres",
    password="postgres"
)


# 경로 설정
output_folder = r"C:\Users\NSU\Desktop\output"
os.makedirs(output_folder, exist_ok=True)


def get_images_from_db(conn):
    """PostgreSQL DB에서 이미지 경로 및 변환된 모서리 좌표(위경도) 가져오기"""
    with conn.cursor() as cur:
        cur.execute("SELECT id, image_path, top_left, top_right, bottom_left, bottom_right FROM upload")
        rows = cur.fetchall()
        images_data = []
        for row in rows:
            image_id, image_path, tl, tr, bl, br = row
            # 각 좌표를 Point로 변환 (Shapely)
            top_left = wkb.loads(tl, hex=True)
            top_right = wkb.loads(tr, hex=True)
            bottom_left = wkb.loads(bl, hex=True)
            bottom_right = wkb.loads(br, hex=True)
            images_data.append({
                "id": image_id,
                "path": image_path,
                "corners": [top_left, top_right, bottom_left, bottom_right]
            })
        return images_data


def save_result_to_db(conn, image_id, image_path, corners):
    """처리된 이미지 결과를 DB에 저장"""
    with conn.cursor() as cur:
        query = sql.SQL("""
            INSERT INTO processed_images (image_id, image_path, top_left, top_right, bottom_left, bottom_right)
            VALUES (%s, %s, ST_GeomFromText(%s, 5186), ST_GeomFromText(%s, 5186), ST_GeomFromText(%s, 5186), ST_GeomFromText(%s, 5186))
        """)
        cur.execute(query, [
            image_id,
            image_path,
            corners[0].wkt,
            corners[1].wkt,
            corners[2].wkt,
            corners[3].wkt
        ])
    conn.commit()


def apply_affine_transform(coord, affine_matrix):
    """Affine 변환 적용"""
    x, y = coord
    re_coord = np.dot(affine_matrix, np.array([x, y, 1]))
    return int(re_coord[0]), int(re_coord[1])


def process_images(images_data, output_folder, model):
    """이미지를 처리하고 탐지된 결과를 0으로 만듦"""
    if len(images_data) < 2:
        logging.error("처리할 이미지가 충분하지 않습니다.")
        return None

    # 첫 번째 이미지를 기준으로 사용
    reference_image_data = images_data[0]
    reference_image_path = reference_image_data['path']
    corners = reference_image_data['corners']

    # 이미지 로드
    reference_image = cv2.imread(reference_image_path)
    if reference_image is None:
        logging.error(f"{reference_image_path} 이미지를 읽을 수 없음.")
        return None

    # YOLOv8 모델로 객체 탐지
    results = model(reference_image)
    object_boxes = []  # 탐지된 BBox 저장 리스트

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1 = max(0, x1 - 25)
            y1 = max(0, y1 - 25)
            x2 = min(reference_image.shape[1], x2 + 25)
            y2 = min(reference_image.shape[0], y2 + 25)
            object_boxes.append([x1, y1, x2, y2])

    logging.info(f"{reference_image_path}에서 {len(object_boxes)}개의 객체를 탐지함.")

    # 탐지된 객체 부분을 0으로 설정 (컬러 이미지일 경우 [0, 0, 0]으로 설정)
    if object_boxes:
        for box in object_boxes:
            x1, y1, x2, y2 = box
            reference_image[y1:y2, x1:x2] = [0, 0, 0]  # 컬러 이미지의 경우 [0, 0, 0]으로 설정

    # 다른 이미지에서 0이 아닌 값으로 대체
    other_image_data = images_data[1]  # 다른 이미지를 가져옴
    other_image_path = other_image_data['path']
    other_image = cv2.imread(other_image_path)
    if other_image is None:
        logging.error(f"{other_image_path} 이미지를 읽을 수 없음.")
    else:
        # 0으로 채워진 영역의 좌표 찾기 및 대체
        zero_coords = np.argwhere(np.all(reference_image == [10, 10, 10], axis=-1))
        if len(zero_coords) == 0:
            logging.info("0으로 채워진 영역이 없음.")
        else:
            # 어핀 변환은 이미 계산된 ref_to_next_aff 사용
            # ref_to_next_aff 행렬은 이미지 처리 로직에 따라 정의되어야 함
            ref_to_next_aff = np.eye(3)  # Placeholder, 실제 어핀 변환 행렬로 대체해야 함
            for coord in zero_coords:
                y, x = coord

                # 어핀 변환 행렬(ref_to_next_aff)을 사용
                re_coord = np.dot(ref_to_next_aff, np.array([x, y, 1]))
                re_x, re_y = int(re_coord[0]), int(re_coord[1])

                # 다른 이미지의 값을 대입
                if 0 <= re_y < other_image.shape[0] and 0 <= re_x < other_image.shape[1]:
                    ref_pixel_value = other_image[re_y, re_x]
                    reference_image[y, x] = ref_pixel_value  # 다른 이미지의 값을 참조 이미지에 대입

    # 처리된 이미지 저장
    output_image_path = os.path.join(output_folder, os.path.basename(reference_image_path))
    cv2.imwrite(output_image_path, reference_image)  # Ensure image is of type ndarray
    logging.info(f"{output_image_path}에 처리된 이미지 저장 완료.")

    # 처리된 이미지 및 좌표 DB에 저장
    save_result_to_db(conn_src, reference_image_data['id'], output_image_path, corners)

    return output_image_path


@app.route('/flask-endpoint', methods=['POST'])
def flask_endpoint():
    """Flask 엔드포인트: 이미지 처리 요청을 처리하는 엔드포인트"""
    try:
        images_data = get_images_from_db(conn_src)
        if images_data:
            processed_image_path = process_images(images_data, output_folder, model)
            if processed_image_path:
                return jsonify({"message": "처리된 이미지 경로", "path": processed_image_path}), 200
            else:
                return jsonify({"message": "이미지 처리에 실패했습니다."}), 500
        else:
            return jsonify({"message": "데이터베이스에서 이미지를 가져오지 못했습니다."}), 500
    except Exception as e:
        logging.error(f"예외 발생: {str(e)}")
        return jsonify({"message": "서버 내부 오류", "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
