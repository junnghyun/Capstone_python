from flask import Flask, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import logging
from typing import List, Tuple, Optional, Dict, Any
from geopy.distance import geodesic  # 지오디스틱 거리 계산 추가

app = Flask(__name__)

# YOLO 모델 로드
yolo_model = YOLO('mshamrai/yolov8n-visdrone')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# PostgreSQL 데이터베이스 연결 설정
database_url = 'postgresql://postgres:postgres@localhost:5432/nsu_db'


def get_image_data() -> Optional[Dict[str, Any]]:
    """
    데이터베이스에서 이미지 경로와 좌표를 가져오는 함수
    """
    conn = psycopg2.connect(database_url)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(
        "SELECT image_path, ST_AsText(top_left) as top_left, "
        "ST_AsText(top_right) as top_right, ST_AsText(bottom_left) as bottom_left, "
        "ST_AsText(bottom_right) as bottom_right FROM upload LIMIT 1"
    )
    data = cursor.fetchone()
    cursor.close()
    conn.close()
    return data


def affine_calculate(
    x1: float, y1: float, x2: float, y2: float, x3: float, y3: float,
    x1_target: float, y1_target: float, x2_target: float, y2_target: float,
    x3_target: float, y3_target: float
) -> np.ndarray:
    """
    아핀 변환 행렬을 계산하는 함수
    """
    img_array = np.array([[x1, y1], [x2, y2], [x3, y3]], dtype=np.float32)
    next_array = np.array([[x1_target, y1_target], [x2_target, y2_target], [x3_target, y3_target]], dtype=np.float32)
    return cv2.getAffineTransform(img_array, next_array)


def process_image(
    image_path: str, model: YOLO
) -> Tuple[Optional[np.ndarray], Optional[List[List[int]]]]:
    """
    이미지를 로드하고 YOLO 모델로 객체 탐지 후 객체가 있는 부분을 0으로 채우는 함수
    """
    img = cv2.imread(image_path)
    if img is None:
        logging.error(f"이미지를 읽을 수 없습니다: {image_path}")
        return None, None

    results = model(img)
    object_boxes = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1 = max(0, x1 - 25)
            y1 = max(0, y1 - 25)
            x2 = min(img.shape[1], x2 + 25)
            y2 = min(img.shape[0], y2 + 25)
            object_boxes.append([x1, y1, x2, y2])

    if object_boxes:
        for box in object_boxes:
            x1, y1, x2, y2 = map(int, box)
            img[y1:y2, x1:x2] = 0
        return img, object_boxes
    else:
        return img, None


def replace_pixels(
    reference_img: np.ndarray, processed_images: List[str],
    processed_folder: str, affine_matrix: np.ndarray,
    original_coords: List[Tuple[float, float]],  # 좌표 추가
    threshold_distance: float = 5.0  # 5미터 이내의 거리만 허용
) -> np.ndarray:
    """
    0으로 채워진 영역에 대해 다른 이미지에서 픽셀 값을 대입하는 함수, 5m 이내로 겹치는 영역만 처리
    """
    zero_coords = []

    # 첫 번째 이미지의 좌표 정보를 기준으로 픽셀 대체 작업
    for image_file in processed_images:
        processed_image_path = os.path.join(processed_folder, image_file)
        img = cv2.imread(processed_image_path)
        if img is None:
            logging.error(f"이미지를 읽을 수 없습니다: {processed_image_path}")
            continue

        coords = np.argwhere(np.all(reference_img == [0, 0, 0], axis=-1))
        for coord in coords:
            y, x = coord
            zero_coords.append((y, x))

    zero_coords = list(set(zero_coords))

    # 다른 이미지들과 비교하여 거리 계산
    for other_image_file in processed_images[1:]:
        other_image_path = os.path.join(processed_folder, other_image_file)
        other_image = cv2.imread(other_image_path)
        if other_image is None:
            logging.error(f"이미지를 읽을 수 없습니다: {other_image_path}")
            continue

        # 원래 좌표와의 거리 비교
        for coord in zero_coords:
            y, x = coord

            # 원래 이미지의 좌표와 비교할 좌표간의 거리를 계산합니다.
            # 이 부분은 이미지 상의 픽셀이 실제 공간의 좌표와 연결되어 있다고 가정합니다.
            re_coord = np.dot(affine_matrix, np.array([x, y, 1]))
            re_x, re_y = int(re_coord[0]), int(re_coord[1])

            # 현재 픽셀과 원래 좌표 간의 실제 거리를 계산
            distance = geodesic(original_coords[0], (re_x, re_y)).meters

            # 5미터 이내인지 확인 후 인페인팅 작업
            if distance <= threshold_distance:
                if 0 <= re_y < other_image.shape[0] and 0 <= re_x < other_image.shape[1]:
                    if 0 <= y < reference_img.shape[0] and 0 <= x < reference_img.shape[1]:
                        reference_img[y, x] = other_image[re_y, re_x]

    return reference_img


@app.route('/flask-endpoint', methods=['POST'])
def process_request() -> jsonify:
    """
    POST 요청을 처리하여 이미지 경로와 좌표를 데이터베이스에서 가져오고 이미지 처리 및 픽셀 교체를 수행하는 엔드포인트
    """
    try:
        # 데이터베이스에서 이미지 데이터 가져오기
        data = get_image_data()
        if not data:
            logging.error("데이터베이스에서 데이터를 찾을 수 없습니다.")
            return jsonify({"error": "No data found in the database"}), 404

        # 좌표 데이터 파싱
        image_path = data['image_path']
        coords = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
        points = {}
        for coord in coords:
            point = data[coord].replace('POINT(', '').replace(')', '').split()
            points[coord] = [float(p) for p in point]

        # 원래 좌표 저장
        original_coords = [(points['top_left'][0], points['top_left'][1]),
                           (points['top_right'][0], points['top_right'][1]),
                           (points['bottom_left'][0], points['bottom_left'][1]),
                           (points['bottom_right'][0], points['bottom_right'][1])]

        # 아핀 변환 행렬 계산
        x1, y1 = points['top_left']
        x2, y2 = points['top_right']
        x3, y3 = points['bottom_left']
        x1_target, y1_target = points['top_left']
        x2_target, y2_target = points['top_right']
        x3_target, y3_target = points['bottom_right']
        affine_matrix = affine_calculate(
            x1, y1, x2, y2, x3, y3, x1_target, y1_target, x2_target, y2_target, x3_target, y3_target
        )

        # 이미지 처리
        img, object_boxes = process_image(image_path, yolo_model)
        if img is None:
            logging.error("이미지 처리에 실패했습니다.")
            return jsonify({"error": "Failed to process image"}), 500

        # 처리된 이미지 저장
        processed_folder = 'processed_images'
        os.makedirs(processed_folder, exist_ok=True)
        processed_image_path = os.path.join(processed_folder, os.path.basename(image_path))
        cv2.imwrite(processed_image_path, img)

        # 픽셀 교체 작업 수행 (5미터 이내로 제한)
        reference_img = img.copy()
        processed_images = [os.path.basename(image_path)]
        result_image = replace_pixels(
            reference_img, processed_images, processed_folder, affine_matrix, original_coords
        )

        # 결과 이미지 저장
        output_folder = 'output_images'
        os.makedirs(output_folder, exist_ok=True)
        output_image_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_image_path, result_image)

        return jsonify({"message": "Image processing complete", "output_image_path": output_image_path})

    except Exception as e:
        logging.error(f"오류가 발생했습니다: {str(e)}")
        return jsonify({"error": "An internal error occurred"}), 500


if __name__ == '__main__':
    app.run('0.0.0.0', 5000)
