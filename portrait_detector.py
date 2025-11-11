import os
from pathlib import Path

import cv2
from ultralytics import YOLO


def get_portrait(image_path, model_path=r'.\model\best.pt', conf_threshold=0.5):
    """
    Detect portraits in an image using YOLOv11 model.
    
    Args:
        image_path (str): path to the input image
        model_path (str): path to model checkpoint (default: .\model\best.pt)
        conf_threshold (float): confidence threshold (default: 0.5)
    
    Returns:
        dict: Result including:
            - success (bool): execution status
            - image (ndarray): original image
            - detections (list): list of detected objects with bbox, confidence, class_id, class_name
            - annotated_image (ndarray): image with annotations
            - num_detections (int): number of detected objects
    """
    try:
        # Kiểm tra file ảnh có tồn tại
        if not Path(image_path).exists():
            return {
                'success': False,
                'error': f'File is not exist: {image_path}',
                'image': None,
                'detections': [],
                'annotated_image': None
            }

        model = YOLO(model_path)

        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            return {
                'success': False,
                'error': f'Can not read image: {image_path}',
                'image': None,
                'detections': [],
                'annotated_image': None
            }

        # Run inference
        results = model(image, conf=conf_threshold)

        # get detections
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                    'confidence': float(box.conf[0]),
                    'class_id': int(box.cls[0]),
                    'class_name': model.names[int(box.cls[0])]
                }
                detections.append(detection)

        # draw annotated image
        annotated_image = results[0].plot()

        return {
            'success': True,
            'image': image,
            'detections': detections,
            'annotated_image': annotated_image,
            'num_detections': len(detections)
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'image': None,
            'detections': [],
            'annotated_image': None
        }


def crop_portraits(result, output_dir='outputs', class_filter='portrait'):
    """
    Crop and save portraits from detection results.
    
    Args:
        result (dict): result from get_portrait
        output_dir (str): directory to save cropped images (default: 'outputs')
        class_filter (str): class name to filter (default: 'portrait')
    
    Returns:
        dict: Kết quả bao gồm:
            - success (bool): result status
            - cropped_count (int): number of cropped portraits
            - saved_paths (list): list of saved file paths
    """
    try:
        if not result['success'] or result['image'] is None:
            return {
                'success': False,
                'error': 'Detection result is not successful.',
                'cropped_count': 0,
                'saved_paths': []
            }

        os.makedirs(output_dir, exist_ok=True)

        image = result['image']
        detections = result['detections']

        portrait_detections = [d for d in detections if d['class_name'] == class_filter]

        if len(portrait_detections) == 0:
            return {
                'success': True,
                'cropped_count': 0,
                'saved_paths': [],
                'message': f'Can not find "{class_filter}"'
            }

        # Only crop the portrait with the highest confidence
        best_portrait = max(portrait_detections, key=lambda d: d['confidence'])

        bbox = best_portrait['bbox']
        x1, y1, x2, y2 = map(int, bbox)

        cropped = image[y1:y2, x1:x2]

        conf = best_portrait['confidence']
        filename = f"portrait_conf{conf:.2f}.jpg"
        filepath = os.path.join(output_dir, filename)

        cv2.imwrite(filepath, cropped)

        return {
            'success': True,
            'cropped_count': 1,
            'saved_paths': [filepath]
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'cropped_count': 0,
            'saved_paths': []
        }


def save_result(result, output_path):
    """
    Save the annotated image to the specified path.
    
    Args:
        result (dict): result from get_portrait
        output_path (str): Path to save the annotated image.
    
    Returns:
        bool: True if saved successfully, False otherwise.
    """
    if result['success'] and result['annotated_image'] is not None:
        cv2.imwrite(output_path, result['annotated_image'])
        return True
    return False


def get_default_detect(image_path, conf_threshold=0.5, model_name='yolov8n.pt'):
    """
    Detect objects using YOLOv8 default pretrained model (COCO dataset)
    without using custom-trained weights.

    Args:
        image_path (str): path to the input image
        conf_threshold (float): confidence threshold (default: 0.5)
        model_name (str): YOLO model name (default: 'yolov8n.pt')

    Returns:
        dict: Result including:
            - success (bool): status
            - image (ndarray): original image
            - detections (list): list of objects (bbox, conf, class_id, class_name)
            - annotated_image (ndarray): image with bounding boxes
            - num_detections (int): number of detected objects
    """
    try:
        # Kiểm tra ảnh tồn tại
        if not Path(image_path).exists():
            return {
                'success': False,
                'error': f'File không tồn tại: {image_path}',
                'image': None,
                'detections': [],
                'annotated_image': None
            }

        # Load model YOLO mặc định
        model = YOLO(model_name)

        image = cv2.imread(image_path)
        if image is None:
            return {
                'success': False,
                'error': f'Không đọc được ảnh: {image_path}',
                'image': None,
                'detections': [],
                'annotated_image': None
            }

        # Dự đoán
        results = model(image, conf=conf_threshold)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),
                    'confidence': float(box.conf[0]),
                    'class_id': int(box.cls[0]),
                    'class_name': model.names[int(box.cls[0])]
                }
                detections.append(detection)

        annotated_image = results[0].plot()

        return {
            'success': True,
            'image': image,
            'detections': detections,
            'annotated_image': annotated_image,
            'num_detections': len(detections)
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'image': None,
            'detections': [],
            'annotated_image': None
        }


if __name__ == '__main__':
    image_path = 'D:/portrait-id-detect/images/cccd2.png'

    print('Detecting...')
    result = get_portrait(image_path)

    if result['success']:
        print(f"Detect {result['num_detections']} objects:")

        for i, det in enumerate(result['detections'], 1):
            print(f"  {i}. {det['class_name']} ({det['confidence']:.0%})")

        output_path = 'output_result.jpg'
        save_result(result, output_path)

        crop_result = crop_portraits(result, output_dir='outputs')
        if crop_result['cropped_count'] > 0:
            print(f"Cropped {crop_result['cropped_count']} portrait -> outputs/")
    else:
        print(f"Error: {result['error']}")
