import os
from pathlib import Path

import cv2
from ultralytics import YOLO


def get_portrait(image_path, model_path=r'.\model\best.pt', conf_threshold=0.5):
    """
    Detect portraits in an image using YOLOv11 model.
    Rotates image and checks if any rotation has fewer than 6 classes, skips it.
    
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
            - rotation_angle (int): angle used for detection (0, 90, 180, 270)
    """
    try:
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

        # Check if image is landscape (width > height) or portrait (height > width)
        height, width = image.shape[:2]

        if width > height:
            # Image is landscape, check 0° and 180° only
            print(f"Image is landscape ({width}x{height}), checking 0° and 180°...")
            rotation_angles = [0, 180]

            for angle in rotation_angles:
                # Rotate image
                if angle == 0:
                    rotated_image = image.copy()
                elif angle == 180:
                    rotated_image = cv2.rotate(image, cv2.ROTATE_180)

                results = model(rotated_image, conf=conf_threshold)

                # Get detections
                detections = []
                unique_classes = set()
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        class_id = int(box.cls[0])
                        unique_classes.add(class_id)
                        detection = {
                            'bbox': box.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                            'confidence': float(box.conf[0]),
                            'class_id': class_id,
                            'class_name': model.names[class_id]
                        }
                        detections.append(detection)

                num_classes = len(unique_classes)
                print(f"Rotation {angle}°: {num_classes} classes detected")

                if num_classes < 6:
                    print(f"Skipping rotation {angle}° (only {num_classes} classes)")
                    continue

                print(f"Using rotation {angle}° with {num_classes} classes")
                annotated_image = results[0].plot()

                return {
                    'success': True,
                    'image': rotated_image,
                    'detections': detections,
                    'annotated_image': annotated_image,
                    'num_detections': len(detections),
                    'rotation_angle': angle
                }

            # If all rotations have < 6 classes, return last rotation result
            print(f"All rotations have < 6 classes. Using last rotation (180°)")
            annotated_image = results[0].plot()

            return {
                'success': True,
                'image': rotated_image,
                'detections': detections,
                'annotated_image': annotated_image,
                'num_detections': len(detections),
                'rotation_angle': 180
            }
        else:
            # Image is portrait, rotate 90° or 270° to make it landscape
            print(f"Image is portrait ({width}x{height}), checking 90° and 270°...")
            rotation_angles = [90, 270]

            for angle in rotation_angles:
                # Rotate image
                if angle == 90:
                    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 270:
                    rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # Run inference on rotated image
                results = model(rotated_image, conf=conf_threshold)

                # Get detections
                detections = []
                unique_classes = set()
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        class_id = int(box.cls[0])
                        unique_classes.add(class_id)
                        detection = {
                            'bbox': box.xyxy[0].cpu().numpy().tolist(),
                            'confidence': float(box.conf[0]),
                            'class_id': class_id,
                            'class_name': model.names[class_id]
                        }
                        detections.append(detection)

                num_classes = len(unique_classes)
                print(f"Rotation {angle}°: {num_classes} classes detected")

                if num_classes < 6:
                    print(f"Skipping rotation {angle}° (only {num_classes} classes)")
                    continue

                print(f"Using rotation {angle}° with {num_classes} classes")
                annotated_image = results[0].plot()

                return {
                    'success': True,
                    'image': rotated_image,
                    'detections': detections,
                    'annotated_image': annotated_image,
                    'num_detections': len(detections),
                    'rotation_angle': angle
                }

            # If all rotations have < 6 classes, return last rotation result
            print(f"All rotations have < 6 classes. Using last rotation (270°)")
            annotated_image = results[0].plot()

            return {
                'success': True,
                'image': rotated_image,
                'detections': detections,
                'annotated_image': annotated_image,
                'num_detections': len(detections),
                'rotation_angle': 270
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


if __name__ == '__main__':
    image_path = 'D:\\portrait-id-detect\\images\\cccd2.png'

    print('Detecting...')
    result = get_portrait(image_path)

    if result['success']:
        print(f"Detect {result['num_detections']} objects:")
        if 'rotation_angle' in result:
            print(f"Rotation angle used: {result['rotation_angle']}°")

        for i, det in enumerate(result['detections'], 1):
            print(f"  {i}. {det['class_name']} ({det['confidence']:.0%})")

        output_path = 'output_result.jpg'
        save_result(result, output_path)

        crop_result = crop_portraits(result, output_dir='outputs')
        if crop_result['cropped_count'] > 0:
            print(f"Cropped {crop_result['cropped_count']} portrait -> outputs/")
    else:
        print(f"Error: {result['error']}")
