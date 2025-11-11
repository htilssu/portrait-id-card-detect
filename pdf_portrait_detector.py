import os
from pathlib import Path

from pdf2image import convert_from_path

from portrait_detector import get_portrait, crop_portraits, save_result


def pdf_to_image(pdf_path, output_dir='temp', page_number=1, dpi=1200):
    """
    Convert trang đầu tiên của PDF sang ảnh
    
    Args:
        pdf_path (str): Đường dẫn đến file PDF
        output_dir (str): Thư mục lưu ảnh tạm (mặc định: 'temp')
        page_number (int): Số trang cần convert (mặc định: 1)
        dpi (int): Độ phân giải ảnh (mặc định: 300)
    
    Returns:
        dict: Kết quả bao gồm:
            - success (bool): Trạng thái thực hiện
            - image_path (str): Đường dẫn ảnh đã lưu
            - error (str): Thông báo lỗi nếu có
    """
    try:
        if not Path(pdf_path).exists():
            return {
                'success': False,
                'image_path': None,
                'error': f'File PDF không tồn tại: {pdf_path}'
            }

        os.makedirs(output_dir, exist_ok=True)

        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=page_number,
            last_page=page_number
        )

        if not images:
            return {
                'success': False,
                'image_path': None,
                'error': 'Không thể convert PDF sang ảnh'
            }

        pdf_name = Path(pdf_path).stem

        image_filename = f"{pdf_name}_page{page_number}.jpg"
        image_path = os.path.join(output_dir, image_filename)

        images[0].save(image_path, 'JPEG')

        return {
            'success': True,
            'image_path': image_path,
            'error': None
        }

    except Exception as e:
        return {
            'success': False,
            'image_path': None,
            'error': str(e)
        }


def process_pdf_portrait(pdf_path, output_dir='outputs', model_path='./model/best.pt',
                         conf_threshold=0.5, class_filter='portrait', dpi=300):
    """
    Xử lý PDF: Convert sang ảnh -> Detect portrait -> Cắt và lưu
    
    Args:
        pdf_path (str): Đường dẫn đến file PDF
        output_dir (str): Thư mục lưu kết quả (mặc định: 'outputs')
        model_path (str): Đường dẫn model YOLO
        conf_threshold (float): Ngưỡng confidence
        class_filter (str): Tên class cần filter (mặc định: 'portrait')
        dpi (int): Độ phân giải khi convert PDF
    
    Returns:
            - success (bool): Trạng thái thực hiện
            - pdf_path (str): Đường dẫn file PDF gốc
            - converted_image (str): Đường dẫn ảnh convert từ PDF
            - annotated_image (str): Đường dẫn ảnh đã đánh dấu
            - cropped_portraits (list): Danh sách đường dẫn portrait đã cắt
            - num_detections (int): Số lượng đối tượng phát hiện
            - error (str): Thông báo lỗi nếu có
    """
    try:
        print(f'Xử lý PDF: {pdf_path}')

        print('Converting PDF...')
        convert_result = pdf_to_image(pdf_path, output_dir='temp', dpi=dpi)

        if not convert_result['success']:
            return {
                'success': False,
                'pdf_path': pdf_path,
                'error': f"Lỗi convert PDF: {convert_result['error']}"
            }

        image_path = convert_result['image_path']
        print(f'✓ Converted -> {image_path}')

        print('Detecting portraits...')
        detect_result = get_portrait(image_path, model_path=model_path, conf_threshold=conf_threshold)

        if not detect_result['success']:
            return {
                'success': False,
                'pdf_path': pdf_path,
                'converted_image': image_path,
                'error': f"Lỗi detect portrait: {detect_result['error']}"
            }

        print(f'✓ Phát hiện {detect_result["num_detections"]} đối tượng')

        for i, det in enumerate(detect_result['detections'], 1):
            print(f'  {i}. {det["class_name"]} ({det["confidence"]:.0%})')

        pdf_name = Path(pdf_path).stem
        annotated_path = os.path.join(output_dir, f"{pdf_name}_annotated.jpg")
        os.makedirs(output_dir, exist_ok=True)

        save_result(detect_result, annotated_path)

        crop_result = crop_portraits(detect_result, output_dir=output_dir, class_filter=class_filter)

        if crop_result['success'] and crop_result['cropped_count'] > 0:
            print(f'✓ Đã cắt {crop_result["cropped_count"]} portrait -> {output_dir}/')

        print('✓ Hoàn thành!')

        return {
            'success': True,
            'pdf_path': pdf_path,
            'converted_image': image_path,
            'annotated_image': annotated_path,
            'cropped_portraits': crop_result['saved_paths'],
            'num_detections': detect_result['num_detections'],
            'num_portraits': crop_result['cropped_count']
        }

    except Exception as e:
        return {
            'success': False,
            'pdf_path': pdf_path,
            'error': str(e)
        }


if __name__ == '__main__':
    pdf_path = 'test-pdf.pdf'
    os.environ['PATH'] = "C:\\Users\\tolas\Downloads\Release-25.07.0-0\poppler-25.07.0\Library\\bin"

    result = process_pdf_portrait(pdf_path, output_dir='outputs', conf_threshold=0.5, dpi=300)

    if not result['success']:
        print(f'✗ Lỗi: {result["error"]}')
