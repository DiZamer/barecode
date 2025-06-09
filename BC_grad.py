import numpy as np
import cv2
import os
from bc_grad_defs import fish_rim_crop, bright_contrast_fit, contours

input_dir = "/home/dizamer/MyPythonProjects/SMLRobotic/OpenCV/barcode/BC_grad/input_image_files/"
files = os.listdir(input_dir)
# file_name = "out-2025-03-18-14-58-42-cam1-dark.jpg"

for filename in files:
    # Полный путь к исходному файлу
    filepath = os.path.join(input_dir, filename)
    image = cv2.imread(filepath)

    # Пропускаем цикл, если директория, а не файл
    if os.path.isdir(filepath):
        continue

    # условие на применение обрезки краев по размеру изображений (fisheye 240 * 320)
    height, width = image.shape[:2]
    if height + width < 600:
        image = fish_rim_crop(image)

    image = bright_contrast_fit(image, 1, 99)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # вычисление величины градиента Sobel для изображений в направлениях x и y
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # вычтите градиент по оси y из градиента по оси x
    # показывает наиболее контрастные участки: линии штрих-кода, надписи и границы
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    if height + width < 600:  # ПАРАМЕТРЫ для fisheye distorted image
        # размытие и пороговое разделение изображения: три варианты на выбор
        # blurred = cv2.GaussianBlur(gradient, (7, 7), 0)
        # blurred = cv2.medianBlur(gradient, 7)
        blurred = cv2.blur(gradient, (6, 6))
        thresh_low = 200
        thresh_high = 255
        kernel_size_width = 3
        kernel_size_hight = 1
        erode_iter = 4
        dilate_iter = 2
        sides_ratio_low = 0.1  # соотношение сторон нижний предел (слишком квадрат)
        sides_ratio_high = 0.7  # соотношение сторон верхний предел (слишком вытянуты)
        size_frac_low = 0.05  # соотношение стороны штрих-кода и размера изображения
        size_frac_high = 0.2

    else:  # ПАРАМЕТРЫ для неискаженных изображений (не fisheye)
        # размытие и пороговое разделение изображения: три варианты на выбор
        # blurred = cv2.GaussianBlur(gradient, (7, 7), 0)
        blurred = cv2.medianBlur(gradient, 7)
        # blurred = cv2.blur(gradient, (8, 8))
        thresh_low = 225
        thresh_high = 255
        kernel_size_width = 27
        kernel_size_hight = 9
        erode_iter = 2
        dilate_iter = 4
        sides_ratio_low = 0.1  # соотношение сторон нижний предел (слишком квадрат)
        sides_ratio_high = 0.65  # соотношение сторон верхний предел (слишком вытянуты)
        size_frac_low = 0.05  # соотношение стороны штрих-кода и размера изображения
        size_frac_high = 0.5

    # бинаризация изображения по оттенку серого на белые и черные пиксели
    (_, thresh) = cv2.threshold(blurred, thresh_low, thresh_high, cv2.THRESH_BINARY)

    # построить замыкающее ядро ​​и применить его к пороговому изображению
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size_width, kernel_size_hight)
    )
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # выполнение эрозий - границы окрашиваются черным и уменьшают объект, удаление шума
    closed = cv2.erode(closed, kernel, iterations=erode_iter)
    # Выполнение дилатаций - оконтуривает белым границы - возвращает размер
    closed = cv2.dilate(closed, kernel, iterations=dilate_iter)

    output_dir = input_dir + "output_data/"
    os.makedirs(output_dir, exist_ok=True)
    txt_filename = os.path.splitext(filename)[0] + "_contours" + ".txt"
    txt_path = os.path.join(output_dir, txt_filename)

    # выделить контуры штрих-кодов и их координаты
    image = contours(
        closed,
        image,
        txt_path,
        sides_ratio_low,
        sides_ratio_high,
        size_frac_low,
        size_frac_high,
    )

    # отображение стадий обработки изображения
    new_image_filename = os.path.splitext(filename)[0] + "_contours" + ".png"
    # txt_path = os.path.join(output_dir, txt_filename)

    cv2.imwrite(os.path.join(output_dir, new_image_filename), image)

    # cv2.imshow("blurred", blurred)
    # cv2.imshow("gradient", gradient)
    # cv2.imshow("thresh", thresh)
    # cv2.imshow("closed", closed)
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
