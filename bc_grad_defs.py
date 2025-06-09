import numpy as np
import cv2
import math

def fish_rim_crop(img):
    """Удаление неинформативного окружения на обзорной камере fisheye для его исключения
    параметры эллипса оптимизированы под конкретную камеру"""

    height, width = img.shape[:2]

    # Создание маски с нулевыми значениями (чёрная маска)
    mask = np.zeros((height, width), dtype=np.uint8)

    # ПАРАМЕТРЫ эллипса (подобраны вручную под камеру):
    center = (
        int(width * 0.5),
        int(height * 0.52),
    )  # Центр эллипса (x, y координата центра)
    axes = (int(width * 0.6), int(height * 0.4))  # Полуоси эллипса (радиусы по x и y)
    angle = 0  # Угол поворота эллипса относительно оси X
    start_angle = 0  # Начальный угол дуги (полностью заполненный эллипс)
    end_angle = 360  # Конечный угол дуги (полностью заполненный эллипс)
    color = 255  # Цвет заполнения внутри эллипса (белый)
    thickness = -1  # Толщина линии (-1 означает заполнить всю фигуру)

    # Рисуем эллипс на маске
    cv2.ellipse(mask, center, axes, angle, start_angle, end_angle, color, thickness)

    # Применяем маску к изображению
    result = cv2.bitwise_and(img, img, mask=mask)
    return result


def bright_contrast_fit(img, low_percent=1, high_percent=99):
    """Автонастройка яркости и контраста изображения - необходимо для последующей обработкой.
    Работает растяжением диапазона от самых темных до светлых пикселей на шкалу 0-255"""

    # Конвертация в оттенки серого для анализа яркости
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Вычисление процентилей
    lo_val, hi_val = np.percentile(gray, [low_percent, high_percent])

    # Линейное преобразование
    scale = 255.0 / (hi_val - lo_val)
    offset = -lo_val * scale
    adjusted = img.astype(np.float32) * scale + offset
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return adjusted


def contours(
    closed, image, txt_path, sides_ratio_low, sides_ratio_high, size_frac_low, size_frac_high
):
    """Поиск контуров на бинарном (пороговом) изображении, сортировка контуров по их площади,
    перебор контуров, их выборка по критериям, их прорисовка и вывод координат"""   
    
    (cnts, _) = cv2.findContours(
        closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    height, width = image.shape[:2]
    file_info = ""

    if not cnts:
        return image
    else:
        cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
        # не обязательная процедура
        counter = 0
        for cnt in cnts_sorted:
            # вычисление повернутого ограничивающего прямоугольника;
            # minAreaRect: ((x_center, y_center), (width, height), angle)
            rect = cv2.minAreaRect(cnt)
            # поиск четырех вершин прямоугольника; boxPoints: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            box = np.int32(cv2.boxPoints(rect))
            # edge1 и edge2 - стороны прямоугольника - нужны для критерия выбора контуров
            edge1 = np.int32((box[1][0] - box[0][0], box[1][1] - box[0][1]))
            edge2 = np.int32((box[2][0] - box[1][0], box[2][1] - box[1][1]))

            usedEdge = edge1
            if cv2.norm(edge2) > cv2.norm(edge1):
                usedEdge = edge2

            # вычисляем угол между самой длинной стороной прямоугольника и горизонтом
            reference = (1, 0)  # горизонтальный вектор, задающий горизонт
            angle = (
                180.0
                / math.pi
                * math.acos(
                    (reference[0] * usedEdge[0] + reference[1] * usedEdge[1])
                    / (cv2.norm(reference) * cv2.norm(usedEdge))
                )
            )

            norm_edge1 = cv2.norm(edge1)
            norm_edge2 = cv2.norm(edge2)

            # сторон прямоугольника вокруг штрих-кода
            rect_sides_ratio = abs(
                (norm_edge1 - norm_edge2) / (norm_edge1 + norm_edge2)
            )

            

            # условие на размер штрих-кода относительно изображения
            size_fraction = (norm_edge1 + norm_edge2) / (height + width)

            cond_1 = sides_ratio_low < rect_sides_ratio < sides_ratio_high
            cond_2 = size_frac_low < size_fraction < size_frac_high
            cond_3 = 0 <= angle <= 50

            # # Собираем информацию о файле
            # file_info = f"Путь: {txt_path}\n"
        
            if cond_1 and cond_2 and cond_3:
                counter += 1
                cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
                center = (int(rect[0][0]), int(rect[0][1]))
                cv2.putText(
                    image,
                    "%d" % counter,
                    (center[0], center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )
                file_info += f"Координаты прямоугольника {counter}: ({center[0]}, {center[1]})\n"
            
        # Записываем информацию в текстовый файл
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(file_info)

    return image