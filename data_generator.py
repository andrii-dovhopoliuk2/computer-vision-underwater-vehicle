"""Генерація синтетичних даних: імітація кадру з камери AUV над еталонною картою дна."""

import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class SyntheticFrame:
    """Результат генерації: кадр та істинна (ground truth) поза для перевірки."""

    frame: np.ndarray              # Зашумлений повернутий кадр (BGR)
    true_center: Tuple[float, float]  # Істинні (X, Y) центру кадру на карті, px
    true_angle_deg: float          # Істинний кут повороту кадру відносно карти, °


def load_map(map_path: str) -> np.ndarray:
    """Завантажує еталонну карту дна у форматі BGR."""
    image = cv2.imread(map_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Не вдалося завантажити карту: {map_path}")
    return image


def generate_frame(
    seaside_map: np.ndarray,
    frame_size: int = 400,
    max_angle_deg: float = 45.0,
    noise_sigma: float = 8.0,
    seed: Optional[int] = None,
) -> SyntheticFrame:
    """Вирізає випадковий кадр, повертає його на випадковий кут і додає гаусів шум.

    Щоб після повороту в кадрі не з'явилися чорні кути, спершу вирізається
    квадрат зі стороною frame_size * sqrt(2) (діагональ кадру), який
    повертається навколо центру, а потім з нього береться центральний
    фрагмент frame_size x frame_size.

    Args:
        seaside_map: Еталонна карта дна (BGR).
        frame_size: Сторона вихідного кадру, px.
        max_angle_deg: Максимальний модуль кута повороту, °.
        noise_sigma: Стандартне відхилення гаусового шуму (рівні яскравості 0..255).
        seed: Зерно генератора випадкових чисел (для відтворюваності).
    """
    rng = random.Random(seed)
    map_h, map_w = seaside_map.shape[:2]

    # Розмір "запасного" вікна, що гарантує відсутність порожніх кутів після повороту
    big = int(math.ceil(frame_size * math.sqrt(2.0)))
    if big > min(map_h, map_w):
        raise ValueError("Карта замала для вибраного розміру кадру.")

    # Випадковий лівий верхній кут вікна
    x0 = rng.randint(0, map_w - big)
    y0 = rng.randint(0, map_h - big)
    patch = seaside_map[y0:y0 + big, x0:x0 + big]

    # Випадковий кут. Поворот виконується навколо центру вікна.
    angle = rng.uniform(-max_angle_deg, max_angle_deg)
    center = (big / 2.0, big / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(patch, rot_mat, (big, big), flags=cv2.INTER_LINEAR)

    # Центральний кадр frame_size x frame_size
    off = (big - frame_size) // 2
    frame = rotated[off:off + frame_size, off:off + frame_size]

    # Гаусів шум: імітація каламутної води та шуму сенсора
    np_rng = np.random.default_rng(seed)
    noise = np_rng.normal(0.0, noise_sigma, frame.shape)
    noisy = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    true_center = (x0 + big / 2.0, y0 + big / 2.0)
    # Кут подається в тій самій конвенції, що й atan2(H[1,0], H[0,0]) у
    # navigator.py (система координат зображення, Y вниз); перевірено експериментально.
    return SyntheticFrame(frame=noisy, true_center=true_center, true_angle_deg=angle)
