"""Візуальна навігація: пошук кадру на еталонній карті (ORB + BFMatcher + RANSAC)."""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class NavigationResult:
    """Результат локалізації кадру на карті."""

    x: float                    # X центру кадру на карті, px
    y: float                    # Y центру кадру на карті, px
    angle_deg: float            # Кут повороту кадру відносно карти, °
    homography: np.ndarray      # Матриця гомографії 3x3 (кадр -> карта)
    corners: np.ndarray         # Кути кадру на карті, форма (4, 2)
    good_matches: List[cv2.DMatch]  # Збіги після тесту Лоу
    inlier_mask: np.ndarray     # Маска інлайєрів RANSAC (N,)
    kp_frame: Sequence[cv2.KeyPoint]
    kp_map: Sequence[cv2.KeyPoint]

    @property
    def num_inliers(self) -> int:
        return int(self.inlier_mask.sum())


class TerrainNavigator:
    """Знаходить положення та орієнтацію кадру на великій еталонній карті."""

    def __init__(
        self,
        reference_map: np.ndarray,
        n_features: int = 5000,
        ratio: float = 0.75,
        ransac_thresh: float = 5.0,
        min_matches: int = 10,
        min_inliers: int = 15,
    ) -> None:
        """
        Args:
            reference_map: Еталонна карта (BGR).
            n_features: Максимум ключових точок ORB.
            ratio: Поріг тесту Лоу (менше = суворіше).
            ransac_thresh: Допустима похибка репроєкції для інлайєра, px.
            min_matches: Мінімум збігів для спроби розрахунку гомографії.
            min_inliers: Мінімум інлайєрів RANSAC для прийняття результату.
        """
        self.reference_map = reference_map
        self.ratio = ratio
        self.ransac_thresh = ransac_thresh
        self.min_matches = min_matches
        self.min_inliers = min_inliers

        self.orb = cv2.ORB_create(nfeatures=n_features)
        # ORB дає бінарні дескриптори -> відстань Хеммінга. crossCheck вимкнено,
        # бо для тесту Лоу потрібні k=2 найближчі сусіди.
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Дескриптори карти обчислюємо один раз
        self.kp_map, self.des_map = self.orb.detectAndCompute(
            self._to_gray(reference_map), None
        )

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    def _match(self, des_frame: np.ndarray) -> List[cv2.DMatch]:
        """Зіставлення дескрипторів + тест співвідношення Лоу.

        Збіг приймається, якщо відстань до найближчого сусіда суттєво менша,
        ніж до другого: d1 < ratio * d2. Це відсікає неоднозначні збіги
        на повторюваних текстурах дна.
        """
        knn = self.matcher.knnMatch(des_frame, self.des_map, k=2)
        good: List[cv2.DMatch] = []
        for pair in knn:
            if len(pair) == 2 and pair[0].distance < self.ratio * pair[1].distance:
                good.append(pair[0])
        return good

    def locate(self, frame: np.ndarray) -> Optional[NavigationResult]:
        """Локалізує кадр на карті. Повертає None, якщо збігів недостатньо."""
        kp_frame, des_frame = self.orb.detectAndCompute(self._to_gray(frame), None)
        if des_frame is None or self.des_map is None:
            return None

        good = self._match(des_frame)
        if len(good) < self.min_matches:
            return None

        # Координати точок: джерело — кадр, призначення — карта
        src = np.float32([kp_frame[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([self.kp_map[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Гомографія H (3x3) задає проєктивне перетворення:
        #   [x', y', w]^T = H * [x, y, 1]^T,  (X, Y) = (x'/w, y'/w).
        # RANSAC: ітеративно (1) випадково обирає 4 пари точок (мінімум для
        # розв'язку H з 8 ступенями свободи), (2) оцінює H, (3) рахує кількість
        # інлайєрів — точок, у яких похибка репроєкції ||H*src - dst|| менша за
        # ransac_thresh. Обирається H з найбільшою підтримкою, після чого вона
        # уточнюється за всіма інлайєрами методом найменших квадратів.
        # Так викиди (хибні збіги) не впливають на результат.
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_thresh)
        if H is None or mask is None:
            return None
        mask = mask.ravel().astype(np.uint8)

        # Мало інлайєрів — рішення ненадійне (імовірно, випадкова гомографія)
        if int(mask.sum()) < self.min_inliers:
            return None

        x, y, angle, corners = self._pose_from_homography(H, frame.shape[:2])
        return NavigationResult(
            x=x, y=y, angle_deg=angle, homography=H, corners=corners,
            good_matches=good, inlier_mask=mask,
            kp_frame=kp_frame, kp_map=self.kp_map,
        )

    @staticmethod
    def _pose_from_homography(
        H: np.ndarray, frame_shape: Tuple[int, int]
    ) -> Tuple[float, float, float, np.ndarray]:
        """Виділяє координати центру, кут та кути рамки з гомографії.

        Центр кадру (w/2, h/2) проєктується через H (перспективне ділення на w').
        Кут повороту — це кут образу осі X кадру на карті. Для перетворення,
        близького до подібності, ліва верхня 2x2 підматриця H ≈ s*R(θ), де
        R = [[cosθ, -sinθ], [sinθ, cosθ]], тож θ = atan2(H[1,0], H[0,0]).
        """
        h, w = frame_shape
        center = np.float32([[[w / 2.0, h / 2.0]]])
        cx, cy = cv2.perspectiveTransform(center, H)[0, 0]

        corners_src = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        corners = cv2.perspectiveTransform(corners_src, H).reshape(4, 2)

        angle = math.degrees(math.atan2(H[1, 0], H[0, 0]))
        return float(cx), float(cy), angle, corners
