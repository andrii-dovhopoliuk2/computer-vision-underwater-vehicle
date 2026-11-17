"""Точка входу: генерація кадру -> локалізація -> візуалізація."""

import cv2
import matplotlib.pyplot as plt

from data_generator import generate_frame, load_map
from navigator import TerrainNavigator

MAP_PATH = "zwoujkzwoujkzw.jpeg"


def main() -> None:
    seabed = load_map(MAP_PATH)
    sample = generate_frame(seabed)

    nav = TerrainNavigator(seabed)
    result = nav.locate(sample.frame)
    if result is None:
        print("Локалізація не вдалася: недостатньо збігів.")
        return

    print(f"Розраховано: X={result.x:.1f}, Y={result.y:.1f}, кут={result.angle_deg:.2f}°")
    print(f"Істина:      X={sample.true_center[0]:.1f}, Y={sample.true_center[1]:.1f}, "
          f"кут={sample.true_angle_deg:.2f}°")
    print(f"Збігів після тесту Лоу: {len(result.good_matches)}, інлайєрів: {result.num_inliers}")

    # Малюємо лише інлайєри RANSAC
    inliers = [m for m, ok in zip(result.good_matches, result.inlier_mask) if ok]
    match_img = cv2.drawMatches(
        sample.frame, result.kp_frame, seabed, result.kp_map, inliers, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    located = seabed.copy()
    cv2.polylines(located, [result.corners.astype("int32").reshape(-1, 1, 2)],
                  True, (0, 255, 0), 6)
    cv2.circle(located, (int(result.x), int(result.y)), 10, (0, 0, 255), -1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Збіги ключових точок (інлайєри RANSAC)")
    axes[1].imshow(cv2.cvtColor(located, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Положення дрона на карті")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
