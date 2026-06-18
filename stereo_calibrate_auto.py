import cv2 as cv
import numpy as np
import time

# === НАСТРОЙКИ ===
cam_id_1 = 4   # ID первой камеры
cam_id_2 = 2   # ID второй камеры

# Размер "внутренних углов" шахматки (НЕ клеток, а пересечений!)
# Например, если распечатал шахматку 9x6 углов -> pattern_size = (9, 6)
pattern_size = (8, 6)

# Размер одной клетки шахматки в метрах
square_size = 0.04  # 2.5 см

# === АВТОСОХРАНЕНИЕ ПАР ===
AUTO_SAVE = True
SAVE_DELAY_SEC = 2            # сколько секунд держать шахматку перед сохранением
MIN_POSE_CHANGE_PIXELS = 35.0   # насколько должна измениться поза шахматки для новой пары

# === ПОДГОТОВКА ОБЪЕКТНЫХ ТОЧЕК (3D в системе шахматки) ===
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []   # 3D точки (одни и те же для обеих камер)
imgpoints1 = []  # 2D в камере 1
imgpoints2 = []  # 2D в камере 2

pair_count = 0


def refine_corners(gray, corners):
    """Уточняет найденные углы шахматной доски."""

    criteria = (
        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    return cv.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        criteria,
    )


def make_pose_signature(corners1, corners2):
    """
    Делает численную подпись позы шахматки по всем углам с двух камер.
    Так мы понимаем, поменял ли пользователь положение доски.
    """

    pts1 = corners1.reshape(-1, 2).astype(np.float32)
    pts2 = corners2.reshape(-1, 2).astype(np.float32)
    return np.vstack([pts1, pts2])


def pose_changed(current_signature, last_saved_signature):
    """Проверяет, достаточно ли изменилась поза шахматки."""

    if last_saved_signature is None:
        return True, float("inf")

    if current_signature.shape != last_saved_signature.shape:
        return True, float("inf")

    diff = current_signature - last_saved_signature
    rms = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))

    return rms >= MIN_POSE_CHANGE_PIXELS, rms


def save_pair(gray1, gray2, corners1, corners2):
    """Сохраняет текущую пару углов для стереокалибровки."""

    global pair_count

    corners1_refined = refine_corners(gray1, corners1.copy())
    corners2_refined = refine_corners(gray2, corners2.copy())

    objpoints.append(objp.copy())
    imgpoints1.append(corners1_refined.copy())
    imgpoints2.append(corners2_refined.copy())

    pair_count += 1
    print(f"СОХРАНЁННАЯ ПАРА №{pair_count}")

    return corners1_refined, corners2_refined


def draw_status(vis, pair_count, status_text, auto_save=True):
    """Рисует служебную информацию на кадре."""

    cv.putText(
        vis,
        f"Pairs: {pair_count}",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv.LINE_AA,
    )

    cv.putText(
        vis,
        f"Auto: {'ON' if auto_save else 'OFF'}",
        (10, 65),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255) if auto_save else (0, 0, 255),
        2,
        cv.LINE_AA,
    )

    cv.putText(
        vis,
        status_text,
        (10, 100),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv.LINE_AA,
    )


# === ОТКРЫВАЕМ КАМЕРЫ ===
cap1 = cv.VideoCapture(cam_id_1)
cap2 = cv.VideoCapture(cam_id_2)

if not cap1.isOpened():
    print("НЕ МОГУ ОТКРЫТЬ КАМЕРУ 1")
    exit()
if not cap2.isOpened():
    print("НЕ МОГУ ОТКРЫТЬ КАМЕРУ 2")
    exit()

print("Управление:")
print("  A   - включить / выключить автосохранение")
print("  S   - сохранить текущую пару вручную, если шахматка найдена на обеих")
print("  ESC - запустить калибровку и выйти")
print("")
print("Автосохранение:")
print(f"  Если шахматка найдена на двух камерах, пара сохранится через {SAVE_DELAY_SEC:.1f} сек.")
print("  После сохранения передвинь / наклони шахматку для следующей пары.")

both_found_since = None
last_saved_signature = None
status_text = "WAITING FOR CHESSBOARD"
image_size = None

# Более устойчивые флаги поиска шахматки
find_flags = (
    cv.CALIB_CB_ADAPTIVE_THRESH
    + cv.CALIB_CB_NORMALIZE_IMAGE
    + cv.CALIB_CB_FAST_CHECK
)

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("ПРОБЛЕМА С ЧТЕНИЕМ КАДРОВ")
        break

    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    image_size = gray1.shape[::-1]

    # Ищем шахматку на обоих кадрах
    ret_c1, corners1 = cv.findChessboardCorners(gray1, pattern_size, find_flags)
    ret_c2, corners2 = cv.findChessboardCorners(gray2, pattern_size, find_flags)

    now = time.perf_counter()

    # ---------- АВТОСОХРАНЕНИЕ ----------
    if AUTO_SAVE and ret_c1 and ret_c2:
        current_signature = make_pose_signature(corners1, corners2)
        moved_enough, pose_delta = pose_changed(current_signature, last_saved_signature)

        if moved_enough:
            if both_found_since is None:
                both_found_since = now

            elapsed = now - both_found_since
            remaining = max(0.0, SAVE_DELAY_SEC - elapsed)
            status_text = f"AUTO SAVE IN {remaining:.1f}s"

            if elapsed >= SAVE_DELAY_SEC:
                corners1_refined, corners2_refined = save_pair(gray1, gray2, corners1, corners2)
                last_saved_signature = make_pose_signature(corners1_refined, corners2_refined)
                both_found_since = None
                status_text = "SAVED! MOVE CHESSBOARD"
        else:
            both_found_since = None
            status_text = f"MOVE CHESSBOARD  delta={pose_delta:.1f}px"

    else:
        both_found_since = None

        if ret_c1 and not ret_c2:
            status_text = "FOUND ONLY ON CAM1"
        elif ret_c2 and not ret_c1:
            status_text = "FOUND ONLY ON CAM2"
        else:
            status_text = "WAITING FOR BOTH CAMERAS"

    # ---------- ВИЗУАЛИЗАЦИЯ ----------
    vis1 = frame1.copy()
    vis2 = frame2.copy()

    if ret_c1:
        cv.drawChessboardCorners(vis1, pattern_size, corners1, ret_c1)
    if ret_c2:
        cv.drawChessboardCorners(vis2, pattern_size, corners2, ret_c2)

    draw_status(vis1, pair_count, status_text, AUTO_SAVE)
    draw_status(vis2, pair_count, status_text, AUTO_SAVE)

    cv.imshow("Cam1", vis1)
    cv.imshow("Cam2", vis2)

    key = cv.waitKey(1) & 0xFF

    if key == ord("a") or key == ord("A"):
        AUTO_SAVE = not AUTO_SAVE
        both_found_since = None
        print("Автосохранение:", "ON" if AUTO_SAVE else "OFF")

    elif key == ord("s") or key == ord("S"):
        if ret_c1 and ret_c2:
            corners1_refined, corners2_refined = save_pair(gray1, gray2, corners1, corners2)
            last_saved_signature = make_pose_signature(corners1_refined, corners2_refined)
            both_found_since = None
            status_text = "SAVED MANUALLY! MOVE CHESSBOARD"
        else:
            print("НЕ НАШЁЛ ШАХМАТКУ НА ОБЕИХ КАМЕРАХ - ПАРА НЕ СОХРАНЕНА")

    elif key == 27:  # ESC
        print("ВЫХОД И КАЛИБРОВКА...")
        break

cap1.release()
cap2.release()
cv.destroyAllWindows()

if pair_count < 5:
    print("СЛИШКОМ МАЛО ПАР ДЛЯ КАЛИБРОВКИ (нужно хотя бы 8–10).")
    exit()

if image_size is None:
    print("Не удалось получить размер изображения.")
    exit()

print("Запускаю стерео-калибровку на", pair_count, "парах...")

# Калибруем каждую камеру отдельно
ret1, K1, D1, rvecs1, tvecs1 = cv.calibrateCamera(
    objpoints,
    imgpoints1,
    image_size,
    None,
    None,
)

ret2, K2, D2, rvecs2, tvecs2 = cv.calibrateCamera(
    objpoints,
    imgpoints2,
    image_size,
    None,
    None,
)

print("Ошибка калибровки Cam1:", ret1)
print("Ошибка калибровки Cam2:", ret2)

# Стерео-калибровка
flags = cv.CALIB_USE_INTRINSIC_GUESS
criteria_stereo = (
    cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
    100,
    1e-5,
)

ret_stereo, K1, D1, K2, D2, R, T, E, F = cv.stereoCalibrate(
    objpoints,
    imgpoints1,
    imgpoints2,
    K1,
    D1,
    K2,
    D2,
    image_size,
    criteria=criteria_stereo,
    flags=flags,
)

print("Готово!")
print("Ошибка stereo calibration:", ret_stereo)
print("K1:\n", K1)
print("D1:\n", D1)
print("K2:\n", K2)
print("D2:\n", D2)
print("R:\n", R)
print("T:\n", T)

np.savez(
    "stereo_params.npz",
    K1=K1,
    D1=D1,
    K2=K2,
    D2=D2,
    R=R,
    T=T,
    E=E,
    F=F,
)

print("Параметры сохранены в stereo_params.npz")
