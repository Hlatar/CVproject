import cv2 as cv
import numpy as np

# === НАСТРОЙКИ ===
cam_id_1 = 4   # ID первой камеры
cam_id_2 = 2   # ID второй камеры

# Размер "внутренних углов" шахматки (НЕ клеток, а пересечений!)
# Например, если распечатал шахматку 9x6 углов -> pattern_size = (9, 6)
pattern_size = (9, 6)

# Размер одной клетки шахматки в метрах (если не знаешь - поставь 1.0,
# тогда 3D будет "в условных единицах")
square_size = 0.025  # 2.5 см, пример. Можешь поменять или поставить 1.0

# === ПОДГОТОВКА ОБЪЕКТНЫХ ТОЧЕК (3D в системе шахматки) ===
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []   # 3D точки (одни и те же для обеих камер)
imgpoints1 = []  # 2D в камере 1
imgpoints2 = []  # 2D в камере 2

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
print("  S  - сохранить текущую пару кадров (если шахматка найдена на обеих)")
print("  ESC - запустить калибровку и выйти")

pair_count = 0

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        print("ПРОБЛЕМА С ЧТЕНИЕМ КАДРОВ")
        break

    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    # Ищем шахматку на обоих кадрах
    ret_c1, corners1 = cv.findChessboardCorners(gray1, pattern_size, None)
    ret_c2, corners2 = cv.findChessboardCorners(gray2, pattern_size, None)

    vis1 = frame1.copy()
    vis2 = frame2.copy()

    if ret_c1:
        cv.drawChessboardCorners(vis1, pattern_size, corners1, ret_c1)
    if ret_c2:
        cv.drawChessboardCorners(vis2, pattern_size, corners2, ret_c2)

    cv.putText(vis1, f"Pairs: {pair_count}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.putText(vis2, f"Pairs: {pair_count}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow("Cam1", vis1)
    cv.imshow("Cam2", vis2)

    key = cv.waitKey(1) & 0xFF

    if key == ord('s') or key == ord('S'):
        # Сохраняем только если шахматка найдена на обеих камерах
        if ret_c1 and ret_c2:
            # Уточняем углы для большей точности
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints1.append(corners1)
            imgpoints2.append(corners2)

            pair_count += 1
            print(f"СОХРАНЁННАЯ ПАРА №{pair_count}")
        else:
            print("НЕ НАШЁЛ ШАХМАТКУ НА ОБЕИХ КАМЕРАХ - ПАРА НЕ СОХРАНЕНА")

    if key == 27:  # ESC
        print("ВЫХОД И КАЛИБРОВКА...")
        break

cap1.release()
cap2.release()
cv.destroyAllWindows()

if pair_count < 5:
    print("СЛИШКОМ МАЛО ПАР ДЛЯ КАЛИБРОВКИ (нужно хотя бы 8–10).")
    exit()

print("Запускаю стерео-калибровку на", pair_count, "парах...")

# Калибруем каждую камеру отдельно (можно и до этого, но тут проще)
ret1, K1, D1, rvecs1, tvecs1 = cv.calibrateCamera(
    objpoints, imgpoints1, gray1.shape[::-1], None, None
)
ret2, K2, D2, rvecs2, tvecs2 = cv.calibrateCamera(
    objpoints, imgpoints2, gray2.shape[::-1], None, None
)

# Стерео-калибровка
flags = cv.CALIB_FIX_INTRINSIC
criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

ret_stereo, K1, D1, K2, D2, R, T, E, F = cv.stereoCalibrate(
    objpoints,
    imgpoints1,
    imgpoints2,
    K1, D1,
    K2, D2,
    gray1.shape[::-1],
    criteria=criteria_stereo,
    flags=flags
)

print("Готово!")
print("K1:\n", K1)
print("D1:\n", D1)
print("K2:\n", K2)
print("D2:\n", D2)
print("R:\n", R)
print("T:\n", T)

np.savez("stereo_params.npz",
         K1=K1, D1=D1,
         K2=K2, D2=D2,
         R=R, T=T,
         E=E, F=F)

print("Параметры сохранены в stereo_params.npz")
