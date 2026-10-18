"""
棋盤格相機校正 demo（Zhang's method）

流程：findChessboardCorners → cornerSubPix → calibrateCamera → 重投影誤差 → undistort

用法（任何目錄都可以執行，預設讀取本檔旁邊的 images/，輸出到 output/）：
    python calibration/calibrate.py
    python calibration/calibrate.py --cols 7 --rows 6 --square 0.03
    python calibration/calibrate.py --experiment      # 張數 vs 誤差實驗（報告素材）

說明：
    - (cols, rows) 是「內角點」數量（方格交會點），不是方格數。9x6 內角點 = 10x7 方格。
    - 輸出 K = [[fx,0,cx],[0,fy,cy],[0,0,1]]，單位 px；畸變係數順序 (k1, k2, p1, p2, k3)。
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# ---------- 檔案 I/O（用 numpy 讀寫，避免 Windows 上 cv2 遇到中文路徑失敗） ----------
def imread(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path, img):
    ok, buf = cv2.imencode(Path(path).suffix, img)
    if not ok:
        raise IOError(f"無法編碼影像：{path}")
    buf.tofile(str(path))


# ---------- 校正流程 ----------
def make_object_points(cols, rows, square):
    """棋盤格角點的 3D 座標（棋盤座標系，Z=0 平面，單位 m）。"""
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return objp


def detect_corners(paths, pattern, debug_dir=None):
    """回傳 [(path, corners)] 與影像尺寸 (w, h)。偵測失敗或尺寸不一致的影像會被跳過。"""
    found, size = [], None
    for p in paths:
        img = imread(p)
        if img is None:
            print(f"  [略過] 無法讀取：{p.name}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        this_size = gray.shape[::-1]
        if size is None:
            size = this_size
        elif this_size != size:
            print(f"  [略過] 解析度 {this_size} 與第一張 {size} 不同：{p.name}")
            continue
        ok, corners = cv2.findChessboardCorners(
            gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not ok:
            print(f"  [略過] 找不到棋盤格：{p.name}")
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)
        found.append((p, corners))
        if debug_dir is not None:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern, corners, True)
            imwrite(debug_dir / f"{p.stem}_corners.png", vis)
    return found, size


def calibrate(objp, corner_list, size):
    objpoints = [objp] * len(corner_list)
    return cv2.calibrateCamera(objpoints, corner_list, size, None, None)


def sq_errors(objp, corners, rvec, tvec, K, dist):
    """單張影像每個角點的『重投影距離平方』（px²）。"""
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    diff = corners.reshape(-1, 2) - proj.reshape(-1, 2)
    return np.sum(diff**2, axis=1)


def per_image_rms(objp, corner_list, rvecs, tvecs, K, dist):
    return [
        float(np.sqrt(np.mean(sq_errors(objp, c, r, t, K, dist))))
        for c, r, t in zip(corner_list, rvecs, tvecs)
    ]


def holdout_rms(objp, corner_list, K, dist):
    """用『固定內參』替未參與校正的影像解外參（solvePnP），再算重投影 RMS。"""
    errs = []
    for c in corner_list:
        ok, rvec, tvec = cv2.solvePnP(objp, c, K, dist)
        if ok:
            errs.append(sq_errors(objp, c, rvec, tvec, K, dist))
    return float(np.sqrt(np.mean(np.concatenate(errs))))


def save_undistort_samples(found, K, dist, size, out_dir, n=3):
    w, h = size
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))  # alpha=1 保留所有像素
    for p, _ in found[:n]:
        img = imread(p)
        und = cv2.undistort(img, K, dist, None, new_K)
        left, right = img.copy(), und.copy()
        cv2.putText(left, "Original", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(right, "Undistorted", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 160, 0), 2)
        imwrite(out_dir / f"undistort_{p.stem}.png", np.hstack([left, right]))


# ---------- 實驗：張數 vs 誤差 ----------
def run_experiment(objp, corner_list, size, trials, seed=0):
    """隨機抽 n 張校正，用『沒抽到的影像』評估。訓練誤差會隨張數變大，holdout 誤差才反映品質。"""
    rng = random.Random(seed)
    total = len(corner_list)
    print(f"\n[實驗] 共 {total} 張可用，每組隨機重複 {trials} 次（固定 seed={seed}）")
    print(f"{'n_train':>8} | {'train RMS':>10} | {'holdout RMS':>12} | {'fx (mean ± std)':>20}")
    for n in (5, 8, 10, 15, 20):
        if n > total - 2:  # 至少留 2 張當驗證
            continue
        tr, ho, fxs = [], [], []
        for _ in range(trials):
            idx = list(range(total))
            rng.shuffle(idx)
            train = [corner_list[i] for i in idx[:n]]
            test = [corner_list[i] for i in idx[n:]]
            try:
                rms, K, dist, _, _ = calibrate(objp, train, size)
                ho.append(holdout_rms(objp, test, K, dist))
            except cv2.error:
                continue
            tr.append(rms)
            fxs.append(K[0, 0])
        if tr:
            print(
                f"{n:>8} | {np.mean(tr):>10.3f} | {np.mean(ho):>12.3f} | "
                f"{np.mean(fxs):>11.1f} ± {np.std(fxs):<6.1f}"
            )


# ---------- 主程式 ----------
def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="棋盤格相機校正 demo")
    ap.add_argument("--images", type=Path, default=here / "images", help="棋盤格影像資料夾")
    ap.add_argument("--out", type=Path, default=here / "output", help="輸出資料夾")
    ap.add_argument("--cols", type=int, default=9, help="每列內角點數（預設 9）")
    ap.add_argument("--rows", type=int, default=6, help="每行內角點數（預設 6）")
    ap.add_argument("--square", type=float, default=0.025, help="方格邊長，單位 m（預設 0.025）")
    ap.add_argument("--experiment", action="store_true", help="執行張數 vs 誤差實驗")
    ap.add_argument("--trials", type=int, default=10, help="實驗每組重複次數")
    args = ap.parse_args()

    if not args.images.is_dir():
        sys.exit(f"找不到影像資料夾：{args.images}\n請先放入棋盤格照片，或用 --images 指定路徑。")
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not paths:
        sys.exit(f"{args.images} 內沒有影像（支援 {sorted(IMG_EXTS)}）。")

    args.out.mkdir(parents=True, exist_ok=True)
    debug_dir = args.out / "corners"
    debug_dir.mkdir(exist_ok=True)

    pattern = (args.cols, args.rows)
    objp = make_object_points(args.cols, args.rows, args.square)

    print(f"讀取 {len(paths)} 張影像，內角點 {args.cols}x{args.rows}，方格 {args.square * 1000:.1f} mm")
    found, size = detect_corners(paths, pattern, debug_dir)
    print(f"成功偵測棋盤格：{len(found)}/{len(paths)} 張，解析度 {size}")
    if len(found) < 3:
        sys.exit("可用影像少於 3 張，無法校正。請確認 --cols/--rows 是『內角點數』，並多拍幾張。")
    if len(found) < 10:
        print("[警告] 建議至少 10 張、含不同角度與距離，否則 K 不穩定。")

    corner_list = [c for _, c in found]
    rms, K, dist, rvecs, tvecs = calibrate(objp, corner_list, size)
    errs = per_image_rms(objp, corner_list, rvecs, tvecs, K, dist)
    mean_err = float(np.mean(errs))

    w, h = size
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    hfov = math.degrees(2 * math.atan(w / (2 * fx)))

    print("\n=== 校正結果 ===")
    print(f"內參矩陣 K =\n{np.array2string(K, precision=2, suppress_small=True)}")
    print(f"fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}  (影像中心 {w / 2:.1f}, {h / 2:.1f})")
    print(f"畸變係數 (k1,k2,p1,p2,k3) = {np.round(dist.ravel(), 5).tolist()}")
    print(f"由 fx 反推水平視角 HFOV = {hfov:.2f}°（對照模擬器公式 fx = (W/2)/tan(HFOV/2)）")
    print(f"整體重投影 RMS（cv2 回傳）= {rms:.3f} px")
    print(f"各影像 RMS 的平均值        = {mean_err:.3f} px")
    worst = sorted(zip(errs, found), key=lambda x: -x[0])[:3]
    print("誤差最大的 3 張：" + "、".join(f"{p.name}({e:.2f})" for e, (p, _) in worst))
    print("驗收（< 0.5 px）：" + ("通過" if mean_err < 0.5 else "未通過，可剔除誤差大的影像或重拍"))

    result = {
        "image_size": [w, h],
        "pattern_inner_corners": [args.cols, args.rows],
        "square_size_m": args.square,
        "num_images_used": len(found),
        "K": K.tolist(),
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "dist_coeffs_k1_k2_p1_p2_k3": dist.ravel().tolist(),
        "hfov_deg_from_fx": hfov,
        "rms_reprojection_error_px": float(rms),
        "mean_per_image_rms_px": mean_err,
        "per_image_rms_px": {p.name: e for (p, _), e in zip(found, errs)},
    }
    out_json = args.out / "calibration_result.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    save_undistort_samples(found, K, dist, size, args.out)
    print(f"\n已輸出：{out_json}\n         {args.out}/undistort_*.png（前後對照）\n         {debug_dir}/（角點偵測結果）")

    if args.experiment:
        run_experiment(objp, corner_list, size, args.trials)


if __name__ == "__main__":
    main()
