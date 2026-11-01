
import cv2
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from ultralytics import YOLO


# =====================================================
# 1. 系統設定
# =====================================================

BASE_DIR = Path("System_Path")

VIDEO_PATH = BASE_DIR / "road_01.mp4"
MODEL_PATH = BASE_DIR / "yolov8s.pt"

FRAME_DIR = BASE_DIR / "frames"
OUTPUT_DIR = BASE_DIR / "output"

FRAMES_JSONL = BASE_DIR / "frames.jsonl"
DETECTIONS_JSONL = OUTPUT_DIR / "detections.jsonl"
DEMO_VIDEO = OUTPUT_DIR / "traffic_demo.mp4"


# =====================================================
# 2. YOLO 模型設定
# =====================================================

CONFIDENCE = 0.4
IMAGE_SIZE = 640

# COCO 六種常見道路交通類別：
# 0 person
# 1 bicycle
# 2 car
# 3 motorcycle
# 5 bus
# 7 truck

CLASS_IDS = [0, 1, 2, 3, 5, 7]

# None = 自動選擇可用裝置
# "cpu" = 使用 CPU
# "0" = 使用第一張 NVIDIA GPU
DEVICE = None


# =====================================================
# 3. 影片時間設定
# =====================================================

# 如果知道影片實際開始拍攝時間，
# 可以填入 ISO 8601 時間，例如：
# "2026-11-01T13:00:00+08:00"
#
# 不知道就維持 None。
# None 時使用影片內的相對時間。

VIDEO_START_TIME_ISO = None

# 每隔幾個原始影格擷取一次
# 1 = 每個影格都擷取，建議用於 Pixel 誤差分析
FRAME_INTERVAL = 1


# =====================================================
# 4. 建立資料夾
# =====================================================

FRAME_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 5. 時間戳記格式
# =====================================================

def get_timestamp(frame_index, fps, start_time=None):

    timestamp_ms = round(frame_index / fps * 1000)

    if start_time is not None:

        current_time = start_time + timedelta(
            milliseconds=timestamp_ms
        )

        timestamp = current_time.isoformat(
            timespec="milliseconds"
        )

    else:

        seconds = timestamp_ms // 1000
        milliseconds = timestamp_ms % 1000

        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60

        timestamp = (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{remaining_seconds:02d}."
            f"{milliseconds:03d}"
        )

    return timestamp, timestamp_ms


# =====================================================
# 6. 第一階段：影片擷取影格
# =====================================================

def extract_frames():

    print("\n========== 開始擷取影片影格 ==========")

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"找不到影片：{VIDEO_PATH}"
        )

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("無法開啟影片！")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        raise RuntimeError("無法取得影片 FPS！")

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    start_time = None

    if VIDEO_START_TIME_ISO:
        start_time = datetime.fromisoformat(
            VIDEO_START_TIME_ISO
        )

    saved_count = 0
    frame_index = 0

    # frames.jsonl 使用 UTF-8 編碼
    with open(
        FRAMES_JSONL,
        "w",
        encoding="utf-8"
    ) as json_file:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # 依照設定決定是否擷取
            if frame_index % FRAME_INTERVAL == 0:

                filename = (
                    f"frame_{frame_index:06d}.jpg"
                )

                image_path = FRAME_DIR / filename

                success = cv2.imwrite(
                    str(image_path),
                    frame
                )

                if not success:
                    raise RuntimeError(
                        f"圖片儲存失敗：{image_path}"
                    )

                timestamp, timestamp_ms = (
                    get_timestamp(
                        frame_index,
                        fps,
                        start_time
                    )
                )

                # 每張圖片一筆 JSON
                frame_data = {

                    "frame_index": frame_index,

                    "image": str(
                        image_path.relative_to(BASE_DIR)
                    ).replace("\\", "/"),

                    "timestamp": timestamp,

                    "timestamp_ms": timestamp_ms,

                    "fps": fps,

                    "width": width,

                    "height": height
                }

                json_file.write(
                    json.dumps(
                        frame_data,
                        ensure_ascii=False
                    ) + "\n"
                )

                saved_count += 1

            frame_index += 1

            if frame_index % 100 == 0:

                print(
                    f"擷取進度："
                    f"{frame_index}/{total_frames}"
                )

    cap.release()

    print("\n影格擷取完成！")
    print("影片總影格：", total_frames)
    print("成功擷取：", saved_count)
    print("圖片位置：", FRAME_DIR)
    print("時間資料：", FRAMES_JSONL)


# =====================================================
# 7. 第二階段：YOLOv8s 物件偵測
# =====================================================

def detect_objects():

    print("\n========== 開始 YOLOv8s 偵測 ==========")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"找不到模型：{MODEL_PATH}"
        )

    if not FRAMES_JSONL.exists():
        raise FileNotFoundError(
            "找不到 frames.jsonl，請先擷取影片！"
        )

    # 載入 YOLO 模型
    model = YOLO(str(MODEL_PATH))

    print("YOLOv8s 載入成功！")

    # 取得模型類別名稱
    model_names = model.names

    writer = None

    processed_count = 0

    # 讀取 frames.jsonl
    with open(
        FRAMES_JSONL,
        "r",
        encoding="utf-8"
    ) as frame_file, open(
        DETECTIONS_JSONL,
        "w",
        encoding="utf-8"
    ) as detection_file:

        for line in frame_file:

            if not line.strip():
                continue

            frame_data = json.loads(line)

            frame_index = frame_data["frame_index"]

            image_path = BASE_DIR / frame_data["image"]

            # 讀取圖片
            frame = cv2.imread(str(image_path))

            if frame is None:
                print("無法讀取圖片：", image_path)
                continue

            # YOLOv8s 偵測
            results = model.predict(
                source=frame,
                classes=CLASS_IDS,
                conf=CONFIDENCE,
                imgsz=IMAGE_SIZE,
                device=DEVICE,
                verbose=False
            )

            result = results[0]

            detections = []

            # 取得辨識框
            if result.boxes is not None:

                boxes = result.boxes

                xyxy_list = (
                    boxes.xyxy.cpu().tolist()
                )

                class_list = (
                    boxes.cls.cpu().tolist()
                )

                confidence_list = (
                    boxes.conf.cpu().tolist()
                )

                for xyxy, class_id, confidence in zip(
                    xyxy_list,
                    class_list,
                    confidence_list
                ):

                    class_id = int(class_id)

                    # 轉換成 Pixel 座標
                    x1, y1, x2, y2 = xyxy

                    detection = {

                        "class_id": class_id,

                        "class_name": model_names[class_id],

                        "confidence": round(
                            float(confidence), 4
                        ),

                        "xyxy": [
                            round(float(x1), 2),
                            round(float(y1), 2),
                            round(float(x2), 2),
                            round(float(y2), 2)
                        ]
                    }

                    detections.append(detection)

            # 每張圖片一筆偵測結果
            detection_data = {

                "frame_index": frame_index,

                "image": frame_data["image"],

                "timestamp": frame_data["timestamp"],

                "timestamp_ms": frame_data["timestamp_ms"],

                "image_width": frame.shape[1],

                "image_height": frame.shape[0],

                "detections": detections

            }

            detection_file.write(
                json.dumps(
                    detection_data,
                    ensure_ascii=False
                ) + "\n"
            )

            # 繪製 YOLO 偵測框
            annotated_frame = result.plot()

            # 顯示影格編號
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_index}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            # 建立輸出影片
            if writer is None:

                height, width = annotated_frame.shape[:2]

                fps = frame_data["fps"] / FRAME_INTERVAL

                fourcc = cv2.VideoWriter_fourcc(
                    *"mp4v"
                )

                writer = cv2.VideoWriter(
                    str(DEMO_VIDEO),
                    fourcc,
                    fps,
                    (width, height)
                )

                if not writer.isOpened():
                    raise RuntimeError(
                        "無法建立 Demo 影片！"
                    )

            writer.write(annotated_frame)

            processed_count += 1

            if processed_count % 100 == 0:

                print(
                    f"偵測進度：{processed_count}"
                )

    if writer is not None:
        writer.release()

    print("\n========== 偵測完成 ==========")

    print("處理影格數量：", processed_count)

    print("JSONL 輸出位置：", DETECTIONS_JSONL)

    print("Demo 影片位置：", DEMO_VIDEO)


# =====================================================
# 8. 主程式
# =====================================================

if __name__ == "__main__":

    try:

        # 第一階段：影片轉換成圖片及時間資料
        extract_frames()

        # 第二階段：YOLO 偵測及輸出
        detect_objects()

        print("\n所有工作已完成！")

    except Exception as error:

        print("\n程式執行失敗：", error)

        raise