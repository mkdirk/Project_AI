import cv2
import time
import torch
from ultralytics import YOLO

# -------------------- CONFIG --------------------
YOLO_WEIGHTS = r"best.pt"  # ไฟล์ Model ของคุณ
YOLO_CONF = 0.25           # Confidence Threshold
YOLO_IMGSZ = 640           # ปรับลดลงเหลือ 640 ถ้าต้องการความเร็วที่สูงขึ้น (หรือใช้ 960 ตามเดิม)
YOLO_DEVICE = 0            # 0 สำหรับ CUDA (GPU)

# -------------------- INITIALIZE YOLO --------------------
if not torch.cuda.is_available():
    print("⚠️ ไม่พบ CUDA — YOLO จะทำงานบน CPU")
    device = 'cpu'
else:
    print(f"✅ ใช้ GPU: {torch.cuda.get_device_name(0)}")
    device = YOLO_DEVICE

model = YOLO(YOLO_WEIGHTS)

# -------------------- WEBCAM SETUP --------------------
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("❌ ไม่สามารถเปิดกล้องได้")

# ตั้งค่าความละเอียดกล้อง
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("[INFO] เริ่มการทำงาน... กด 'q' เพื่อออก")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------------------- YOLO INFERENCE --------------------
    t0 = time.time()
    
    # รัน Model (Stream=True จะช่วยให้ประหยัด Memory มากขึ้นในกรณีรันต่อเนื่อง)
    results = model.predict(
        frame, 
        conf=YOLO_CONF, 
        imgsz=YOLO_IMGSZ, 
        device=device, 
        verbose=False
    )
    
    t1 = time.time()
    latency_ms = int((t1 - t0) * 1000)

    # -------------------- VISUALIZATION --------------------
    # ใช้ Built-in method ของ Ultralytics ในการวาดภาพ (เร็วและครบถ้วน)
    # หรือจะวาดเองแบบโค้ดเก่าก็ได้ แต่ .plot() จะจัดการเรื่องสีและ Label ให้เสร็จสรรพ
    annotated_frame = results[0].plot()

    # แสดงค่า Latency บนหน้าจอ
    cv2.putText(annotated_frame, f"YOLO Latency: {latency_ms} ms", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    
    # คำนวณ FPS คร่าวๆ
    fps = 1 / (t1 - t0) if (t1 - t0) > 0 else 0
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # -------------------- DISPLAY --------------------
    cv2.imshow("YOLO GPU Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
