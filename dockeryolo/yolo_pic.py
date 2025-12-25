import cv2
import os
import torch
from ultralytics import YOLO

# -------------------- CONFIG --------------------
YOLO_WEIGHTS = r"best.pt"      # ไฟล์ Model ของคุณ
YOLO_CONF = 0.25               # Confidence Threshold
YOLO_IMGSZ = 640               # ขนาดภาพที่ใช้ประมวลผล
YOLO_DEVICE = 0                # 0 สำหรับ CUDA (GPU) หรือ 'cpu'

# ตั้งค่าโฟลเดอร์
IMAGE_INPUT_PATH = "images/input"   # โฟลเดอร์ที่เก็บรูปต้นฉบับ
IMAGE_OUTPUT_PATH = "images/output" # โฟลเดอร์สำหรับเซฟรูปที่ตรวจจับแล้ว

# สร้างโฟลเดอร์ถ้ายังไม่มี
os.makedirs(IMAGE_OUTPUT_PATH, exist_ok=True)

# -------------------- INITIALIZE YOLO --------------------
device = 0 if torch.cuda.is_available() else 'cpu'
print(f"✅ ประมวลผลโดยใช้: {device}")

model = YOLO(YOLO_WEIGHTS)

# -------------------- PROCESS IMAGES --------------------
# ดึงรายชื่อไฟล์นามสกุลรูปภาพ (.jpg, .png, .jpeg)
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
image_files = [f for f in os.listdir(IMAGE_INPUT_PATH) if f.lower().endswith(valid_extensions)]

if not image_files:
    print(f"❌ ไม่พบรูปภาพใน: {IMAGE_INPUT_PATH}")
else:
    print(f"🔍 พบรูปภาพทั้งหมด {len(image_files)} รูป")

    for filename in image_files:
        img_path = os.path.join(IMAGE_INPUT_PATH, filename)
        
        # อ่านรูปภาพ
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # -------------------- YOLO INFERENCE --------------------
        results = model.predict(
            frame, 
            conf=YOLO_CONF, 
            imgsz=YOLO_IMGSZ, 
            device=device, 
            verbose=True # แสดงผล Log ใน Terminal
        )

        # -------------------- VISUALIZATION --------------------
        # วาด Bounding Box ลงในรูป
        annotated_frame = results[0].plot()

        # -------------------- SAVE & SHOW --------------------
        save_path = os.path.join(IMAGE_OUTPUT_PATH, f"result_{filename}")
        cv2.imwrite(save_path, annotated_frame)
        print(f"💾 บันทึกผลลัพธ์: {save_path}")

        # (Option) แสดงผลบนหน้าจอ - กดปุ่มใดๆ เพื่อดูรูปถัดไป หรือ 'q' เพื่อเลิก
        cv2.imshow("YOLO Image Detection", annotated_frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("✅ ประมวลผลครบถ้วน")