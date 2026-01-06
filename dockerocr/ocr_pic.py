import os, re, json, time, math
import cv2
import numpy as np
from paddleocr import PaddleOCR
import csv

cuda_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin' 

if os.path.exists(cuda_path):
    os.add_dll_directory(cuda_path)
    print("Added CUDA path explicitly")
else:
    print("CUDA path not found, please check")


# -------------------- CONFIG --------------------
INPUT_DIR = "images/input"    # โฟลเดอร์รูปต้นฉบับ
OUTPUT_DIR = "images/output"  # โฟลเดอร์ผลลัพธ์
POSTCODE_PATH = r"districts.json" # ไฟล์ตรวจสอบรหัสไปรษณีย์

# สร้างโฟลเดอร์ถ้ายังไม่มี
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# นามสกุลไฟล์รูปภาพที่รองรับ
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# -------------------- POSTAL & CONTEXT --------------------
POSTAL_KEYWORDS = {"รหัสไปรษณีย์","ไปรษณีย์","postcode","postal","zip","zip code","zipcode","post code"}
PHONE_WORDS = {"โทร","tel","phone"}

def load_postcode_whitelist(path: str):
    whitelist = set()
    if not os.path.exists(path):
        print(f"⚠️ Warning: ไม่พบไฟล์ {path} การตรวจสอบ Whitelist จะถูกข้าม")
        return whitelist

    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for row in data:
                for key in ("postalCode", "postcode", "zip", "zipcode"):
                    code = str(row.get(key, "")).strip()
                    if code.isdigit() and len(code) == 5:
                        whitelist.add(code)
        else:
            def walk(obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k in ("postalCode","postcode","zip","zipcode","zip_code"):
                            code = str(v).strip()
                            if code.isdigit() and len(code) == 5:
                                whitelist.add(code)
                        walk(v)
                elif isinstance(obj, list):
                    for it in obj:
                        walk(it)
            walk(data)
    else:  # CSV logic placeholder
        pass
    return whitelist

TH_POST_WHITELIST = load_postcode_whitelist(POSTCODE_PATH)

def is_thai_postal(code: str) -> bool:
    return code in TH_POST_WHITELIST

def find_postal_candidates(text: str):
    digits = re.sub(r"\D", "", text or "")
    return [digits[i:i+5] for i in range(max(0, len(digits)-4)) if len(digits[i:i+5]) == 5]

def box_center(box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return (sum(xs)/4.0, sum(ys)/4.0)

def l2(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# -------------------- OCR INIT --------------------
print(f"[INFO] Initializing OCR...")
ocr = PaddleOCR(
    device="gpu:0", # เปลี่ยนเป็น "cpu" ถ้าไม่มีการ์ดจอ NVIDIA
    use_textline_orientation=True,
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="th_PP-OCRv5_mobile_rec",
    text_det_limit_side_len=1280,
    text_det_box_thresh=0.5,        
    text_det_unclip_ratio=1.6,      
    text_rec_score_thresh=0.4       
)

# -------------------- MAIN PROCESS FUNCTION --------------------
def process_image(filename):
    img_path = os.path.join(INPUT_DIR, filename)
    
    # 1. อ่านภาพ
    # print(f"Processing: {filename}")
    # ใช้ imdecode เพื่อรองรับ path ภาษาไทย
    try:
        frame = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"❌ Error reading {filename}: {e}")
        return

    if frame is None:
        print(f"❌ Error: อ่านไฟล์ภาพไม่ได้ {filename}")
        return

    vis = frame.copy()
    H, W = vis.shape[:2]
    diag = math.hypot(W, H)
    NEAR_RADIUS = 0.12 * diag

    # 2. Run OCR
    t0 = time.time()
    results = ocr.ocr(vis, cls=True)
    t1 = time.time()
    latency_ms = int((t1 - t0) * 1000)

    texts, scores, boxes = [], [], []
    keyword_centers, phone_centers = [], []

    # 3. Parse Result
    if results and len(results) > 0 and results[0] is not None:
        det = results[0]
        raw_texts  = [line[1][0] for line in det] if isinstance(det[0], list) else (det.get("rec_texts", []) or [])
        raw_scores = [line[1][1] for line in det] if isinstance(det[0], list) else (det.get("rec_scores", []) or [])
        raw_boxes  = [line[0] for line in det]    if isinstance(det[0], list) else (det.get("boxes", []) or [])
        
        texts = raw_texts
        scores = raw_scores
        
        clean_boxes = []
        for b in raw_boxes:
            try:
                quad = [(float(p[0]), float(p[1])) for p in b]
                if len(quad) == 4: clean_boxes.append(quad)
            except: pass
        boxes = clean_boxes

        for box, t in zip(boxes, texts):
            t_low = (t or "").lower()
            c = box_center(box)
            if any(kw in t_low for kw in POSTAL_KEYWORDS):
                keyword_centers.append(c)
            if any(pw in t_low for pw in PHONE_WORDS):
                phone_centers.append(c)

        # 4. Ranking
        postal_candidates = []
        for idx, (t, s, box) in enumerate(zip(texts, scores, boxes), start=1):
            conf = float(s) if isinstance(s, (int, float)) else 0.0
            t_low = (t or "").lower()
            cx, cy = box_center(box)

            for code in find_postal_candidates(t):
                valid_whitelist = is_thai_postal(code)
                score = 0
                if valid_whitelist: score += 4
                if keyword_centers:
                    d = min(l2((cx,cy), kc) for kc in keyword_centers)
                    if d <= NEAR_RADIUS: score += 2
                if cy > H * 0.55: score += 1
                if cx > W * 0.55: score += 1
                score += int(round(conf * 2))
                if phone_centers:
                    dph = min(l2((cx,cy), pc) for pc in phone_centers)
                    if dph <= NEAR_RADIUS: score -= 2
                if any(kw in t_low for kw in POSTAL_KEYWORDS):
                    score += 1

                postal_candidates.append({
                    "line_index": idx,
                    "text": t,
                    "ocr_score": conf,
                    "code": code,
                    "valid_th_range": valid_whitelist,
                    "score": int(score),
                    "center": (cx, cy)
                })

        best_postal = None
        if postal_candidates:
            postal_candidates.sort(
                key=lambda c: (c["score"], c["ocr_score"], c["center"][1], c["center"][0]),
                reverse=True
            )
            best_postal = postal_candidates[0]

        # 5. Draw
        cand_lines = {c["line_index"] for c in postal_candidates}
        for i, (box, t) in enumerate(zip(boxes, texts), start=1):
            pts = np.array(box, dtype=np.int32)
            color = (0, 200, 0)
            if i in cand_lines: color = (0, 165, 255)
            if best_postal and i == best_postal["line_index"]: color = (0, 0, 255)
            cv2.polylines(vis, [pts], True, color, 2)

        # Info Box
        cv2.rectangle(vis, (0,0), (W, 80), (0,0,0), -1)
        if best_postal:
            txt = f"POSTAL: {best_postal['code']} (Score: {best_postal['score']})"
            cv2.putText(vis, txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            print(f"   -> ✅ Result: {best_postal['code']} (file: {filename})")
        else:
            cv2.putText(vis, "POSTAL: Not Found", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            print(f"   -> ⚠️ Not Found (file: {filename})")

        # 6. Save
        # ตั้งชื่อไฟล์ Output ให้เหมือน Input
        save_img_path = os.path.join(OUTPUT_DIR, filename)
        
        # ใช้ imencode รองรับ path ภาษาไทย
        is_success, im_buf = cv2.imencode(os.path.splitext(filename)[1], vis)
        if is_success:
            im_buf.tofile(save_img_path)

        # Save Text
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(OUTPUT_DIR, txt_filename)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Image: {filename}\n")
            f.write(f"Best Postal: {best_postal['code'] if best_postal else 'None'}\n")
            f.write("-" * 20 + "\n")
            for t, s in zip(texts, scores):
                f.write(f"{t}\t{s}\n")
        
        # Show (Optional - แสดงแว้บเดียวเพื่อให้รู้ว่าทำงานอยู่)
        cv2.imshow("Batch Processing...", vis)
        cv2.waitKey(1) # รอ 1ms แล้วทำต่อเลย ไม่ต้องกดปุ่ม

    else:
        print(f"   -> No text detected in {filename}")

# -------------------- RUN LOOP --------------------
if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        print(f"❌ ไม่พบโฟลเดอร์ {INPUT_DIR} กรุณาสร้างและใส่รูปภาพ")
        exit()

    files = os.listdir(INPUT_DIR)
    img_files = [f for f in files if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS]

    print(f"Found {len(img_files)} images in {INPUT_DIR}")
    print("-" * 40)

    for i, f in enumerate(img_files):
        print(f"[{i+1}/{len(img_files)}] Processing {f}...")
        process_image(f)

    print("-" * 40)
    print("✅ All done! Check results in 'images/output'")
    cv2.destroyAllWindows()