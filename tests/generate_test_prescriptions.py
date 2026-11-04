import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "test_prescriptions")
os.makedirs(DATA_DIR, exist_ok=True)


def create_clear_handwritten_prescription() -> str:
    """Creates a simulated clear clinical prescription."""
    img = Image.new("RGB", (800, 600), color=(252, 250, 245))
    draw = ImageDraw.Draw(img)

    # Doctor header
    draw.text((40, 30), "CITY HEALTH CLINIC", fill=(30, 40, 60))
    draw.text((40, 50), "Dr. Vivek Sharma, MBBS, MD | Reg No: 84920", fill=(70, 80, 100))
    draw.line([(40, 75), (760, 75)], fill=(180, 190, 200), width=2)

    # Patient info
    draw.text((40, 95), "Patient: Vivek S, 19/M | Date: 12-Oct-2026", fill=(40, 50, 60))
    draw.text((40, 120), "Diagnosis: Acute Hypoglycemia & Dehydration", fill=(50, 60, 70))
    draw.line([(40, 145), (760, 145)], fill=(200, 210, 220), width=1)

    # Handwritten-style Rx section
    draw.text((40, 170), "Rx", fill=(20, 30, 50))
    draw.text((60, 210), "1. 5% Dextrose (IV) 1 bot stat", fill=(10, 20, 40))
    draw.text((60, 260), "2. Adequate fluid intake", fill=(10, 20, 40))
    draw.text((60, 310), "3. ORS 2 sachets in 1L water", fill=(10, 20, 40))
    draw.text((60, 360), "4. Tab Paracetamol 650mg SOS", fill=(10, 20, 40))

    # Doctor signature
    draw.text((550, 500), "Dr. V. Sharma", fill=(20, 40, 80))
    draw.text((550, 520), "[Signature]", fill=(100, 110, 120))

    path = os.path.join(DATA_DIR, "rx_clear_handwritten.png")
    img.save(path)
    return path


def create_printed_prescription() -> str:
    """Creates a clear printed prescription."""
    img = Image.new("RGB", (800, 600), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.text((40, 30), "METROPOLITAN DIABETES & CARDIAC CARE", fill=(15, 35, 75))
    draw.text((40, 55), "Clinical Consultation & Electronic Prescription", fill=(80, 90, 110))
    draw.line([(40, 80), (760, 80)], fill=(0, 100, 150), width=2)

    draw.text((40, 105), "Patient ID: P-90812 | Name: John Doe (54/M)", fill=(30, 40, 50))
    draw.text((40, 130), "Diagnosis: Type 2 Diabetes Mellitus, Essential Hypertension", fill=(30, 40, 50))
    draw.line([(40, 155), (760, 155)], fill=(220, 225, 230), width=1)

    draw.text((40, 180), "MEDICATIONS:", fill=(15, 35, 75))
    draw.text((60, 215), "1. Tab Metformin 500mg BID after meals for 30 days", fill=(20, 30, 40))
    draw.text((60, 265), "2. Tab Lisinopril 10mg OD morning with water", fill=(20, 30, 40))
    draw.text((60, 315), "3. Tab Atorvastatin 20mg HS bedtime for 30 days", fill=(20, 30, 40))

    path = os.path.join(DATA_DIR, "rx_printed.png")
    img.save(path)
    return path


def create_low_quality_handwritten_prescription() -> str:
    """Creates a degraded/noisy prescription with an unclear handwritten entry."""
    img = Image.new("RGB", (700, 500), color=(235, 230, 220))
    draw = ImageDraw.Draw(img)

    draw.text((40, 40), "Dr. Clinic Slip", fill=(60, 60, 60))
    draw.text((40, 80), "Rx:", fill=(40, 40, 40))
    draw.text((60, 120), "1. Tab Amoxicillin 500mg TID", fill=(30, 30, 30))
    draw.text((60, 170), "2. Xyqzwkln 250mg qd (scribbled illegible)", fill=(70, 70, 70))

    # Add Gaussian noise and blur to simulate low quality scan
    arr = np.array(img)
    noise = np.random.normal(0, 18, arr.shape).astype(np.int16)
    noisy_arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(noisy_arr, (3, 3), 0.8)

    out_img = Image.fromarray(blurred)
    path = os.path.join(DATA_DIR, "rx_low_quality.png")
    out_img.save(path)
    return path


if __name__ == "__main__":
    p1 = create_clear_handwritten_prescription()
    p2 = create_printed_prescription()
    p3 = create_low_quality_handwritten_prescription()
    print("Created test images:")
    print("1:", p1)
    print("2:", p2)
    print("3:", p3)
