import streamlit as st
import cv2
import os
import pandas as pd
from datetime import datetime, timedelta
from ultralytics import YOLO
import tempfile
import time
import uuid
import requests 

# K85129216888957
OCR_SPACE_API_KEY = "K81773839888957"


def lastfour(n):
    i = len(n) - 4
    while i < len(n):
        if not n[i].isdigit():
            return False
        i += 1
    return True


def threendfour(n):
    if len(n) > 3:
        return n[2].isdigit() and n[3].isdigit()
    return False


def firsttwo(n):
    return len(n) >= 2 and n[0].isalpha() and n[1].isalpha()


def new(n):
    if len(n) == 8:
        return True
    elif len(n) == 9:
        return n[4].isalpha()
    elif len(n) == 10:
        return n[4].isalpha() and n[5].isalpha()
    return False


def cleaning(n):
    result = ""
    for c in n:
        if c.isalnum():
            result += c
    return result



def imagetostring_from_array(image_array):
    try:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

        gray = cv2.resize(
            gray,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        _, gray = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        success, encoded_image = cv2.imencode(".jpg", gray)

        if not success:
            return ""

        response = requests.post(
            "https://api.ocr.space/parse/image",
            files={
                "file": (
                    "plate.jpg",
                    encoded_image.tobytes(),
                    "image/jpeg"
                )
            },
            data={
                "apikey": OCR_SPACE_API_KEY,
                "language": "eng",
                "OCREngine": 2,
                "isOverlayRequired": False
            },
            timeout=30
        )

        result = response.json()

        print("\nOCR RESPONSE:")
        print(result)

        if result.get("ParsedResults"):

            text = result["ParsedResults"][0].get(
                "ParsedText",
                ""
            )

            print("OCR RAW:", text)

            number_plate = cleaning(
                text.upper()
                .replace("\n", "")
                .replace(" ", "")
                .replace("IND", "")
            )

            print("OCR CLEAN:", number_plate)

            if (
                8 <= len(number_plate) <= 10
                and firsttwo(number_plate)
                and threendfour(number_plate)
                and lastfour(number_plate)
                and new(number_plate)
            ):
                return number_plate

        return ""

    except Exception as e:
        print("OCR ERROR:", e)
        return ""



model = YOLO("license_plate_detector.pt")


csv_file = "live_detected_number_plates.csv"

if not os.path.exists(csv_file):
    pd.DataFrame(
        columns=["Timestamp", "Detected Number Plate"]
    ).to_csv(csv_file, index=False)

if "last_saved" not in st.session_state:
    st.session_state.last_saved = {}

last_saved = st.session_state.last_saved


st.set_page_config(layout="wide")
st.title("Indian Number Plate Detection using YOLO + OCR.Space")

video_input = st.file_uploader(
    "Upload a Video",
    type=["mp4", "avi", "mov"]
)

start_detection = st.button("Start Detection")


if start_detection and video_input:

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(video_input.read())

    cap = cv2.VideoCapture(temp_file.name)

    video_col, plate_col = st.columns([3, 1])

    with video_col:
        st.subheader("Live Video Feed")
        stframe = st.empty()

    with plate_col:
        st.subheader("Detected Plates")
        plate_placeholder = st.empty()

    table_placeholder = st.empty()
    download_placeholder = st.empty()

    detected_plates_display = []

    frame_count = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        detected_plates = []

        if frame_count % 5 == 0:

            results = model(frame)

            for r in results:

                for box in r.boxes:

                    x1, y1, x2, y2 = map(int,box.xyxy[0])

                    cropped_plate = frame[y1:y2,x1:x2]

                    if cropped_plate.size == 0:
                        continue
                    cv2.imwrite(f"plates\debug_plate_{frame_count}.jpg",cropped_plate)

                    number_plate = imagetostring_from_array(cropped_plate)

                    if number_plate:

                        print(f"DETECTED: {number_plate}")

                        now = datetime.now()

                        if (number_plate not in last_saved or (now - last_saved[number_plate]) > timedelta(minutes=5)):

                            timestamp = now.strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )

                            pd.DataFrame(
                                [[timestamp, number_plate]],
                                columns=[
                                    "Timestamp",
                                    "Detected Number Plate"
                                ]
                            ).to_csv(
                                csv_file,
                                mode="a",
                                header=False,
                                index=False
                            )

                            last_saved[number_plate] = now

                            detected_plates.append(
                                number_plate
                            )

                            print(f"SAVED: {number_plate}")

                        else:

                            remaining = (
                                timedelta(minutes=5)
                                - (now - last_saved[number_plate])
                            ).seconds

                            print(f"SKIPPED: {number_plate} "f"(wait {remaining}s)")

                    else:
                        print("UNREADABLE PLATE")

            stframe.image(
                frame,
                channels="BGR",
                caption="Live Detection",
                use_container_width=True
            )

        if detected_plates:

            for plate in detected_plates:
                detected_plates_display.append(plate)

            plate_placeholder.markdown(
                "### Detected Plates\n\n" +
                "\n".join(
                    [f"✅ {plate}" for plate in reversed(detected_plates_display)]
                )
            )

        
        if frame_count % 10 == 0:

            if os.path.exists(csv_file):

                df = pd.read_csv(csv_file)


                table_placeholder.dataframe(
                    df,
                    use_container_width=True,
                    height=300
                )

                download_placeholder.download_button(
                    "⬇️ Download CSV",
                    data=df.to_csv(index=False),
                    file_name="detected_number_plates.csv",
                    mime="text/csv",
                    key=f"download-{uuid.uuid4()}"
                )

        frame_count += 1

        time.sleep(0.03)

    cap.release()

    st.success("✅ Video Processing Completed Successfully!")