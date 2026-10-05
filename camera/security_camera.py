import cv2
import mediapipe as mp
import time

# Configuración
CAMERA_INDEX = 0
PERSON_SCALE = 0.5   # se reduce el frame para que el detector de personas vaya más rápido
FACE_CONFIDENCE = 0.5

PERSON_COLOR = (0, 200, 0)   # verde (BGR)
FACE_COLOR = (255, 120, 0)   # azul


def draw_label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y = max(y, th + 6)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def detect_people(hog, img):
    small = cv2.resize(img, None, fx=PERSON_SCALE, fy=PERSON_SCALE)
    rects, weights = hog.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)
    boxes = [[int(v / PERSON_SCALE) for v in r] for r in rects]
    # Quitar cajas duplicadas que se solapan sobre la misma persona
    keep = cv2.dnn.NMSBoxes(boxes, [float(w) for w in weights], 0.3, 0.4) if len(boxes) else []
    return [boxes[i] for i in (keep.flatten() if hasattr(keep, "flatten") else keep)]


def detect_faces(face_detector, img):
    h, w, _ = img.shape
    results = face_detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    boxes = []
    for det in results.detections or []:
        bb = det.location_data.relative_bounding_box
        boxes.append([int(bb.xmin * w), int(bb.ymin * h), int(bb.width * w), int(bb.height * h)])
    return boxes


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    face_detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=FACE_CONFIDENCE
    )

    prev_time = time.time()
    while True:
        success, img = cap.read()
        if not success:
            break

        people = detect_people(hog, img)
        faces = detect_faces(face_detector, img)

        for i, (x, y, w, h) in enumerate(people, start=1):
            cv2.rectangle(img, (x, y), (x + w, y + h), PERSON_COLOR, 2)
            draw_label(img, f"Persona {i}", x, y, PERSON_COLOR)

        for i, (x, y, w, h) in enumerate(faces, start=1):
            cv2.rectangle(img, (x, y), (x + w, y + h), FACE_COLOR, 2)
            draw_label(img, f"Cara {i}", x, y, FACE_COLOR)

        now = time.time()
        fps = 1 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(img, f"FPS: {fps:.0f}  Personas: {len(people)}  Caras: {len(faces)}",
                    (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Camara de seguridad", img)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):  # q o Esc para salir
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
