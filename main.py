import cv2
import mediapipe as mp
from mtcnn import MTCNN
from multiprocessing import Process, Queue, Value
import time
import numpy as np

def initialize_detectors():
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    face_detector = MTCNN()
    return hands_detector, face_detector

def serialize_hand_landmarks(hands_results):
    serialized_hands = []
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            serialized_hand = []
            for landmark in hand_landmarks.landmark:
                serialized_hand.append((landmark.x, landmark.y, landmark.z))
            serialized_hands.append(serialized_hand)
    return serialized_hands

def face_detection_process(frame_queue, face_queue, log_queue, running):
    _, face_detector = initialize_detectors()
    while running.value:
        if frame_queue.empty():
            continue
        frame = frame_queue.get()
        start_time = time.time()
        faces = face_detector.detect_faces(frame)
        elapsed_time = time.time() - start_time
        log_queue.put(f"Face detection time: {elapsed_time:.3f} seconds")
        face_queue.put(faces)

def hand_detection_process(frame_queue, hand_queue, log_queue, running):
    hands_detector, _ = initialize_detectors()
    while running.value:
        if frame_queue.empty():
            continue
        frame = frame_queue.get()
        start_time = time.time()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_results = hands_detector.process(frame_rgb)
        elapsed_time = time.time() - start_time
        log_queue.put(f"Hand detection time: {elapsed_time:.3f} seconds")
        serialized_hands = serialize_hand_landmarks(hands_results)
        hand_queue.put(serialized_hands)

def post_processing_process(frame_queue, processed_frame_queue, running):
    while running.value:
        if frame_queue.empty():
            continue
        frame = frame_queue.get()
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed_frame_queue.put(processed_frame)

def logging_process(log_queue, running):
    while running.value:
        if log_queue.empty():
            continue
        log_message = log_queue.get()
        print(log_message)

def main():
    hands_detector, face_detector = initialize_detectors()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_queue = Queue(maxsize=10)
    face_queue = Queue(maxsize=5)
    hand_queue = Queue(maxsize=5)
    processed_frame_queue = Queue(maxsize=5)
    log_queue = Queue()
    running = Value('b', True)

    face_process = Process(target=face_detection_process, args=(frame_queue, face_queue, log_queue, running))
    hand_process = Process(target=hand_detection_process, args=(frame_queue, hand_queue, log_queue, running))
    post_process = Process(target=post_processing_process, args=(frame_queue, processed_frame_queue, running))
    log_process = Process(target=logging_process, args=(log_queue, running))

    face_process.start()
    hand_process.start()
    post_process.start()
    log_process.start()

    frame_count = 0
    last_faces = []
    last_hands = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1

            if frame_count % 5 == 0:
                if not frame_queue.full():
                    frame_queue.put(np.copy(frame))
                if not face_queue.empty():
                    last_faces = face_queue.get()
                if not hand_queue.empty():
                    last_hands = hand_queue.get()

            if not processed_frame_queue.empty():
                frame = processed_frame_queue.get()

            # Drawing last known face detections
            for face in last_faces:
                x, y, w, h = face['box']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 155, 255), 2)
                cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Drawing last known hand detections
            for hand_landmarks in last_hands:
                x_min = min([lm[0] for lm in hand_landmarks]) * frame.shape[1]
                y_min = min([lm[1] for lm in hand_landmarks]) * frame.shape[0]
                x_max = max([lm[0] for lm in hand_landmarks]) * frame.shape[1]
                y_max = max([lm[1] for lm in hand_landmarks]) * frame.shape[0]
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
                cv2.putText(frame, "Hand", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)
                for lm in hand_landmarks:
                    x, y = int(lm[0] * frame.shape[1]), int(lm[1] * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            cv2.imshow('Face and Hand Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        running.value = False
        face_process.join()
        hand_process.join()
        post_process.join()
        log_process.join()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
