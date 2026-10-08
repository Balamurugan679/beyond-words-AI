import cv2
import mediapipe as mp
import math
import pyttsx3

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Text-to-speech
engine = pyttsx3.init()
engine.setProperty("rate", 150)

cap = cv2.VideoCapture(1)

def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def fingers_up(hand):
    """
    Returns:
    [thumb, index, middle, ring, pinky]

    1 = finger extended
    0 = finger folded
    """

    fingers = []

    # Thumb
    if hand.landmark[4].x < hand.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Index
    if hand.landmark[8].y < hand.landmark[6].y:
        fingers.append(1)
    else:
        fingers.append(0)

    # Middle
    if hand.landmark[12].y < hand.landmark[10].y:
        fingers.append(1)
    else:
        fingers.append(0)

    # Ring
    if hand.landmark[16].y < hand.landmark[14].y:
        fingers.append(1)
    else:
        fingers.append(0)

    # Pinky
    if hand.landmark[20].y < hand.landmark[18].y:
        fingers.append(1)
    else:
        fingers.append(0)

    return fingers


def recognize_sign(hand_landmarks):

    fingers = fingers_up(hand_landmarks)

    thumb, index, middle, ring, pinky = fingers

    if (
        thumb == 1 and
        index == 1 and
        middle == 0 and
        ring == 0 and
        pinky == 1
    ):
        return "I LOVE YOU"

    if (
        thumb == 1 and
        index == 1 and
        middle == 1 and
        ring == 1 and
        pinky == 1
    ):
        return "HELLO"


    if (
        thumb == 1 and
        index == 1 and
        middle == 0 and
        ring == 0 and
        pinky == 0
    ):
        return "THANKS"

    return "UNKNOWN"

last_sign = ""
spoken_sign = ""

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("Could not read camera")
        break

    # Flip camera for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand
    results = hands.process(rgb_frame)

    detected_sign = "UNKNOWN"
    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw all 21 hand points
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Recognize sign
            detected_sign = recognize_sign(hand_landmarks)
    cv2.rectangle(
        frame,
        (0, 0),
        (640, 100),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "BEYOND WORDS",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Sign: " + detected_sign,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )
    if detected_sign != "UNKNOWN":

        if detected_sign != spoken_sign:

            engine.say(detected_sign)
            engine.runAndWait()

            spoken_sign = detected_sign

    else:
        spoken_sign = ""

    cv2.putText(
        frame,
        "Show: HELLO | THANKS | I LOVE YOU",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow("Beyond Words - Sign Language Recognition", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
hands.close()