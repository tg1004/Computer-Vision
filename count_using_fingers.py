import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands=mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw=mp.solutions.drawing_utils

cap=cv2.VideoCapture(0)

finger_tips=[4,8,12,16,20]

while True:
    ret,frame=cap.read()
    if not ret:
        break
    frame=cv2.flip(frame,1)

    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    result=hands.process(rgb)

    finger_cnt=0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            lm_list=[]
            h,w,_=(frame.shape)

            for id,lm in enumerate(hand_landmarks.landmark):
                cx,cy=int(lm.x*w),int(lm.y*h)
                lm_list.append((id,cx,cy))

            if lm_list[4][1]>lm_list[3][1]:
                finger_cnt+=1

            for tip in finger_tips[1:]:
                if lm_list[tip][2]<lm_list[tip-2][2]:
                    finger_cnt+=1

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
            )
        cv2.putText(
            frame,
            f"Fingers: {finger_cnt}",
            (20,60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0,255,0),
            3
        )

        cv2.imshow("Hand Gesture Deyection",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()




