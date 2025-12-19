import cv2
import mediapipe as mp
import random
import winsound
import time

mp_hands=mp.solutions.hands
hands=mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw=mp.solutions.drawing_utils

cap=cv2.VideoCapture(0)

finger_tips=[4,8,12,16,20]
gestures=["Rock","Paper","Scissors"]

rounds=3
current_round=1
player_score=0
computer_score=0

last_time=time.time()
countdown=3
game_state="countdown"
player_move="None"
computer_move="None"

def count_fingers(lm_list):
    fingers=0
    if lm_list[4][1]>lm_list[3][1]:
        fingers+=1

    for tip in finger_tips[1:]:
        if lm_list[tip][2]<lm_list[tip-2][2]:
            fingers+=1
    return fingers

def fingers_to_gestures(fingers):
    if fingers==0:
        return "Rock"
    elif fingers==2:
        return "Scissors"
    elif fingers==5:
        return "Paper"
    return "Invalid"

def decide_winner(player,computer):
    if player == computer:
        return "Draw"
    if (player == "Rock" and computer == "Scissors") or \
        (player=="Scissors" and computer=="Paper") or \
        (player=="Paper" and computer=="Rock"):
        return "Player"
    return "Computer"

def reset_game():
    global current_round,player_score,computer_score
    global countdown,last_time,game_state

    current_round=1
    player_score=0
    computer_score=0
    countdown=3
    last_time=time.time()
    game_state="countdown"

while True:
    ret,frame=cap.read()
    if not ret:
        break

    frame=cv2.flip(frame,1)
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    result=hands.process(rgb)

    h,w,_ =frame.shape

    if game_state=="countdown":
        if time.time() -last_time>=1:
            winsound.Beep(1000,200)
            countdown-=1
            last_time=time.time()
        if countdown ==0:
            game_state="play"
            countdown=3

        cv2.putText(frame,f"Round {current_round}",(200,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)

        cv2.putText(frame,str(countdown),(280,200),cv2.FONT_HERSHEY_SIMPLEX,4,(0,0,255),6)

    elif game_state=="play":
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                lm_list=[]
                for id,lm in enumerate(hand_landmarks.landmark):
                    cx,cy =int(lm.x*w),int(lm.y*h)
                    lm_list.append((id,cx,cy))

                fingers=count_fingers(lm_list)
                player_move=fingers_to_gestures(fingers)

                if player_move !='Invalid':
                    computer_move=random.choice(gestures)
                    winner=decide_winner(player_move,computer_move)

                    if winner =="Player":
                        player_score+=1
                    elif winner =="Computer":
                        computer_score+=1

                    game_state="result"
                    last_time=time.time()

                mp_draw.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)

    elif game_state=="result":
        cv2.putText(frame, f"You:{player_move}",(50,150),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.putText(frame, f"Computer: {computer_move}",(50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if time.time() - last_time>2:
            current_round+=1
            if current_round>rounds:
                game_state='final'
            else:
                game_state='countdown'
                last_time=time.time()

    elif game_state=='final':
        if player_score>computer_score:
            winner_text='YOU WIN!!'
        elif player_score<computer_score:
            winner_text='COMPUTER WINS!!'
        else:
            winner_text="DRAW!!"

        cv2.putText(frame,winner_text,(150,200),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,255),3)
        cv2.putText(frame,f"Score: {player_score}-{computer_score}",(180,250),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
        cv2.putText(frame, "Press R to Play Again",(130, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.putText(frame, "Press Q to Quit",(170, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imshow("Rock Paper Scissors Game",frame)

    key=cv2.waitKey(1) & 0xFF

    if key==ord('q'):
        break
    if key==ord('r') and game_state=="final":
        reset_game()


cap.release()
cv2.destroyAllWindows()