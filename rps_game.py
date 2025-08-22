import cv2
import mediapipe as mp
import time
import numpy as np

# --- Constants ---
WINNING_SCORE = 5
FONT = cv2.FONT_HERSHEY_DUPLEX
# --- NEW: Wager Constants ---
STARTING_BALANCE = 100
WAGER_AMOUNT = 10
# Colors
C_BLUE = (255, 0, 0)
C_RED = (0, 0, 255)
C_GREEN = (0, 255, 0)
C_WHITE = (255, 255, 255)
C_BLACK = (0, 0, 0)

## --- UI Drawing Function ---
def draw_ui(canvas, game_state, scores, winner, countdown, gestures, balances):
    """Handles all drawing operations, now including player balances."""
    p1_score, p2_score = scores
    p1_bal, p2_bal = balances # Unpack balances
    ui_h, ui_w, _ = canvas.shape
    
    # --- Header ---
    cv2.putText(canvas, "PLAYER 1", (50, 90), FONT, 1.5, C_BLUE, 3)
    # --- NEW: Display Player 1 Balance ---
    cv2.putText(canvas, f"Balance: {p1_bal}", (50, 600), FONT, 1, C_WHITE, 2)

    score_text = f"{p1_score} - {p2_score}"
    text_size, _ = cv2.getTextSize(score_text, FONT, 2.5, 4)
    cv2.putText(canvas, score_text, ((ui_w - text_size[0]) // 2, 100), FONT, 2.5, C_WHITE, 4)
    
    cv2.putText(canvas, "PLAYER 2", (ui_w - 350, 90), FONT, 1.5, C_RED, 3)
    # --- NEW: Display Player 2 Balance ---
    cv2.putText(canvas, f"Balance: {p2_bal}", (ui_w - 350, 600), FONT, 1, C_WHITE, 2)


    # --- Player View Borders ---
    cv2.rectangle(canvas, (50, 120), (610, 540), C_BLUE, 3)
    cv2.rectangle(canvas, (ui_w - 610, 120), (ui_w - 50, 540), C_RED, 3)

    # --- Footer / Status Bar (Dynamic) ---
    p1_gesture, p2_gesture = gestures
    status_bar_y = ui_h - 70
    status_text = ""
    if game_state == "PLAY":
        if p1_gesture == "Unknown" and p2_gesture == "Unknown": status_text = "Show your gestures!"
        elif p1_gesture != "Unknown" and p2_gesture == "Unknown": status_text = "Waiting for Player 2..."
        elif p1_gesture == "Unknown" and p2_gesture != "Unknown": status_text = "Waiting for Player 1..."
        text_size, _ = cv2.getTextSize(status_text, FONT, 1.5, 2)
        cv2.putText(canvas, status_text, ((ui_w - text_size[0]) // 2, status_bar_y), FONT, 1.5, C_WHITE, 2)
    
    # --- Centered Status Messages ---
    if game_state == "COUNTDOWN":
        cv2.putText(canvas, str(countdown), ((ui_w - 70) // 2, ui_h // 2 + 50), FONT, 6, C_RED, 10)
    elif game_state == "RESULT":
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, ui_h // 2 - 50), (ui_w, ui_h // 2 + 50), C_BLACK, -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
        result_text = f"Round Winner: {winner}"
        text_size, _ = cv2.getTextSize(result_text, FONT, 2, 3)
        cv2.putText(canvas, result_text, ((ui_w - text_size[0]) // 2, ui_h // 2 + 15), FONT, 2, C_GREEN, 3)
    elif game_state == "GAME_OVER":
        final_winner = "Player 1" if p1_score >= WINNING_SCORE else "Player 2"
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, ui_h // 2 - 100), (ui_w, ui_h // 2 + 100), C_BLACK, -1)
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
        cv2.putText(canvas, "GAME OVER", ((ui_w - 450) // 2, ui_h // 2 - 20), FONT, 3, C_RED, 7)
        cv2.putText(canvas, f"{final_winner} WINS!", ((ui_w - 480) // 2, ui_h // 2 + 60), FONT, 2.5, C_GREEN, 5)

# --- Gesture Classification (no changes needed) ---
def classify_gesture(hand_landmarks):
    landmarks = hand_landmarks.landmark; mp_hands = mp.solutions.hands
    fingertip_y = [landmarks[mp_hands.HandLandmark.THUMB_TIP].y, landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y, landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y, landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y, landmarks[mp_hands.HandLandmark.PINKY_TIP].y]
    pip_y = [landmarks[mp_hands.HandLandmark.THUMB_IP].y, landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y, landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y, landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y, landmarks[mp_hands.HandLandmark.PINKY_PIP].y]
    fingers_extended = [tip < pip for tip, pip in zip(fingertip_y, pip_y)]
    if fingers_extended[1] and fingers_extended[2] and not fingers_extended[3] and not fingers_extended[4]: return "Scissors"
    elif all(fingers_extended): return "Paper"
    elif not any(fingers_extended[1:]): return "Rock"
    return "Unknown"

# --- Function to determine the round winner (no changes needed) ---
def get_round_winner(g1, g2):
    if g1 == "Unknown" or g2 == "Unknown": return None
    if g1 == g2: return "Tie"
    elif (g1 == "Rock" and g2 == "Scissors") or (g1 == "Scissors" and g2 == "Paper") or (g1 == "Paper" and g2 == "Rock"): return "Player 1"
    else: return "Player 2"

# --- Game State and Setup ---
gameState = "COUNTDOWN"; player1_score, player2_score = 0, 0; last_round_winner = None
# --- NEW: Wager Management Variables ---
player1_balance, player2_balance = STARTING_BALANCE, STARTING_BALANCE

countdown_start_time = time.time(); result_display_start_time = 0
mp_drawing = mp.solutions.drawing_utils; mp_hands = mp.solutions.hands

# --- Webcam setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened(): raise IOError("Cannot open webcam")
ret, frame = cap.read()
if not ret: raise IOError("Could not read frame from webcam")
cam_h, cam_w, _ = frame.shape
ui_h, ui_w = 720, 1280

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        ui_canvas = np.zeros((ui_h, ui_w, 3), dtype=np.uint8)
        player_view_w, player_view_h = 560, 420
        p1_view = cv2.resize(frame, (player_view_w, player_view_h))
        p2_view = p1_view.copy()
        
        player1_gesture, player2_gesture = "Unknown", "Unknown"
        countdown_value = 0

        # --- Main Game Logic Loop ---
        if gameState == "COUNTDOWN":
            elapsed_time = time.time() - countdown_start_time
            countdown_value = 3 - int(elapsed_time)
            if countdown_value <= 0: gameState = "PLAY"
        
        elif gameState == "PLAY":
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * cam_w
                    gesture = classify_gesture(hand_landmarks)
                    if wrist_x < cam_w // 2:
                        player1_gesture = gesture
                        mp_drawing.draw_landmarks(p1_view, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    else:
                        player2_gesture = gesture
                        mp_drawing.draw_landmarks(p2_view, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            round_winner = get_round_winner(player1_gesture, player2_gesture)
            if round_winner:
                # --- NEW: Update Scores and Balances ---
                if round_winner == "Player 1":
                    player1_score += 1
                    player1_balance += WAGER_AMOUNT
                    player2_balance -= WAGER_AMOUNT
                elif round_winner == "Player 2":
                    player2_score += 1
                    player2_balance += WAGER_AMOUNT
                    player1_balance -= WAGER_AMOUNT
                
                last_round_winner, result_display_start_time = round_winner, time.time()
                gameState = "RESULT"

        elif gameState == "RESULT":
            if player1_score >= WINNING_SCORE or player2_score >= WINNING_SCORE or player1_balance <= 0 or player2_balance <= 0:
                gameState = "GAME_OVER"
            elif time.time() - result_display_start_time > 2.5:
                gameState, countdown_start_time = "COUNTDOWN", time.time()

        # --- Place Player Views and Gestures onto Canvas ---
        ui_canvas[120:120+player_view_h, 50:50+player_view_w] = p1_view
        ui_canvas[120:120+player_view_h, ui_w - player_view_w - 50:ui_w - 50] = p2_view
        cv2.putText(ui_canvas, f"Gesture: {player1_gesture}", (60, 580), FONT, 1, C_WHITE, 2)
        cv2.putText(ui_canvas, f"Gesture: {player2_gesture}", (ui_w - 520, 580), FONT, 1, C_WHITE, 2)
        
        # --- Call the centralized drawing function ---
        draw_ui(ui_canvas, gameState, (player1_score, player2_score), last_round_winner, countdown_value, 
                (player1_gesture, player2_gesture), (player1_balance, player2_balance)) # Pass balances to UI

        cv2.imshow('Rock Paper Scissor UI', ui_canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()