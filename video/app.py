import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# 페이지 설정
st.set_page_config(page_title="실시간 동작 인식", layout="wide")

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_shoulder_angle(shoulder, elbow, is_left=False):
    # 수직 기준선 생성
    vertical_point = [shoulder[0], shoulder[1] + 100]
    
    # 벡터 계산
    vector1 = np.array([vertical_point[0] - shoulder[0], vertical_point[1] - shoulder[1]])
    vector2 = np.array([elbow[0] - shoulder[0], elbow[1] - shoulder[1]])
    
    # 각도 계산
    dot_product = np.dot(vector1, vector2)
    norms = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    
    angle = np.arccos(dot_product / norms)
    angle = np.degrees(angle)
    
    # 최종 각도 계산
    if is_left:
        if elbow[1] > shoulder[1]:
            final_angle = 0 if angle < 10 else angle
        else:
            if angle <= 90:
                final_angle = 180 - angle
            else:
                final_angle = angle
    else:
        if elbow[1] > shoulder[1]:
            final_angle = 0 if angle < 10 else angle
        else:
            if angle <= 90:
                final_angle = 180 - angle
            else:
                final_angle = angle
    
    return min(180, max(0, final_angle))

def main():
    st.title("실시간 동작 인식")
    
    # 웹캠 프레임 표시할 위치
    frame_placeholder = st.empty()
    
    # 웹캠 시작
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        
        # MediaPipe 처리
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # 좌표 추출
            landmarks = results.pose_landmarks.landmark
            
            # 왼쪽 어깨 각도 계산
            left_shoulder = [int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width),
                           int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height)]
            left_elbow = [int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * width),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * height)]
            
            # 오른쪽 어깨 각도 계산
            right_shoulder = [int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width),
                            int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)]
            right_elbow = [int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * width),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * height)]
            
            # 각도 계산
            left_angle = calculate_shoulder_angle(left_shoulder, left_elbow, is_left=True)
            right_angle = calculate_shoulder_angle(right_shoulder, right_elbow, is_left=False)
            
            # 각도 표시
            cv2.putText(image_rgb, f'L: {int(left_angle)}', 
                       (left_shoulder[0] - 50, left_shoulder[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image_rgb, f'R: {int(right_angle)}', 
                       (right_shoulder[0] + 10, right_shoulder[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 프레임 표시
        frame_placeholder.image(image_rgb)
        
    cap.release()

if __name__ == '__main__':
    main() 