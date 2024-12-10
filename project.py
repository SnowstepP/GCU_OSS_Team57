# 비디오 실행
video_capture = cv2.VideoCapture(0)

prev_faces = []

while True:
    # ret, frame 반환
    ret, frame = video_capture.read()
    
    if not ret:
        break

    # 얼굴인식을 위해 gray 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 인식
    # scaleFactor이 1에 가까울수록 표정 인식이 잘 되고 멀 수록 잘 안됨
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  # 출력
    cv2.imshow('Expression Recognition', frame)

    # esc 누를 경우 종료
    key = cv2.waitKey(25)
    if key == 27:
        break

if video_capture.Opened():
    video_capture.release()
cv2.destroyAllWindows()
