import cv2
from flask import Flask, render_template, Response
import time

app = Flask(__name__)
address = "https://192.168.88.212:8080/video"
video = cv2.VideoCapture(address)
model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if model.empty():
    print("Error: Cascade file not loaded properly")

crop_started = False
start_time = None

def gen_frames():
    global crop_started, start_time

    while True:
        success, frame = video.read()
        if not success:
            break

        if not crop_started:
            start_time = time.time()
            crop_started = True

        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time >= 10:
            try:
                # Detect faces in the original frame
                faces = model.detectMultiScale(frame)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (220, 0, 0), 2)
                
                # Crop faces from the original frame and yield both the original and cropped frames
                for (x, y, w, h) in faces:
                    crop_img = frame[y:y+h, x:x+w]
                    ret, buffer = cv2.imencode('.jpg', crop_img)
                    crop_frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + crop_frame + b'\r\n')

            except Exception as e:
                print(f"Error in gen_frames: {e}")
                continue

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('test.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
