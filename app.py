from flask import Flask, request, jsonify
import pickle
import cv2
import numpy as np

app = Flask(__name__)

# Load your trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get the uploaded file
    file = request.files['file']
    video_path = 'uploaded_video.mp4'
    file.save(video_path)

    # Example: Extract frames and convert them to features for your model
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Process the frame as needed (resize, normalize, etc.)
        frames.append(frame)
    
    cap.release()
    cv2.destroyAllWindows()

    # Example: Predict based on the processed frames
    # This depends on how your model was trained
    # For simplicity, let's assume your model can handle a list of frames
    frames_np = np.array(frames)  # Convert list to numpy array
    predictions = model.predict(frames_np)
    result = predictions.tolist()  # Convert numpy array to list

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
