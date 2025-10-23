#import cv2
#import mediapipe as mp
#
#mp_drawing = mp.solutions.drawing_utils
#mp_pose = mp.solutions.pose
#
#cap = cv2.VideoCapture(r"Tom Thrower.mp4")
#
#with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#    while True:
#        ret, frame = cap.read()
#        if not ret:
#            break
#        
#        h, w, _ = frame.shape
#        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#        results = pose.process(rgb)
#
#        # 📐 Define stance box dynamically — center and bigger
#        box_width = int(w * 0.3)   # 60% of frame width
#        box_height = int(h * 0.80)  # 70% of frame height
#
#        box_x1 = (w - box_width) // 2
#        box_y1 = (h - box_height) // 2
#        box_x2 = box_x1 + box_width
#        box_y2 = box_y1 + box_height
#
#        box_color = (0, 255, 0)  # green
#        warning_lines = []       # collect warnings as list
#
#        if results.pose_landmarks:
#            for id, lm in enumerate(results.pose_landmarks.landmark):
#                x, y = int(lm.x * w), int(lm.y * h)
#                if not (box_x1 <= x <= box_x2 and box_y1 <= y <= box_y2):
#                    box_color = (0, 0, 255)  # red
#                    landmark_name = mp_pose.PoseLandmark(id).name
#                    warning_lines.append(f"{landmark_name} out of box")
#
#        # 🟩 Draw stance box
#        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), box_color, 2)
#
#        # ✨ Draw landmarks
#        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#
#        # 🛑 Show warning — line by line
#        if warning_lines:
#            for i, line in enumerate(warning_lines):
#                y_position = 30 + i * 25
#                cv2.putText(frame, line, (30, y_position),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
#
#        cv2.imshow("Golfer Position", frame)
#        if cv2.waitKey(1) & 0xFF == 27:
#            break
#
#cap.release()
#cv2.destroyAllWindows()
#





import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ✅ Key landmarks to monitor
keypoints_ids = [
    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp_pose.PoseLandmark.LEFT_WRIST.value,
    mp_pose.PoseLandmark.RIGHT_WRIST.value,
    mp_pose.PoseLandmark.LEFT_ANKLE.value,
    mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    mp_pose.PoseLandmark.NOSE.value
]

# Reference for top and bottom (for distance calculation)
top_id = mp_pose.PoseLandmark.NOSE.value
bottom_ids = [mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value]

# 🎥 Load input video
cap = cv2.VideoCapture(r"tiger-slow-mo.mp4")

# 📝 Set up video writer to save output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec for .mp4
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('tiger_feedback_output.mp4', fourcc, fps, (frame_width, frame_height))

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        # 📐 Define stance box dynamically (centered)
        box_width = int(w * 0.28)   # adjust width as needed
        box_height = int(h * 0.79)  # adjust height
        box_x1 = (w - box_width) // 2
        box_y1 = (h - box_height) // 2
        box_x2 = box_x1 + box_width
        box_y2 = box_y1 + box_height

        feedback_lines = []
        box_color = (0, 255, 0)  # green initially

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # 🧭 Check LEFT / RIGHT / TOP / BOTTOM position
            for pid in keypoints_ids:
                x = int(landmarks[pid].x * w)
                y = int(landmarks[pid].y * h)
                keypoint_name = mp_pose.PoseLandmark(pid).name

                if x < box_x1:
                    feedback_lines.append(f"{keypoint_name}: Move RIGHT to stay inside the box")
                    box_color = (0, 0, 255)
                elif x > box_x2:
                    feedback_lines.append(f"{keypoint_name}: Move LEFT to stay inside the box")
                    box_color = (0, 0, 255)

                if y < box_y1:
                    feedback_lines.append(f"{keypoint_name}: Lower stance / Move back")
                    box_color = (0, 0, 255)
                elif y > box_y2:
                    feedback_lines.append(f"{keypoint_name}: Move forward / straighten up")
                    box_color = (0, 0, 255)

            # 📏 Distance feedback using top and bottom points
            top_y = landmarks[top_id].y * h
            bottom_y = (landmarks[bottom_ids[0]].y + landmarks[bottom_ids[1]].y) / 2 * h
            body_height = bottom_y - top_y
            box_height_range = box_height

            min_height = 0.5 * box_height_range
            max_height = 0.9 * box_height_range

            if body_height < min_height:
                feedback_lines.append("👣 Move CLOSER to the camera")
                box_color = (0, 0, 255)
            elif body_height > max_height:
                feedback_lines.append("📏 Move AWAY from the camera")
                box_color = (0, 0, 255)

        # 🟥 Draw stance box
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), box_color, 3)

        # 🧍 Draw pose landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 📝 Display feedback on frame
        for i, line in enumerate(feedback_lines):
            cv2.putText(frame, line, (30, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # 🖨️ Print feedback to console too
        if feedback_lines:
            print("\n".join(feedback_lines))

        # 💾 Write frame to output video
        out.write(frame)

        # 👀 Show on screen
        cv2.imshow("Golfer Position Feedback", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
out.release()  # 🛑 close video writer
cv2.destroyAllWindows()

