import cv2
import json

drawing = False
start_point = None
state = {"current_type": "entrance"}
regions = {
    "entrance": [],
    "exit": [],
    "ignore": []
}

colors = {
    "entrance": (0, 255, 0),
    "exit": (0, 0, 255),
    "ignore": (180, 180, 180)
}

def draw_rectangle(event, x, y, flags, param):
    global drawing, start_point, regions, state
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(start_point[0], x), min(start_point[1], y)
        x2, y2 = max(start_point[0], x), max(start_point[1], y)
        regions[state["current_type"]].append((x1, y1, x2, y2))

def save_config(filename="roi_config.json"):
    with open(filename, "w") as f:
        json.dump(regions, f, indent=4)
    print(f"[ROI] Saved to {filename}")

def main(video_path="recordings/motion4.mp4"):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to load video frame.")
        return

    # Resize frame to 960px wide (preserve aspect ratio)
    max_width = 960
    scale = max_width / frame.shape[1]
    frame = cv2.resize(frame, (max_width, int(frame.shape[0] * scale)))
    resized_height = frame.shape[0]

    clone = frame.copy()
    window_name = "ROI Picker"

    # Create resizable window first
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, max_width, resized_height)
    cv2.setMouseCallback(window_name, draw_rectangle)

    print("Instructions:")
    print("  e = draw entrance  |  x = draw exit  |  i = draw ignore")
    print("  s = save and exit  |  q = quit without saving")

    while True:
        temp = clone.copy()
        for label, boxes in regions.items():
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(temp, (x1, y1), (x2, y2), colors[label], 2)
                cv2.putText(temp, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[label], 1)

        cv2.imshow(window_name, temp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("e"):
            state["current_type"] = "entrance"
            print("[ROI] Mode: Entrance")
        elif key == ord("x"):
            state["current_type"] = "exit"
            print("[ROI] Mode: Exit")
        elif key == ord("i"):
            state["current_type"] = "ignore"
            print("[ROI] Mode: Ignore")
        elif key == ord("s"):
            save_config()
            break
        elif key == ord("q"):
            print("Exited without saving.")
            break

    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
