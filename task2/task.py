import cv2 as cv
from ultralytics import YOLO


WIDTH = 1920
HEIGHT = 1080

center = (WIDTH //2, HEIGHT//2)  # Changed to tuple


def detect():
    # Create and set window size
    cv.namedWindow("YOLOv11 Detection", cv.WINDOW_NORMAL)
    cv.resizeWindow("YOLOv11 Detection", WIDTH, HEIGHT)
    
    source = cv.VideoCapture(0)
    # Set the width, height and FPS
    source.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
    source.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    source.set(cv.CAP_PROP_FPS, 30)

    model = YOLO("lastv6.pt")
    while True:
        _, frame = source.read()
        frame = cv.flip(frame,1)
        # Run detection and get results with annotations
        results = model(frame)
        boxPos = None

        # Check if there are any detections
        if len(results[0].boxes) > 0:  # If there are any detections
            if results[0].boxes.cls[0] == 1:  # If the first detection is class 1
                ver = "top"
                hor = "left"
                boxPos = results[0].boxes.xywh
                print("Detection found! Box position:", boxPos)
                boxCenter = (int(boxPos[0][0] + boxPos[0][2] // 2), int(boxPos[0][1] + boxPos[0][3] //2))
                if boxCenter[0] > center[0]:
                    hor = "right"
                if boxCenter[1] > center[1]:
                    ver = "bottom"
                 
                cv.putText(frame, f"Center is: {ver} {hor}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 
                          1, (0, 255, 0), 2, cv.LINE_AA)
        
        # Plot results on the frame
        annotated_frame = results[0].plot()
        cv.circle(annotated_frame, center, 2, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
        
        
        
        s
        
        
        # Display the annotated frame
        cv.imshow("YOLOv11 Detection", annotated_frame)
        
        if cv.waitKey(1) == ord("q"):
            break
    
    # Release resources
    source.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    detect()