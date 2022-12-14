import cv2 
import tkinter.filedialog

camera_input=False




if not camera_input:
    filename = tkinter.filedialog.askopenfilename()
    print(filename)

    frame = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    
    
else:
    webcam = cv2.VideoCapture(0)
    check, frame = webcam.read()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
key = cv2. waitKey(1)


while True:
    try:
        if camera_input:
            check, frame = webcam.read()
        #print(check) #prints true as long as the webcam is running
        #print(frame) #prints matrix values of each framecd 
        
        key = cv2.waitKey(1)
        image = frame
        #image = cv2.cvtColor(img , cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
            cv2.rectangle(image,(x-4,y-4),(x+w+4,y+h+4),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
        if camera_input:
            cv2.imshow("Capturing", image)
        if key == ord('s') or not camera_input:
            img = frame
            image = cv2.cvtColor(img , cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x,y,w,h) in faces:
                
                roi_gray = gray[y:y+h, x:x+w]

            # resize image
            output = cv2.resize(roi_gray, (50, 50))
            xx,yy,d=image.shape
            cv2.imwrite(filename='subjects_photos/' + input('file number (1~100)?') + '.png', img=output)
            
        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
        if not camera_input:
            print("Program ended.")
            cv2.destroyAllWindows()
            break
           
    except Exception as er:
        if "name 'y' is not defined" in str(er ):
            print('no face')
            break
        else:
            print(er)
 
