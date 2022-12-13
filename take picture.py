import cv2 , pygame
import tkinter.filedialog

camera_input=False



if not camera_input:
    filename = tkinter.filedialog.askopenfilename()
    print(filename)
else:
    webcam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
key = cv2. waitKey(1)

if camera_input:
    check, frame = webcam.read()
else:
    src = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    width = int(1080)
    height = int(src.shape[0] * 1080 /src.shape[1] )
    dsize = (width, height)
    # resize image
    framem = cv2.resize(src, dsize)
while True:
    try:
        if camera_input:
            check, frame = webcam.read()
        else:
            frame = framem
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
        cv2.imshow("Capturing", image)
        if key == ord('s'):
            img = frame
            image = cv2.cvtColor(img , cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            print(90)
            for (x,y,w,h) in faces:
                cv2.rectangle(image,(x-4,y-4),(x+w+4,y+h+4),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
            

            xx,yy,d=image.shape
            cv2.imwrite(filename='aa.jpg', img=gray)
            pi2=(pygame.transform.chop(
                pygame.image.load('aa.jpg'),(0,0,x,y)))
            pi=pygame.transform.scale(pygame.transform.chop(pi2,(h,w,xx-x,yy-y))
                                      ,(50,50))
            pygame.image.save(pi , 'subjects_photos/' + input('file name ?') + '.png')
        
            
        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
        
    except:
        print("  er.")
 
