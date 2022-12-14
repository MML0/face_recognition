import matplotlib.pyplot as plt
import numpy as np
import random , os , cv2  , time , pygame , tkinter.filedialog

np.random.seed(0)
random.seed(0)

model_folser = 'model'
model_name_load = '50x50-4l'
    

folder = 'subjects_photos'
list_dir = os.listdir('subjects_photos')
print(list_dir)
names = {0:'MML' , 1:'mml' , 2:'mml' , 3:'sol' , 4:'sol', 5:'sol'}

camera_input=False



if not camera_input:
    filename = tkinter.filedialog.askopenfilename()
    print(filename)

    frame = cv2.imread(filename, cv2.IMREAD_UNCHANGED)    
else:
    webcam = cv2.VideoCapture(0)
    check, frame = webcam.read()

print('all photos : ', len (list_dir))

photodata=[]
data = []
t=[]

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def load_photos():
    global data , y , image2 , frame 
    
    data = []
    t=[]
    y=[]
    if camera_input:
        check, frame = webcam.read()
    else :
         frame = cv2.imread(filename, cv2.IMREAD_UNCHANGED)   

    img = frame
    image = cv2.cvtColor(img , cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        output = cv2.resize(roi_gray, (50, 50))
        image2 = output
        #key = cv2.waitKey(2000)
        #cv2.imshow('a',output)
        #key = cv2.waitKey(2000)
        break

    for i in (list_dir):
        if (1) :
            photodata=[]

            image1 = cv2.imread(folder+"/"+i)
            #key = cv2.waitKey(2000)
            #cv2.imshow(str(i),image1)
            #key = cv2.waitKey(2000)

                    
            for j in range(50):
                for k in range(50):
                    photodata.append(image1[j][k][0]/255)
            for j in range(50):
                for k in range(50):
                    photodata.append(image2[j][k]/255)
            data.append(photodata)        
            t.append(0)         
                    
            #cv2.imshow('a',image1)
            #k = cv2.waitKey(10) & 0xff
            #time.sleep(0.021)
        
        
    y= np.array(t)

load_photos()


print('data: ',len(data))

class Layer :
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.1 * np.random.rand(n_inputs,n_neurons)-0.05
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
class activation :
    def forward(self,inputs):
        self.output = np.maximum(0, inputs)
class activation_softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1,keepdims=True))
        probabilities = exp_values/np.sum(exp_values, axis=1,keepdims=True)
        self.output = probabilities
class Loss:

    def calculate (self,output,y):
        sample_losses = self.forward(output,y)
        data_loss=np.mean(sample_losses)
        return data_loss
class Loss_C(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7,1-1e-7)
        if len(y_true.shape)==1:
            correct_confidences = y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape)==2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        negetive_log_likelhoods = -np.log(correct_confidences)
        #negetive_log_likelhoods = correct_confidences
        return negetive_log_likelhoods

class Loss_C2(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        outputlen = len(y_pred[0])
        loss_n = 0
        for i in range (samples):
            for j in range (outputlen):
                loss_n += abs(y_pred[i][j]-y_true[i][j])**2 

                
        return loss_n / samples



layer1 = Layer(5000,500)
layer2 = Layer(500,50)
layer3 = Layer(50,10)
layer4 = Layer(10,2)

#load last weights 

try :
    
    best_loss = 999999
    
    w1=open(model_folser + '/' + 'best_layer1_weights'+model_name_load+'.npy','rb')
    b1=open(model_folser + '/' + 'best_layer1_biases'+model_name_load+'.npy','rb')
    w2=open(model_folser + '/' + 'best_layer2_weights'+model_name_load+'.npy','rb')
    b2=open(model_folser + '/' + 'best_layer2_biases'+model_name_load+'.npy','rb')
    w3=open(model_folser + '/' + 'best_layer3_weights'+model_name_load+'.npy','rb')
    b3=open(model_folser + '/' + 'best_layer3_biases'+model_name_load+'.npy','rb')
    w4=open(model_folser + '/' + 'best_layer4_weights'+model_name_load+'.npy','rb')
    b4=open(model_folser + '/' + 'best_layer4_biases'+model_name_load+'.npy','rb')
    
    
    layer1.weights = np.load(w1 , allow_pickle=True)
    layer1.biases  = np.load(b1 , allow_pickle=True)
    layer2.weights = np.load(w2 , allow_pickle=True)
    layer2.biases  = np.load(b2 , allow_pickle=True)
    layer3.weights = np.load(w3 , allow_pickle=True)
    layer3.biases  = np.load(b3 , allow_pickle=True)
    layer4.weights = np.load(w4 , allow_pickle=True)
    layer4.biases  = np.load(b4 , allow_pickle=True)
    print('best weights loaded ☻')

except Exception as er:
    
    print(er)
    print('models are missing !')


activation1=activation()
activation2=activation()
activation3=activation()
activation4=activation_softmax()
loss_function = Loss_C()


while True :
    try:
        load_photos()

        #print(i)
        layer1.forward(data)
        activation1.forward(layer1.output)

        layer2.forward(activation1.output)
        activation2.forward(layer2.output)

        layer3.forward(activation2.output)
        activation3.forward(layer3.output)

        layer4.forward(activation3.output)
        activation4.forward(layer4.output)

        
        predictions = np.argmax(activation4.output,axis=1)
        # , activation4.output)
        #time.sleep(1)
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
         
            #print(predictions)
            pred2n=[]

            for pred in  str(activation4.output).replace('\n','').replace('[','').replace(']','').split(' ') :
                pred2n+=[float(pred)]
            predictions2=[]

            for k in range(1,len(pred2n)+1,2):
                predictions2+=[pred2n[k]]


            for k in range(len(predictions2)):
                if predictions2[k]== max(predictions2) and max(predictions2)>0.3:
                    print(names[k] , predictions2 )
            #print( predictions[2] )
        if camera_input:
            cv2.imshow("Capturing", image)
        else:
            break
        if key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
        
    except Exception as er:
        #print(er)
        pass
 
