#import matplotlib.pyplot as plt
import numpy as np
import random , os , cv2  , time 

np.random.seed(0)
random.seed(0)

model_name = '50x50-4l'
model_name_load = '50x50-4l'

list_dirf = os.listdir()
model_folder = 'model'
folder = 'data'
list_dir = os.listdir('data')
print('photos : ', len (list_dir))
a_data = 1400





photodata=[]
data = []
t=[]

for i in range(a_data):
    try :
        photodata=[]
        a1=random.randint(1,len(list_dir)-1)
        a2=random.randint(0,1)
        image1 = cv2.imread(folder+"/"+list_dir[a1])
        subnum = list_dir[a1].split('-')[0][1:]
        subindexnum = list_dir[a1].split('-')[1]
        if a2 ==0: #another person => 0
            a3=random.randint(1,len(list_dir)-1)
            subnum2 = list_dir[a3].split('-')[0][1:-1]
            if  subnum2 ==  subnum :
                a3+=1
            subnum2 = list_dir[a3].split('-')[0][1:-1]
            image2 = cv2.imread(folder+"/"+list_dir[a3])
        if a2 ==1: #same person another photo=> 1
            a4=random.randint(1,6)
            
            if  subindexnum[0:-4] ==  str(a4) :
                a4+=1
            
            image2 = cv2.imread(folder+"/m"+subnum+'-'+str(a4)+'.png')
            
        for j in range(50):
            for k in range(50):
                photodata.append(image1[j][k][0]/255)
        for j in range(50):
            for k in range(50):
                photodata.append(image2[j][k][0]/255)
        data.append(photodata)        
        t.append(a2)         
            
    except :
        print(i)
        
    
y= np.array(t)
cv2.destroyAllWindows()

print('data set: ',len(data))


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
def save ():
    w1=open(model_folder + '/' + 'best_layer1_weights'+model_name+'.npy','wb')
    b1=open(model_folder + '/' + 'best_layer1_biases'+model_name+'.npy','wb')
    w2=open(model_folder + '/' + 'best_layer2_weights'+model_name+'.npy','wb')
    b2=open(model_folder + '/' + 'best_layer2_biases'+model_name+'.npy','wb')
    w3=open(model_folder + '/' + 'best_layer3_weights'+model_name+'.npy','wb')
    b3=open(model_folder + '/' + 'best_layer3_biases'+model_name+'.npy','wb')
    w4=open(model_folder + '/' + 'best_layer4_weights'+model_name+'.npy','wb')
    b4=open(model_folder + '/' + 'best_layer4_biases'+model_name+'.npy','wb')

    best_loss_file = open(model_folder + '/' + 'best_loss'+model_name+'.txt','w')
    best_loss_file.write(str(best_loss))
    best_loss_file.close()

    np.save(w1, best_layer1_weights.copy() )
    np.save(b1, best_layer1_biases.copy()  )
    np.save(w2, best_layer2_weights.copy() )
    np.save(b2, best_layer2_biases.copy()  )
    np.save(w3, best_layer3_weights.copy() )
    np.save(b3, best_layer3_biases.copy()  )
    np.save(w4, best_layer4_weights.copy() )
    np.save(b4, best_layer4_biases.copy()  )
    print('saved \n')

def test(a1,a2=0,a3=0,a4=0):
    data = [a1,a2,a3,a4]
    layer1.forward(data)
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    layer3.forward(activation2.output)
    activation3.forward(layer3.output)

    #print(activation2.output)
    #print(y)
    predictions = np.argmax(activation3.output,axis=1)
    #acc = np.mean(predictions==y)
    #print(predictions)
    loss = loss_function.calculate(activation3.output,y)
    #print('test acc:',round(acc*100000)/1000)
    #print('test loss:',loss)
    #print(predictions,layer3.output)
    print(layer3.output)



layer1 = Layer(5000,500)
layer2 = Layer(500,50)
layer3 = Layer(50,10)
layer4 = Layer(10,2)

#load last weights

try :
    
    best_loss = 999999
    
    w1=open(model_folder + '/' + 'best_layer1_weights'+model_name_load+'.npy','rb')
    b1=open(model_folder + '/' + 'best_layer1_biases'+model_name_load+'.npy','rb')
    w2=open(model_folder + '/' + 'best_layer2_weights'+model_name_load+'.npy','rb')
    b2=open(model_folder + '/' + 'best_layer2_biases'+model_name_load+'.npy','rb')
    w3=open(model_folder + '/' + 'best_layer3_weights'+model_name_load+'.npy','rb')
    b3=open(model_folder + '/' + 'best_layer3_biases'+model_name_load+'.npy','rb')
    w4=open(model_folder + '/' + 'best_layer4_weights'+model_name_load+'.npy','rb')
    b4=open(model_folder + '/' + 'best_layer4_biases'+model_name_load+'.npy','rb')
    
    
    layer1.weights = np.load(w1 , allow_pickle=True)
    layer1.biases  = np.load(b1 , allow_pickle=True)
    layer2.weights = np.load(w2 , allow_pickle=True)
    layer2.biases  = np.load(b2 , allow_pickle=True)
    layer3.weights = np.load(w3 , allow_pickle=True)
    layer3.biases  = np.load(b3 , allow_pickle=True)
    layer4.weights = np.load(w4 , allow_pickle=True)
    layer4.biases  = np.load(b4 , allow_pickle=True)
    print('best weights loaded ☻')
    print('loading loss ')
    best_loss_filer = open(model_folder + '/' + 'best_loss'+model_name_load+'.txt','r')
    best_loss = eval (best_loss_filer.read())
    print('OK !')

except Exception as er:
    
    print(er, '\n making new model or loss ...')

activation1=activation()
activation2=activation()
activation3=activation()
activation4=activation_softmax()
loss_function = Loss_C()

#best_loss = 70.554
#best_loss = 120.38655649204002

best_layer1_weights = layer1.weights.copy()
best_layer1_biases  = layer1.biases.copy()
best_layer2_weights = layer2.weights.copy()
best_layer2_biases  = layer2.biases.copy() 
best_layer3_weights = layer3.weights.copy()
best_layer3_biases  = layer3.biases.copy() 
best_layer4_weights = layer4.weights.copy()
best_layer4_biases  = layer4.biases.copy() 




lr=0.1

best_loss += 1
rv= 0
for i in range(10000000):
    #print(i)
    layer1.weights += rv * np.random.rand(5000,500)-rv/2
    layer1.biases  += rv * np.random.rand(1,500)-rv/2
    layer2.weights += rv * np.random.rand(500,50)-rv/2
    layer2.biases  += rv * np.random.rand(1,50)-rv/2
    layer3.weights += rv * np.random.rand(50,10)-rv/2
    layer3.biases  += rv * np.random.rand(1,10)-rv/2
    layer4.weights += rv * np.random.rand(10,2)-rv/2
    layer4.biases  += rv * np.random.rand(1,2)-rv/2
    
    layer1.forward(data)
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    layer3.forward(activation2.output)
    activation3.forward(layer3.output)

    layer4.forward(activation3.output)
    activation4.forward(layer4.output)

    #print(activation2.output)
    #print(y)
    predictions = np.argmax(activation4.output,axis=1)
    acc = np.mean(predictions==y)
    #print(predictions)
    loss = loss_function.calculate(activation4.output,y)
    #print('loss : ',loss , rv)
    
    if loss<best_loss:
        print('\nrv : ' , rv)
        rv= lr
        print('acc : ', round(acc*100000)/1000)
        print('loss : ', loss ,'\ndelta loss : ', best_loss-loss )
        save()
        best_layer1_weights = layer1.weights.copy()
        best_layer1_biases  = layer1.biases.copy()
        best_layer2_weights = layer2.weights.copy()
        best_layer2_biases  = layer2.biases.copy()         
        best_layer3_weights = layer3.weights.copy()
        best_layer3_biases  = layer3.biases.copy()         
        best_layer4_weights = layer4.weights.copy()
        best_layer4_biases  = layer4.biases.copy()         
        best_loss = loss
    else:
        rv -= rv/10
        layer1.weights = best_layer1_weights.copy()
        layer1.biases  = best_layer1_biases.copy()
        layer2.weights = best_layer2_weights.copy()
        layer2.biases  = best_layer2_biases.copy()
        layer3.weights = best_layer3_weights.copy()
        layer3.biases  = best_layer3_biases.copy()
        layer4.weights = best_layer4_weights.copy()
        layer4.biases  = best_layer4_biases.copy()
    if i %9000==0:
        print('\nrv : ' , rv)
        print('acc : ',round(acc*100000)/1000)
        print('loss : ',loss)
    if i %50==0:
        print(i)


