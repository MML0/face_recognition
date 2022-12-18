import numpy as np
import random ,math
import matplotlib.pyplot as plt

random.seed(0)
np.random.seed(0)

data = []
t=[]

for i in range(1920):
    a1=(random.randint(0,314)/100)
    #a2=(random.random())
    #a3=(random.random())
    #a4=(random.random())
    data.append([a1])#,a2,a3,a3])
    #t.append([(4*a1),0])
    #t.append([a1,a2])
    t.append([math.sin(a1)])

y= np.array(t)

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
        negetive_log_likelhoods = -np.log(correct_confidences+0.000000000000000001)
        #negetive_log_likelhoods = correct_confidences
        return negetive_log_likelhoods

class Loss_C2(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        outputlen = len(y_pred[0])
        loss_n = 0
        for i in range (samples):
            for j in range (outputlen):
                loss_n += abs(y_pred[i][j]-y_true[i][j])

                
        return loss_n / samples



def test(a1,a2=0,a3=0,a4=0):
    data = [a1]#,a2,a3,a4]
    layer1.forward(data)
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    layer3.forward(activation2.output)
    activation3.forward(layer3.output)

    #print(activation2.output)
    #print(y)
    predictions = np.argmax(layer3.output,axis=1)
    #acc = np.mean(predictions==y)
    #print(predictions)
    loss = loss_function.calculate(layer3.output,y)
    #print('test acc:',round(acc*100000)/1000)
    #print('test loss:',loss)
    #print(predictions,layer3.output)
    return layer3.output[0][0]
    print(layer3.output)




layer1 = Layer(1,10)
layer2 = Layer(10,10)
layer3 = Layer(10,1)

activation1=activation()
activation2=activation()
activation3=activation()#activation_softmax()
loss_function = Loss_C2()


best_loss =99999
best_layer1_weights = layer1.weights.copy()
best_layer1_biases  = layer1.biases.copy()
best_layer2_weights = layer2.weights.copy()
best_layer2_biases  = layer2.biases.copy() 
best_layer3_weights = layer3.weights.copy()
best_layer3_biases  = layer3.biases.copy() 


lr = 0.5
rv = lr
train_count = 1000

for i in range(train_count):
    
    layer1.weights += rv * np.random.rand(1,10)-rv/2
    layer1.biases  += rv * np.random.rand(1,10)-rv/2
    layer2.weights += rv * np.random.rand(10,10)-rv/2
    layer2.biases  += rv * np.random.rand(1,10)-rv/2
    layer3.weights += rv * np.random.rand(10,1)-rv/2
    layer3.biases  += rv * np.random.rand(1,1)-rv/2
    
    layer1.forward(data)
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    layer3.forward(activation2.output)
    activation3.forward(layer3.output)

    #print(activation2.output)
    #print(y)
    predictions = np.argmax(layer3.output,axis=1)
    acc = 1#np.mean(predictions==y)
    #print(predictions)
    loss = loss_function.calculate(layer3.output,y)
    
    if loss<best_loss:
        print('loss:',loss)
        print('rv : ' , rv,'\n')
        rv= lr
        #print('acc:',acc)
        best_layer1_weights = layer1.weights.copy()
        best_layer1_biases  = layer1.biases.copy()
        best_layer2_weights = layer2.weights.copy()
        best_layer2_biases  = layer2.biases.copy()         
        best_layer3_weights = layer3.weights.copy()
        best_layer3_biases  = layer3.biases.copy()         
        best_loss = loss
    else:
        rv -= rv/40
        layer1.weights = best_layer1_weights.copy()
        layer1.biases  = best_layer1_biases.copy()
        layer2.weights = best_layer2_weights.copy()
        layer2.biases  = best_layer2_biases.copy()
        layer3.weights = best_layer3_weights.copy()
        layer3.biases  = best_layer3_biases.copy()




    if i % (train_count/10)==0:
        ycor = []
        xcor = []
        xcor2 =[]
        for j in range(314):
            ycor+=[j/100]
            xcor += [test(j/100)]
            xcor2+= [math.sin(j/100)]



        plt.plot(ycor , xcor , label = 'output of the neuron network') 
        plt.plot(ycor , xcor2 , label = 'math.sin') 
        plt.plot([0] , [0] , label = 'i = ' + str(i)) 



        plt.ylim(-0.5,1.2)
        plt.xlim(-0.2,3.5)

        plt.legend()
        plt.show()












