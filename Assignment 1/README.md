# ELL409 Assignment
To run the code, execute the file "Main.ipynb".<br />
A model can be made as following: <br />
```
model = NeuralNet(hidden_dims, input_dim=28*28, 
                  num_classes=10, reg=0.0,
                  weight_scale=1e-2, activation='relu',
                  loss_fn='softmax')
```
* hidden_dims is a list of number of neurons in each hidden layer. For e.g. to make a Neural network with 2 hidden
layers having 200 neurons in first layer and 100 in second, [200, 100] should be passed.
* If dimension of examples are d1 X d2 X ... X dk, then input_dim should be d1 * d2 * ... * dk.
* num_classes represent number the number of output classes.
* reg refers to the  vlaue of lambda used in regularization.
* weight_scale refers to the initial weights.
* activation can be of two types: 'relu' or 'sigmoid'.
* loss function can be of two types: 'softmax' and 'svm'.

To train a model, an object of Solver Class has to be created. This can be done as following:
```
solver = Solver(model, data, lr=1e-3, 
                lr_decay=1.0, batch_size=100,
                num_epochs=10, print_every=10, 
                verbose=True)
solver.train()
```
* model is the model which has to be trained.
* data is a python dictionary as following:
```
    data = {
      'X_train': # training data
      'y_train': # training labels
      'X_val': # validation data
      'y_val': # validation labels
    }
```
* lr is the learning rate.
* lr_decay is the factor by which learning rate would be multiplied after each epoch.
* batch_size is  the number of training examples used in one iteration to train the model.
* num_epochs is the number of epochs for which the model would be trained.
* print_every refers to the number of iterations after which loss would be printed if verbose is True.
* verbose is a boolean value which refers to whether training and validation accuracy have to be
 printed after each epoch.
 
<br />
To print the value of a given weight after given epoch, use the following command:

```
print(solver.weights_history['W1'][5][8][15])
```
This prints the value of weight linking the (8+1)th neuron of first layer to
 (15+1)th neuron of next layer after 5 epochs.
 
 To find the value of neuron for a given input, use the following command:
 ```
x = data['X_train'][0]                              # First Training Example
print(model.find_output(x)[3][6])
``` 
 This prints the value of (6+1)th neuron of the third layer for input x (0 refers to input layer).
 