import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
#Check if GPU is available for computing
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

EPOCHS = 200
#0. Preprocess the data, categorize the string variables.
def preProcess(data_loc):
    df = pd.read_csv(data_loc)
    df = pd.get_dummies(df, columns=["fueltype", "aspiration", "doornumber", "carbody", "drivewheel", "enginelocation",  "cylindernumber", "enginetype", "fuelsystem"])
    #print(y.astype(float))
    #Shuffle input
    df = df.sample(frac=1).reset_index(drop=True)

    print(df)
    x_train = torch.tensor(df.iloc[0:160, 1:].values, dtype=torch.float32)
    y_train = torch.tensor(df.iloc[0:160, 0].values, dtype=torch.float32)
    #print(x)
    x_test = torch.tensor(df.iloc[160:, 1:].values, dtype=torch.float32)
    y_test = torch.tensor(df.iloc[160:, 0].values, dtype=torch.float32)
    print(x_train.shape)
    print(x_test.shape)
    return x_train, y_train, x_test, y_test

#1. Create the model/Parametrization
#We choose to override the torch.nn.Module class to create our own model.
class two_layer_model(torch.nn.Module):
    def __init__(self) -> None: #Constructor, but also we don't need *args and **kwargs
        super().__init__()
        self.input = torch.nn.Linear(52, 64)
        self.activation1 = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(64, 32)
        self.activation2 = torch.nn.LeakyReLU()
        self.linear2 = torch.nn.Linear(32, 1)
        self.activation3 = torch.nn.LeakyReLU()

    def forward(self, x): #Note that x should be an input tensor that has shape of 1, 52 (ie a batch size of 1)
        x = self.input(x)
        x = self.activation1(x)
        x = self.linear1(x)
        x = self.activation2(x)
        x = self.linear2(x)
        x = self.activation3(x)
        return x
    
    def string(self): #
        total_params = sum(p.numel() for p in self.parameters())
        
        return f"This is a two layer model, with 52 input nodes, 32 hidden nodes, and 1 output node. It has " + str(total_params)  + " parameters"
    
my_model = two_layer_model()


#1.5 Setup our tensorboard
writer = SummaryWriter('runs/Basic NN')
writer.add_graph(my_model, torch.rand([1, 52]))


#2. Create the loss function
loss_function = torch.nn.MSELoss(reduction="mean")

#3. Create the optimizer
optimizer = torch.optim.Adam(my_model.parameters(), lr=1e-3) #lr is the learning rate

#4. Create the training loop
def train_loop(x, y, model, loss_function, optimizer):
    for epoch in range(EPOCHS):
        #Forward pass
        step = 0
        for index, elem in enumerate(x): #This is more like stochastic gradient descent rather than batch gradient descent
            y_pred = model(elem)
            loss = loss_function(y_pred, y[index])
            #Backward pass
            loss.backward() #Compute the gradients
            optimizer.step() #Update the parameters
            optimizer.zero_grad() #Clear the gradients
            step += 1
        #Write to tensorboard
        writer.add_scalar('training loss',
                            loss / 1000,
                            epoch)
        print(f"Epoch {epoch} | Loss: {loss.item()}")
#5. Create the test loop
def test_loop(x, y, model):
    #print(x.shape)
    #print(y.shape)
    for index, elem in enumerate(x):
        y_pred = model(elem)
        loss = loss_function(y_pred, y[index])
        #Write test loss to tensorboard
        writer.add_scalar('test loss',
                    loss / 1000,
                    index)
        print(f"Test Loss: {loss.item()}")

#6. Create the main function with visualization
def main():
    #training_loc = "car_price_training.csv"
    #test_loc = "car_price_test.csv"
    data_loc = "car_prices.csv"
    x_train, y_train, x_test, y_test = preProcess(data_loc)
    train_loop(x_train, y_train, my_model, loss_function, optimizer)
    test_loop(x_test, y_test, my_model)
    writer.close()

#7. Run the main function!
main()
