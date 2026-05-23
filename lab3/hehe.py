
import torch
torch.__version__

from torch import nn 
import matplotlib.pyplot as plt



weight = 0.7
bias = 0.3

#creating values for X
start = 0
end = 1
step = 0.02
X = torch.arange(start , end , step).unsqueeze(dim = 1)
y = weight * X + bias


X[:10] , y[:10]


train_split = int(0.8 * len(X))
X_train , y_train = X[:train_split] , y[:train_split]
X_test , y_test = X[train_split:] , y[train_split:]
len(X_train), len(y_train), len(X_test), len(y_test)




import matplotlib.pyplot as plt

def plot_predictions(train_data = X_train,
                    train_labels = y_train,
                    test_data = X_test,
                    test_labels = y_test,
                    predictions = None) :
    plt.figure(figsize = (6,4))
    # plot train data in blue
    plt.scatter(train_data , train_labels , c='b' , s=4 , label = "Training data")
    # plot the test data in red
    plt.scatter(test_data , test_labels , c='r' , s = 4 , label = "Testing Data")
    # plot the predictions in yellow (if exits)
    if predictions is not None :
        # predictions were made on test data
        plt.scatter(test_data , predictions , c ='y' , s=4 , label = "prediction data")
    plt.legend(prop ={"size" : 14})


plot_predictions()


class LinearRegressionModel(nn.Module): # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
    def __init__(self):
        super().__init__() 
        self.weights = nn.Parameter(torch.randn(1, # <- start with random weights (this will get adjusted as the model learns)
                                                dtype=torch.float), # <- PyTorch loves float32 by default
                                   requires_grad=True) # <- can we update this value with gradient descent?)

        self.bias = nn.Parameter(torch.randn(1, # <- start with random bias (this will get adjusted as the model learns)
                                            dtype=torch.float), # <- PyTorch loves float32 by default
                                requires_grad=True) # <- can we update this value with gradient descent?))

    # Forward defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" is the input data (e.g. training/testing features)
        return self.weights * x + self.bias # <- this is the linear regression formula (y = m*x + b)


torch.manual_seed(42)

#creating an instance of the model
model_0 = LinearRegressionModel()

#check out the parameters :
list(model_0.parameters())


model_0.state_dict()


with torch.inference_mode():
    y_preds = model_0(X_test)

# %% [code] {"execution":{"iopub.status.busy":"2025-02-05T18:31:54.116522Z","iopub.execute_input":"2025-02-05T18:31:54.116833Z","iopub.status.idle":"2025-02-05T18:31:54.132965Z","shell.execute_reply.started":"2025-02-05T18:31:54.116785Z","shell.execute_reply":"2025-02-05T18:31:54.132136Z"},"jupyter":{"outputs_hidden":false}}
y_preds , y_test

# %% [code] {"execution":{"iopub.status.busy":"2025-02-05T18:31:54.133779Z","iopub.execute_input":"2025-02-05T18:31:54.134164Z","iopub.status.idle":"2025-02-05T18:31:54.520701Z","shell.execute_reply.started":"2025-02-05T18:31:54.134111Z","shell.execute_reply":"2025-02-05T18:31:54.519641Z"},"jupyter":{"outputs_hidden":false}}
plot_predictions(predictions = y_preds)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Train model
# the whole idea of training is for a model to move from some *unknown* parameters (these maybe random) to some *known* parameters. 
# or other words , from poor representation to better representation of the data
# 
# * One way to measure how poor or how wrong my models predictions are is to use a loss function
# Loss function may also be called as Cost function or criterion in different areas.
# 
# 
# Things we need to train
# * **Loss function** : to measure how wrong my model's predictions are to the ideal outputs, lower is better definitely
# * **optimizer** : takes into account the loss of a model and adjusts the model's parameters(e.g. wight & bias in our case) *to improve the loss function*
#   And specifically for pyTorch, we need :
#   * a training loop
#   * a testing loop

# %% [code] {"execution":{"iopub.status.busy":"2025-02-05T18:31:54.521701Z","iopub.execute_input":"2025-02-05T18:31:54.522007Z","iopub.status.idle":"2025-02-05T18:31:54.529112Z","shell.execute_reply.started":"2025-02-05T18:31:54.521980Z","shell.execute_reply":"2025-02-05T18:31:54.528074Z"},"jupyter":{"outputs_hidden":false}}
list(model_0.parameters())

# %% [code] {"execution":{"iopub.status.busy":"2025-02-05T18:31:54.530229Z","iopub.execute_input":"2025-02-05T18:31:54.530545Z","iopub.status.idle":"2025-02-05T18:31:56.685865Z","shell.execute_reply.started":"2025-02-05T18:31:54.530519Z","shell.execute_reply":"2025-02-05T18:31:56.684935Z"},"jupyter":{"outputs_hidden":false}}
# Setup a loss function
loss_fn = nn.L1Loss()
# setup an optimizer
optimizer = torch.optim.SGD(params = model_0.parameters(),
                           lr = 0.001) #learning rate = most important hyperparameter you can set

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # PyTorch training loop intuition
# building a training loop(and testing loop) in pytorch
# 0. loop through data
# 1. forward pass
# 2. calculate the loss
# 3. optimizer zero grad
# 4. loss backward (backpropagation)
# 5. optimizer step
# 
# #### A song for remembering the steps : https://www.youtube.com/watch?v=Nutpusq_AFw

# %% [code] {"execution":{"iopub.status.busy":"2025-02-05T18:31:56.686854Z","iopub.execute_input":"2025-02-05T18:31:56.687332Z","iopub.status.idle":"2025-02-05T18:31:57.755819Z","shell.execute_reply.started":"2025-02-05T18:31:56.687298Z","shell.execute_reply":"2025-02-05T18:31:57.755076Z"},"jupyter":{"outputs_hidden":false}}
# An epoch is one loop through the data (this is hyperparameter because we've set it to ourselve)
epochs = 2000

# step 0 : loop through the data
epoch_count = []
loss_values = []
test_loss_values = []

for epoch in range(epochs) :
    # set the model into training mode
    model_0.train()

    # 1. forward pass
    y_pred = model_0(X_train)
    #2. calculate the loss
    loss = loss_fn(y_pred , y_train)
    #print(loss)
    # 3. optimizer zero grad = gradient value zero kore dey
    optimizer.zero_grad()
    #4. calculates gradient =  perform back propagation on the loss with respect to the parameters of the model
    loss.backward()
    #5.gradient value komay(optimize kore) =  step the optimizer (perform gradient descent)
    optimizer.step()

    ### Now Testing
    
    model_0.eval() #switching to evaluation mode =  turns off gradient tracking
    with torch.inference_mode() : # turns off gradient tracking  ( torch.no_grad() --> works same but inference_mode better )
        test_pred = model_0(X_test)
        test_loss = loss_fn(test_pred , y_test)
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        if epoch % 100 == 0 :
            print(f"Epoch : {epoch} | Loss: {loss} | Test Loss : {test_loss}")
            print(model_0.state_dict())

# %% [code] {"execution":{"iopub.status.busy":"2025-02-05T18:31:57.756778Z","iopub.execute_input":"2025-02-05T18:31:57.757167Z","iopub.status.idle":"2025-02-05T18:31:58.308079Z","shell.execute_reply.started":"2025-02-05T18:31:57.757133Z","shell.execute_reply":"2025-02-05T18:31:58.307023Z"},"jupyter":{"outputs_hidden":false}}
with torch.inference_mode() : # turns off gradient tracking  ( torch.no_grad() --> works same but inference_mode better )
    y_preds_new = model_0(X_test)
    plot_predictions(predictions = y_preds)
    print(loss)
    plot_predictions(predictions = y_preds_new)

# %% [code] {"execution":{"iopub.status.busy":"2025-02-05T18:31:58.309294Z","iopub.execute_input":"2025-02-05T18:31:58.309706Z","iopub.status.idle":"2025-02-05T18:31:58.573326Z","shell.execute_reply.started":"2025-02-05T18:31:58.309664Z","shell.execute_reply":"2025-02-05T18:31:58.572274Z"},"jupyter":{"outputs_hidden":false}}
# Plot the loss curves
import numpy as np
plt.plot(epoch_count , np.array(torch.tensor(loss_values).numpy()) , label = "Train loss")
plt.plot(epoch_count , np.array(torch.tensor(test_loss_values).numpy()) , label = "Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epoch values")
plt.legend()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Saving out PyTorch Model

# %% [code] {"execution":{"iopub.status.busy":"2025-02-05T18:31:58.574094Z","iopub.execute_input":"2025-02-05T18:31:58.574350Z","iopub.status.idle":"2025-02-05T18:31:58.581313Z","shell.execute_reply.started":"2025-02-05T18:31:58.574327Z","shell.execute_reply":"2025-02-05T18:31:58.580287Z"},"jupyter":{"outputs_hidden":false}}
from pathlib import Path

# 1. create model directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents = True , exist_ok = True)

# 2. create model save path
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"model path : {MODEL_SAVE_PATH}")

# 3. saving the model
torch.save(obj=model_0.state_dict() , f=MODEL_SAVE_PATH)

## we only saved model_0.state_dict() , not the whole model

# %% [code] {"execution":{"iopub.status.busy":"2025-02-05T18:31:58.582216Z","iopub.execute_input":"2025-02-05T18:31:58.582441Z","iopub.status.idle":"2025-02-05T18:31:58.600058Z","shell.execute_reply.started":"2025-02-05T18:31:58.582421Z","shell.execute_reply":"2025-02-05T18:31:58.599033Z"},"jupyter":{"outputs_hidden":false}}
import os
print(os.listdir("/kaggle/working/models"))

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Loading model
# #### only state_dict not entire model

# %% [code] {"execution":{"iopub.status.busy":"2025-02-05T18:31:58.600979Z","iopub.execute_input":"2025-02-05T18:31:58.601252Z","iopub.status.idle":"2025-02-05T18:31:58.623350Z","shell.execute_reply.started":"2025-02-05T18:31:58.601227Z","shell.execute_reply":"2025-02-05T18:31:58.622280Z"},"jupyter":{"outputs_hidden":false}}
print(model_0.state_dict())

loaded_model_0 = LinearRegressionModel()
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH ,weights_only=True))
print(loaded_model_0.state_dict())

# %% [code] {"execution":{"iopub.status.busy":"2025-02-05T18:31:58.626683Z","iopub.execute_input":"2025-02-05T18:31:58.626994Z","iopub.status.idle":"2025-02-05T18:31:58.633229Z","shell.execute_reply.started":"2025-02-05T18:31:58.626971Z","shell.execute_reply":"2025-02-05T18:31:58.632374Z"},"jupyter":{"outputs_hidden":false}}
# Testing :
loaded_model_0.eval()
with torch.inference_mode():
    y_pred_load = loaded_model_0(X_test)
    loss_load = loss_fn(y_pred_load , y_preds_new)
    print(loss_load) #0

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 7:38