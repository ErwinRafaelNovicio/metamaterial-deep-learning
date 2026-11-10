import torch
import torch.nn as nn
import joblib
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm, trange

# 1. Definizione del Modello
class MetaBeamMR_MLP(nn.Module):
    def __init__(self, n_input, n_output):
        super(MetaBeamMR_MLP, self).__init__()
        self.inp = nn.Linear(n_input, 8)
        self.hid1 = nn.Linear(8, 8)
        self.hid2 = nn.Linear(8, 8)
        self.out = nn.Linear(8, n_output)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.inp(x))
        x = self.activation(self.hid1(x))
        x = self.activation(self.hid2(x))
        x = self.out(x)
        return x

# 2. Funzione di Training
def train_model(model, train_data, loss_criterion, optimizer, loss_logger,epoch_idx):
    model.train() # Imposta il modello in modalità training
    total_loss = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_data, desc="Training", leave=False)):
        # SPOSTA I DATI SULLA GPU
        
        
        pred_output = model(data)
        loss = loss_criterion(pred_output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_data)
    loss_logger.append(avg_loss)
    return model, optimizer, loss_logger

# 3. Funzione di Testing
def test_model(model, test_data, loss_criterion, loss_logger):
  total_loss = 0
  with torch.no_grad():
    for data,target in test_data:
      pred_output = model(data)
      loss = loss_criterion(pred_output, target)
      #loss_logger.append(loss.item())
      total_loss += loss.item()

    avg_loss = total_loss / len(test_data)
    loss_logger.append(avg_loss)

    return loss_logger

# 4. Valutazione e Grafici
def assess_results(model, X, y, data_type):
    model.eval()
    with torch.no_grad():
        # Sposta l'input sulla GPU per la predizione
        #X = X.to(device)
        # La predizione avviene su GPU, riportiamo il risultato su CPU per matplotlib/numpy
        y_hat = model(X)
        # Assicuriamoci che anche y sia su CPU
        #y = y.cpu()

        # Prediction and plotting (Line)
        plt.figure(figsize=(4, 4))
        plt.plot(y_hat.numpy(), 'ro', label='Predicted output')
        plt.plot(y.numpy(), 'bo', alpha=0.4, label='True output')
        plt.title(f'Predicted vs Estimated output: {data_type} dataset')
        plt.legend()
        plt.show(block=False)
        plt.pause(0.1) # Dà il tempo al sistema operativo di disegnare la finestra

        # Prediction and plotting (Scatter)
        #plt.figure(figsize=(4, 4))
        #plt.scatter(y_hat.numpy(), y.numpy(), alpha=0.5)
        #plt.xlabel('Predicted output')
        #plt.ylabel('True output')
        #plt.title(f'Predicted vs Estimated output: {data_type} dataset')
        #plt.legend()
        #plt.show(block=False)
        #plt.pause(0.1)

        # RMSE calculation
        mse = torch.mean((y_hat - y) ** 2)
        rmse = torch.sqrt(mse)

        # R2 calculation
        total_variance = torch.var(y)
        residual_variance = torch.mean((y_hat - y) ** 2)
        R2 = 1 - residual_variance / total_variance

    return rmse.item(), R2.item()

# 5. Predizione nuovo dato
def predict_stress(model, scaler_x, scaler_y, B, L, Strain):
    model.eval()
    
    raw_input = np.array([[B, L, Strain]])
    scaled_input = scaler_x.transform(raw_input)
    
    # Crea il tensore e spostalo sulla GPU
    input_tensor = torch.FloatTensor(scaled_input)

    with torch.no_grad():
        # Riporta la predizione su CPU prima di passarla a numpy/sklearn
        scaled_prediction = model(input_tensor)

    physical_prediction = scaler_y.inverse_transform(scaled_prediction.numpy())
    return physical_prediction[0][0]