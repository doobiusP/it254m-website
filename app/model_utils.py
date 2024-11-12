import json
import os

import torch
from lstm_model import LSTMModel


# def train_evaluate_model(x_train, y_train, x_test, y_test, params, device, num_epochs=100):
#     model = LSTMModel(input_dim=x_train.shape[2], hidden_dim=params['hidden_dim'], 
#                       num_layers=params['num_layers'], output_dim=params['output_dim'], 
#                       dropout=params['dropout']).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
#     criterion = nn.MSELoss()
    
#     best_loss = float('inf')
#     best_model_state = None

#     for _ in range(1, num_epochs + 1):
#         model.train()
#         optimizer.zero_grad()
#         outputs = model(x_train.to(device))
#         loss = criterion(outputs, y_train.to(device))
#         loss.backward()
#         optimizer.step()

#         model.eval()
#         with torch.no_grad():
#             test_loss = criterion(model(x_test.to(device)), y_test.to(device))

#         if test_loss < best_loss:
#             best_loss = test_loss
#             best_model_state = model.state_dict()

#     return best_loss, best_model_state

def load_or_train_model(x_train : torch.Tensor, y_train : torch.Tensor, x_test : torch.Tensor, y_test : torch.Tensor, model_path : str, params_path : str, device : str) -> LSTMModel:
    assert(os.path.exists(model_path) and os.path.exists(params_path) and "Can't find the parameters needed to load into the model!")
    best_params = json.load(open(params_path))
    model = LSTMModel(input_dim=x_train.shape[2], 
                        hidden_dim=best_params['hidden_dim'],
                        num_layers=best_params['num_layers'], 
                        output_dim=best_params['output_dim'],
                        dropout=best_params['dropout']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model
