import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from loguru import logger
import random


from app.my_models.cnn_torch.torchCnnModel import SimpleTorchCNNModel
from app.utils import my_utils as util

class SimpleTorchCNNModelRunner:
    def __init__(self, megData, nchans, ntimes, p_drop_true=0.572):
        """
        Parameters:
        -----------

        p_drop_true : float
            The drop probability for `True` labels when balancing the classes. set to 0 to not apply balancing

        
        
        """
        logger.info("SimpleTorchCNNModelRunner is inited")
        self.check_gpu()
        self.megData = megData      # instance of TorchMegLoader
        self.nchans: int = nchans
        self.ntimes: int = ntimes
        self.p_drop_true: float = p_drop_true     # Drop probability for `True` labels

    
    def check_gpu(self):
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)} is available")
        else:
            logger.error("no GPU, program exit")
            raise Exception



    def train(self, epochs=10, batch_size=1, learning_rate=0.001, train_test_ratio=0.1, test_ratio=None, to_save_res=True):
        

        # for testing
        ratio = (0.2, 0.1, 0.7)
        
        train_size = int(ratio[0] * len(self.megData))
        test_size = int(ratio[1] * len(self.megData))
        not_used = len(self.megData) - train_size - test_size  # remaining data for test

        #train_size = int(train_test_ratio * len(self.megData))
        #test_size = len(self.megData) - train_size
        rand_generator = torch.Generator().manual_seed(33)      # use for fix random seede
        train_dataset, test_dataset, not_used_dataset = random_split(self.megData, 
                                                    lengths=[train_size, test_size, not_used], 
                                                    generator=rand_generator)


        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize the model, loss function, and optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            assert device.type == "cuda"
        except Exception as e:
            logger.error(f"device is: {device}")
            logger.error(f"device.type: {device.type},  is not cuda! program exit!")
            raise AssertionError

        model = SimpleTorchCNNModel(self.nchans, self.ntimes).to(device)
        criterion = nn.BCELoss()    
        # notes:
        # using BCELoss expect X, y both in type float32
        
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        model.train()
                # Counters for `True` and `False` labels used in training
        true_count = 0
        false_count = 0

        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:

                # before .view, dim of inputs: [batch (10), 100(#event), 1, 208 (#chan), 41(#timepoint)]
                # batch likely controlled by batch_size in pytorch Dataloader
                # #event come from how __getitem__ of my torchMegLoader return the epoch (batch_size in torchMegLoader)
                # need to flatten batch and event dim
                inputs = inputs.view(-1, 1, self.nchans, self.ntimes)
                labels = labels.view(-1, 1)


                if self.p_drop_true > 0:

                    filtered_inputs = []
                    filtered_labels = []

                    for i in range(len(labels)):
                        label = labels[i].item()
                        if label == 1.0:
                            # Drop the data with 0.572 probability if the label is `True`
                            if random.random() > self.p_drop_true:
                                filtered_inputs.append(inputs[i])
                                filtered_labels.append(labels[i])
                                true_count += 1
                        else:
                            # Always keep the `False` labels
                            filtered_inputs.append(inputs[i])
                            filtered_labels.append(labels[i])
                            false_count += 1

                    if len(filtered_inputs) == 0:
                        continue  # Skip if all data was dropped in this batch

                    # Convert filtered lists back to tensors
                    inputs = torch.stack(filtered_inputs).to(device)
                    labels = torch.stack(filtered_labels).to(device)


                
                logger.debug(f"inputs: {inputs.shape}")
                logger.debug(f"labels: {labels.shape}")

                # expexted dimension of inputs: [batch_size(#event), 1, nchans, ntimes]

                #optimizer.zero_grad()   # empty gradient
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                #optimizer.step()
                running_loss += loss.item()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}')
            print(f'Truthy samples used: {true_count}, Falsy samples used: {false_count}')

        # Evaluation
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.view(-1, 1, self.nchans, self.ntimes)
                labels = labels.view(-1, 1)
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                y_pred.extend(outputs.cpu().numpy())
                y_true.extend(labels.cpu().numpy())

        y_pred = (np.array(y_pred) >= 0.5).astype(int)
        y_true = np.array(y_true) 
        accuracy = accuracy_score(y_true, y_pred)
        print(f'Accuracy: {accuracy:.2f}')

        # Save predictions and ground truth to DataFrame
        prediction_df = pd.DataFrame({
            'prediction': y_pred.flatten(),
            'ground_truth': y_true.flatten()
        })
        # calculate metrics
        prediction_df = util.add_comparison_column(prediction_df)
        if to_save_res:
            try:
                prediction_df.to_csv(util.get_unique_file_name("voiced_prediction_cnn.csv", "./results"))
            except Exception as err:
                logger.error(err)
                logger.error("fail to output csv, skipping output csv")

            dstr = """epoch: 2\n
                    tmin: -0.1\n
                    tmax: 0.3\n
                    preprocess_low_pass: 35,\n
                    preprocess_high_pass: 180, \n
                    data: sub 0, ses 0 , task 0\n
                    cnn layer: 1--32 (K=1,5)--64(k=208,1)--128(linear)--1\n
                    result: accuracy: 0.69
                    """

            util.get_eval_metrics(prediction_df, 
                              file_name="voiced_metrics_cnn", save_path="./results", 
                              description_str=dstr)


# Example usage
# Assuming you have your data in variables X and y
# runner = SimpleTorchCNNModelRunner(X, y)
# runner.train(epochs=10, batch_size=32, learning_rate=0.001)