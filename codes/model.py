import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from pytorch_lightning import LightningDataModule, LightningModule
from torch import Tensor

class MHClassifier(nn.Module):
    def __init__(self, peptide_input, protein_input, peptide_hideen_layers, protein_hideen_layers,peptide_protein_hideen_layers, output_size):
        super(MHClassifier, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        ## Peptide encoder 
        layer_modules = []
        bias = False
        input_p = peptide_input
        for layer in peptide_hideen_layers:
            layer_modules.append(
                nn.Sequential(
                    nn.Linear(input_p,out_features = layer,bias=bias),
                    nn.BatchNorm1d(layer),
                    nn.ReLU(),
                    #nn.Dropout(p=drop_rate)
                    )
                )
            input_p = layer

        self.layers_peptide = nn.Sequential(*layer_modules)

        ## Protein encoder 
        layer_modules = []
        bias = False
        input_pro = protein_input
        for layer in protein_hideen_layers:
            layer_modules.append(
                nn.Sequential(
                    nn.Linear(input_pro,out_features = layer,bias=bias),
                    nn.BatchNorm1d(layer),
                    nn.ReLU(),
                    #nn.Dropout(p=drop_rate)
                    )
                )
            input_pro = layer
        
        self.layers_protein = nn.Sequential(*layer_modules)
        
        ## Protein-Peptide Concatenation layer
        layer_modules = []
        bias = False
        self.peptide_protein_concat = nn.Linear(input_p + input_pro, input_p + input_pro)
        
        ## Protein-Peptide combined network 
        layer_modules = []
        bias = False
        input_combined = input_p + input_pro
        for layer in peptide_protein_hideen_layers:
            layer_modules.append(
                nn.Sequential(
                    nn.Linear(input_combined,out_features = layer,bias=bias),
                    nn.BatchNorm1d(layer),
                    nn.ReLU(),
                    #nn.Dropout(p=drop_rate)
                    )
                )
            input_combined = layer
        #layer_modules.append(nn.sigmoid())

        self.Combined_Peptide_protein_layer = nn.Sequential(*layer_modules)

        
    def forward(self, peptide:Tensor, protein:Tensor): 
        pep = self.layers_peptide(peptide)
        pro = self.layers_protein(protein)
        conct_pep_pro = torch.cat((pep, pro), dim=1)
        combined_pep_pro = self.peptide_protein_concat(conct_pep_pro)
        final_output = self.Combined_Peptide_protein_layer(combined_pep_pro)
        return final_output



class DataPlanner(LightningDataModule):
    def __init__(self, X1, X2, y, batch_size):
        super().__init__()
        self.X1 = X1
        self.X2 = X2
        self.y = y
        self.batch_size = batch_size

    def setup(self, stage=None):
        X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(self.X1, self.X2, self.y, test_size=0.2, random_state=42)
        self.train_data = torch.utils.data.TensorDataset(torch.tensor(X1_train), torch.tensor(X2_train), torch.tensor(y_train))
        self.val_data = torch.utils.data.TensorDataset(torch.tensor(X1_val), torch.tensor(X2_val), torch.tensor(y_val))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size)

class LightningModel(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss

    def forward(self, x1, x2):
        return self.model(x1, x2)

    def training_step(self, batch, batch_idx):
        x1, x2, y = batch
        outputs = self(x1, x2)
        loss = self.criterion(outputs, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, y = batch
        outputs = self(x1, x2)
        loss = self.criterion(outputs, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

# Define input sizes
# Classifier = MHClassifier(180,1024,[45,18],[18],[9,1],1)
# peptide_input, protein_input, peptide_hideen_layers, protein_hideen_layers,peptide_protein_hideen_layers, output_size
peptide_input = 160  # Length of peptide sequence 8 * 20 = 160 
protein_input = 1024  # Dimension of categorical variable defining protein sequence
peptide_hideen_layers = [45,18]  # Hidden layer size
protein_hideen_layers = 18  # Hidden layer size
peptide_protein_hideen_layers = [9,1]
output_size = 1  # Output size
batch_size = 100

# Create model instance
model = MHClassifier(peptide_input, protein_input, peptide_hideen_layers, protein_hideen_layers,peptide_protein_hideen_layers, output_size)

# Create Lightning model
lightning_model = LightningModel(model)

# Create DataModule and input the data
data_module = DataPlanner(dataX1, data_X2, data_y, batch_size)

# Create Trainer
trainer = pl.Trainer(max_epochs=50)

# Train the model
trainer.fit(lightning_model, data_module)


