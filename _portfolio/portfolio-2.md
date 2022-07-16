---
title: "NormVAE Model on NeuroImaging data: Paper Implementation"
excerpt: "Implementation of the paper: “NormVAE: Normative modeling on NeuroImaging data using Variational Autoencoders”. I trained the model with our own custom dataset of MCI/AD patient data from ADNI. Generated the deviation maps for studying how much the diseased brain region volumes deviate from that of Healthy Controls..  <br/><img src='/images/normvae.png'>"
collection: portfolio
---

# NormVAE Paper Implementation
Paper link: [Link to the NormVAE paper](https://arxiv.org/pdf/2110.04903.pdf)

GitHub Link : [Link to the GitHub Repo](https://github.com/sandeshkatakam/NormVAE-Neuroimaging)  

# NormVAE Model Implementation

## Data Loading and Preprocessing Steps


```python
import pandas as pd
import numpy as np
df = pd.DataFrame(pd.read_excel("ADNI_sheet_for_VED.xlsx"))


# Data Loading step
healthy_indexes = df.index[df['CDGLOBAL'] == 0].tolist()
healthy_df = df[df.index.isin(healthy_indexes)]
healthy_df
# Preprocessing step
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>COLPROT</th>
      <th>RID</th>
      <th>VISCODE_x</th>
      <th>VISCODE2_x</th>
      <th>EXAMDATE_MRI</th>
      <th>PTDOB</th>
      <th>MRI_AGE</th>
      <th>MRI_AGE_YEARS</th>
      <th>PTGENDER</th>
      <th>PTEDUCAT</th>
      <th>...</th>
      <th>CDGLOBAL</th>
      <th>CDRSB</th>
      <th>CDSOB</th>
      <th>VSWEIGHT</th>
      <th>VSWTUNIT</th>
      <th>VSBPSYS</th>
      <th>VSBPDIA</th>
      <th>APGEN1</th>
      <th>APGEN2</th>
      <th>APOE4_STATUS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ADNIGO</td>
      <td>21</td>
      <td>nv</td>
      <td>m60</td>
      <td>40459</td>
      <td>12056</td>
      <td>77.766667</td>
      <td>77</td>
      <td>2</td>
      <td>18</td>
      <td>...</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>161.0</td>
      <td>1.0</td>
      <td>138.0</td>
      <td>74.0</td>
      <td>2</td>
      <td>3</td>
      <td>others</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ADNI2</td>
      <td>21</td>
      <td>v06</td>
      <td>m72</td>
      <td>40829</td>
      <td>12056</td>
      <td>78.780556</td>
      <td>78</td>
      <td>2</td>
      <td>18</td>
      <td>...</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>172.0</td>
      <td>1.0</td>
      <td>142.0</td>
      <td>70.0</td>
      <td>2</td>
      <td>3</td>
      <td>others</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ADNI2</td>
      <td>21</td>
      <td>v11</td>
      <td>m84</td>
      <td>41186</td>
      <td>12056</td>
      <td>79.755556</td>
      <td>79</td>
      <td>2</td>
      <td>18</td>
      <td>...</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>165.0</td>
      <td>1.0</td>
      <td>106.0</td>
      <td>60.0</td>
      <td>2</td>
      <td>3</td>
      <td>others</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ADNI2</td>
      <td>21</td>
      <td>v21</td>
      <td>m96</td>
      <td>41564</td>
      <td>12056</td>
      <td>80.791667</td>
      <td>80</td>
      <td>2</td>
      <td>18</td>
      <td>...</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>165.0</td>
      <td>1.0</td>
      <td>106.0</td>
      <td>60.0</td>
      <td>2</td>
      <td>3</td>
      <td>others</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ADNI2</td>
      <td>21</td>
      <td>v41</td>
      <td>m120</td>
      <td>42311</td>
      <td>12056</td>
      <td>82.836111</td>
      <td>82</td>
      <td>2</td>
      <td>18</td>
      <td>...</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>158.0</td>
      <td>1.0</td>
      <td>98.0</td>
      <td>58.0</td>
      <td>2</td>
      <td>3</td>
      <td>others</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5087</th>
      <td>ADNI3</td>
      <td>6822</td>
      <td>sc</td>
      <td>sc</td>
      <td>43741</td>
      <td>21187</td>
      <td>61.752778</td>
      <td>61</td>
      <td>2</td>
      <td>14</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>197.6</td>
      <td>1.0</td>
      <td>127.0</td>
      <td>93.0</td>
      <td>3</td>
      <td>3</td>
      <td>others</td>
    </tr>
    <tr>
      <th>5093</th>
      <td>ADNI3</td>
      <td>6831</td>
      <td>sc</td>
      <td>sc</td>
      <td>43776</td>
      <td>15713</td>
      <td>76.833333</td>
      <td>76</td>
      <td>2</td>
      <td>18</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>160.0</td>
      <td>1.0</td>
      <td>127.0</td>
      <td>80.0</td>
      <td>4</td>
      <td>4</td>
      <td>E4</td>
    </tr>
    <tr>
      <th>5097</th>
      <td>ADNI3</td>
      <td>6834</td>
      <td>sc</td>
      <td>sc</td>
      <td>43780</td>
      <td>13522</td>
      <td>82.844444</td>
      <td>82</td>
      <td>2</td>
      <td>16</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>127.0</td>
      <td>1.0</td>
      <td>124.0</td>
      <td>54.0</td>
      <td>3</td>
      <td>3</td>
      <td>others</td>
    </tr>
    <tr>
      <th>5116</th>
      <td>ADNI3</td>
      <td>6853</td>
      <td>sc</td>
      <td>sc</td>
      <td>43865</td>
      <td>19360</td>
      <td>67.091667</td>
      <td>67</td>
      <td>2</td>
      <td>12</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>132.0</td>
      <td>1.0</td>
      <td>161.0</td>
      <td>71.0</td>
      <td>3</td>
      <td>3</td>
      <td>others</td>
    </tr>
    <tr>
      <th>5125</th>
      <td>ADNI3</td>
      <td>6872</td>
      <td>sc</td>
      <td>sc</td>
      <td>44035</td>
      <td>19727</td>
      <td>66.555556</td>
      <td>66</td>
      <td>2</td>
      <td>16</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>165.8</td>
      <td>1.0</td>
      <td>127.0</td>
      <td>87.0</td>
      <td>3</td>
      <td>4</td>
      <td>E4</td>
    </tr>
  </tbody>
</table>
<p>2051 rows × 47 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5131 entries, 0 to 5130
    Data columns (total 47 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   COLPROT                 5131 non-null   object 
     1   RID                     5131 non-null   int64  
     2   VISCODE_x               5131 non-null   object 
     3   VISCODE2_x              5131 non-null   object 
     4   EXAMDATE_MRI            5131 non-null   int64  
     5   PTDOB                   5131 non-null   int64  
     6   MRI_AGE                 5131 non-null   float64
     7   MRI_AGE_YEARS           5131 non-null   int64  
     8   PTGENDER                5131 non-null   int64  
     9   PTEDUCAT                5131 non-null   int64  
     10  CEREBRUM_TCV            5131 non-null   float64
     11  CEREBRUM_TCB            5131 non-null   float64
     12  CEREBRUM_TCC            5131 non-null   float64
     13  CEREBRUM_GRAY           5131 non-null   float64
     14  CEREBRUM_WHITE          5131 non-null   float64
     15  LEFT_HIPPO              5131 non-null   float64
     16  RIGHT_HIPPO             5131 non-null   float64
     17  TOTAL_HIPPO             5131 non-null   float64
     18  TOTAL_CSF               5131 non-null   float64
     19  TOTAL_GRAY              5131 non-null   float64
     20  TOTAL_WHITE             5131 non-null   float64
     21  TOTAL_WMH               5129 non-null   float64
     22  TOTAL_BRAIN             5131 non-null   float64
     23  average_TCV             5131 non-null   float64
     24  Normalised_Left_HIPPO   5131 non-null   float64
     25  Normalised_Right_HIPPO  5131 non-null   float64
     26  Normalised_GM           5131 non-null   float64
     27  Normalised_WM           5131 non-null   float64
     28  Normalised_WMH          5131 non-null   float64
     29  Normalised_CSF          5131 non-null   float64
     30  Normalised_HIPPO        5131 non-null   float64
     31  CDMEMORY                5131 non-null   float64
     32  CDORIENT                5131 non-null   float64
     33  CDJUDGE                 5131 non-null   float64
     34  CDCOMMUN                5131 non-null   float64
     35  CDHOME                  5131 non-null   float64
     36  CDCARE                  5131 non-null   int64  
     37  CDGLOBAL                5131 non-null   float64
     38  CDRSB                   1597 non-null   float64
     39  CDSOB                   1597 non-null   float64
     40  VSWEIGHT                5122 non-null   float64
     41  VSWTUNIT                5089 non-null   float64
     42  VSBPSYS                 5106 non-null   float64
     43  VSBPDIA                 5105 non-null   float64
     44  APGEN1                  5131 non-null   int64  
     45  APGEN2                  5131 non-null   int64  
     46  APOE4_STATUS            5131 non-null   object 
    dtypes: float64(34), int64(9), object(4)
    memory usage: 1.8+ MB
    


```python

healthy_normalized = healthy_df[["Normalised_Left_HIPPO","Normalised_Right_HIPPO", "Normalised_GM", "Normalised_WM", "Normalised_WMH", "Normalised_CSF", "Normalised_HIPPO"]].copy()
healthy_normalized_df = healthy_normalized.reset_index()
healthy_normalized_df = healthy_normalized_df.drop(["index"], axis =1)
healthy_normalized_df.to_excel("original.xlsx", index = False)
```

### Hyperparamaters to be defined: (May be available from the paper to reproduce the exact results)


```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
```




    device(type='cpu')



#### From the paper:
We conditioned both NormVAE and the other baselines on
the age of patients to ensure that the deviations in regional
brain volumes reflect only the disease pathology and not
deviations due to aging effects. All the models were trained
using the Adam optimizer with model hyperparameters as
follows: learning rate = 104

, batch size = 32, latent dimension
= 64, size of dense layer = 512 and number of dense layers
in each of encoder and decoder = 3.

## Splitting data and Preprocessing for Training


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
```


```python

def load_and_standardize_data(path):
    # read in from csv
    df= healthy_normalized_df
    #df = pd.read_excel("ADNI_sheet_for_VED.xlsx")
    #df = pd.read_csv(path, sep=',')
    # replace nan with -99
    df = df.fillna(-99)
    df = df.values.reshape(-1, df.shape[1]).astype('float32')
    # randomly split
    X_train, X_test = train_test_split(df, test_size=0.3, random_state=42)
    # standardize values
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)   
    return X_train, X_test, scaler
```


```python
X_train, X_test, scaler = load_and_standardize_data("ADNI_sheet_for_VED.xlsx")
```


```python
X_train.shape, X_test.shape
```




    ((1435, 7), (616, 7))




```python
from torch.utils.data import Dataset, DataLoader
DATA_PATH = "ADNI_sheet_for_VED.xlsx"
class DataBuilder(Dataset):
    def __init__(self, path, train=True):
        self.X_train, self.X_test, self.standardizer = load_and_standardize_data(DATA_PATH)
        if train:
            self.x = torch.from_numpy(self.X_train)
            self.len=self.x.shape[0]
        else:
            self.x = torch.from_numpy(self.X_test)
            self.len=self.x.shape[0]
        del self.X_train
        del self.X_test 
    def __getitem__(self,index):      
        return self.x[index]
    def __len__(self):
        return self.len
```


```python
traindata_set=DataBuilder(DATA_PATH, train=True)
testdata_set=DataBuilder(DATA_PATH, train=False)

trainloader=DataLoader(dataset=traindata_set,batch_size=1024)
testloader=DataLoader(dataset=testdata_set,batch_size=1024)
```

### Visualization and checking the dimensions of the Input data


```python
trainloader.dataset.x.shape, testloader.dataset.x.shape
```




    (torch.Size([1435, 7]), torch.Size([616, 7]))




```python
 trainloader.dataset.x.shape[1]
```




    7



##  Model Architecture Implementation(Building the Model)


```python
class Autoencoder(nn.Module):
    def __init__(self,D_in=7,H=50,H2=12,latent_dim=64):
        
        #Encoder
        super(Autoencoder,self).__init__()
        self.linear1=nn.Linear(D_in,H)
        self.lin_bn1 = nn.BatchNorm1d(num_features=H)
        self.linear2=nn.Linear(H,H2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=H2)
        self.linear3=nn.Linear(H2,H2)
        self.lin_bn3 = nn.BatchNorm1d(num_features=H2)
        
        # Latent vectors mu and sigma
        self.fc1 = nn.Linear(H2, latent_dim)
        self.bn1 = nn.BatchNorm1d(num_features=latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)

        # Sampling vector
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_bn3 = nn.BatchNorm1d(latent_dim)
        self.fc4 = nn.Linear(latent_dim, H2)
        self.fc_bn4 = nn.BatchNorm1d(H2)
        
        # Decoder
        self.linear4=nn.Linear(H2,H2)
        self.lin_bn4 = nn.BatchNorm1d(num_features=H2)
        self.linear5=nn.Linear(H2,H)
        self.lin_bn5 = nn.BatchNorm1d(num_features=H)
        self.linear6=nn.Linear(H,D_in)
        self.lin_bn6 = nn.BatchNorm1d(num_features=D_in)
        
        self.relu = nn.ReLU()
        
    def encode(self, x):
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

        fc1 = F.relu(self.bn1(self.fc1(lin3)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)
        
        return r1, r2
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))

        lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))
        lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))
        return self.lin_bn6(self.linear6(lin5))


        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
```

### Loss Function (combination of MSE and KL Divergence Loss)


```python
class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
    
    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD
```


```python
loss_mse = customLoss()
```


```python
D_in = 7
H = 50
H2 = 12
model = Autoencoder(D_in, H, H2).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

## Training the Model 


```python
epochs = 1500
log_interval = 50
val_losses = []
train_losses = []
test_losses = []
```


```python
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(trainloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_mse(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    if epoch % 200 == 0:        
        print('====> Epoch: {} Average training loss: {:.4f}'.format(
            epoch, train_loss / len(trainloader.dataset)))
        train_losses.append(train_loss / len(trainloader.dataset))
```


```python
def test(epoch):
    with torch.no_grad():
        test_loss = 0
        for batch_idx, data in enumerate(testloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_mse(recon_batch, data, mu, logvar)
            test_loss += loss.item()
            if epoch % 200 == 0:        
                print('====> Epoch: {} Average test loss: {:.4f}'.format(
                    epoch, test_loss / len(testloader.dataset)))
            test_losses.append(test_loss / len(testloader.dataset))
```


```python
for epoch in range(1, epochs + 1):
    train(epoch)
    test(epoch)
```

    ====> Epoch: 200 Average training loss: 5.2553
    ====> Epoch: 200 Average test loss: 5.5454
    ====> Epoch: 400 Average training loss: 4.5688
    ====> Epoch: 400 Average test loss: 4.8149
    ====> Epoch: 600 Average training loss: 4.2005
    ====> Epoch: 600 Average test loss: 4.5410
    ====> Epoch: 800 Average training loss: 4.1647
    ====> Epoch: 800 Average test loss: 4.4832
    ====> Epoch: 1000 Average training loss: 4.0977
    ====> Epoch: 1000 Average test loss: 4.4351
    ====> Epoch: 1200 Average training loss: 4.0685
    ====> Epoch: 1200 Average test loss: 4.4041
    ====> Epoch: 1400 Average training loss: 4.0465
    ====> Epoch: 1400 Average test loss: 4.3791
    

## Predictions from the Model(Reconstruction of sample from the Latent Space)


```python
with torch.no_grad():
    for batch_idx, data in enumerate(testloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
```


```python
mu.shape, logvar.shape
```




    (torch.Size([616, 64]), torch.Size([616, 64]))




```python
sigma = torch.exp(logvar/2)
```


```python
recon_batch.shape
```




    torch.Size([616, 7])




```python
mu.shape, sigma.shape
```




    (torch.Size([616, 64]), torch.Size([616, 64]))




```python
mu.mean(axis=0), sigma.mean(axis=0)
```




    (tensor([ 6.7321e-05, -8.9583e-04, -1.5867e-03,  3.4130e-04, -4.7437e-03,
             -2.6509e-05,  9.6395e-04,  1.7330e-03,  1.1193e-05, -1.8322e-03,
             -3.9045e-03, -2.5547e-03, -1.0273e-03,  5.5807e-04,  6.1781e-05,
             -5.7793e-04,  1.6116e-04, -1.1905e-03,  2.0707e-04,  2.3820e-04,
             -1.7027e-03, -4.4479e-04,  8.1619e-04, -8.6566e-04,  1.0045e-03,
              8.1418e-05, -6.2074e-04,  7.7874e-04,  3.9366e-04, -2.7248e-03,
             -5.9982e-04,  3.8269e-04,  5.0697e-04, -6.4379e-04, -5.5114e-04,
             -1.0326e-03,  1.1389e-03, -9.0580e-04,  1.2345e-03,  6.0152e-04,
              2.7089e-04,  1.2007e-03,  4.2639e-04, -1.6241e-04, -1.9819e-03,
              6.1678e-04, -1.7487e-03,  1.0522e-03, -7.2773e-04,  3.1068e-04,
             -6.3635e-04,  1.6886e-03, -1.3969e-03,  1.9532e-03, -5.3024e-04,
              1.5317e-03,  9.8963e-04, -3.0573e-05, -2.6588e-04,  1.0513e-03,
             -2.9128e-04, -1.0806e-03,  6.9774e-05,  1.3817e-03]),
     tensor([0.9997, 0.9997, 0.9998, 0.9989, 0.5908, 0.9999, 0.9990, 0.9990, 1.0011,
             1.0004, 0.3951, 1.0006, 0.9997, 0.9992, 0.9988, 0.9986, 0.9994, 0.9975,
             1.0011, 0.9994, 1.0006, 0.9989, 0.9975, 1.0003, 1.0010, 0.9997, 0.9983,
             1.0011, 0.9993, 1.0002, 0.9988, 1.0008, 0.9997, 0.9998, 0.9994, 0.9995,
             0.9988, 0.9997, 1.0004, 1.0008, 1.0009, 0.9987, 0.9994, 0.9991, 0.9989,
             1.0010, 0.6045, 1.0005, 0.9993, 0.9993, 1.0002, 0.9996, 0.9998, 0.9986,
             1.0002, 0.9996, 0.9987, 0.9993, 1.0008, 0.9995, 1.0008, 1.0014, 0.9989,
             0.9996]))




```python
# sample z from q
no_samples = 20
q = torch.distributions.Normal(mu.mean(axis=0), sigma.mean(axis=0))
z = q.rsample(sample_shape=torch.Size([no_samples]))
```


```python
z.shape
```




    torch.Size([20, 64])




```python
z[:5]
```




    tensor([[-2.6207e-01,  1.8191e+00, -2.1019e-01, -2.3689e+00, -6.4797e-01,
             -3.4001e-01,  5.2944e-01, -1.6836e+00,  1.0925e+00,  1.1712e+00,
             -5.1861e-01, -1.9329e+00,  5.3887e-01,  1.7292e+00,  2.3229e+00,
             -4.4991e-01, -1.4317e+00,  1.3813e+00, -4.4005e-01,  1.3080e+00,
              1.0775e-01, -1.0039e+00, -6.9395e-01, -1.8043e+00,  3.4235e-02,
              4.7414e-01, -1.0163e-01,  2.1672e-01,  6.8726e-01, -8.7158e-01,
              1.3425e-01,  3.6847e-02, -5.5306e-01, -6.7322e-01, -4.4646e-01,
              1.3380e+00,  1.2718e+00, -1.5523e+00, -9.1517e-01,  4.4068e-01,
              7.0552e-01, -2.0342e-01,  2.7887e-02, -2.8240e-01, -1.5634e+00,
              3.0968e-01,  2.6210e-01,  5.3343e-01,  1.1519e+00,  7.8934e-01,
              4.8992e-01, -1.0377e+00, -2.0961e-01,  4.4531e-01,  4.2487e-02,
              2.7978e-01,  4.2848e-01,  2.4535e-01,  3.2812e-01,  6.9987e-01,
             -1.3124e-01,  1.7767e+00,  2.5858e-01, -1.2272e+00],
            [ 1.0209e+00, -1.4452e+00,  1.2785e+00, -6.2328e-01, -6.8169e-01,
             -1.7512e+00,  2.8541e-01, -2.7175e-02, -6.3855e-01,  4.3762e-02,
             -7.8638e-01,  1.5444e+00, -8.1110e-01,  2.1806e-01, -7.9236e-01,
             -6.9815e-03,  2.1864e-01,  1.1985e+00,  4.5730e-01,  4.6778e-01,
              1.2955e+00, -1.4020e+00, -1.4028e+00,  1.2996e+00, -2.4095e+00,
             -7.5843e-01, -1.0689e-01, -6.0003e-01,  3.8757e-01, -9.5297e-01,
             -7.1976e-03, -1.1149e+00, -1.0675e+00, -1.9261e+00, -6.1760e-02,
             -1.5507e+00, -4.9858e-01,  9.0438e-02, -8.2458e-01, -1.1393e+00,
             -1.1230e-01,  8.5906e-01, -1.1825e-01,  6.5475e-01, -1.5792e-01,
             -3.0652e-01, -2.8458e-01,  5.1345e-01, -2.7567e-01,  1.9618e+00,
             -1.6396e-01,  8.3016e-01,  8.4419e-01,  9.3068e-01, -7.7973e-01,
              1.1696e+00,  1.9484e-01, -1.8576e+00, -1.5479e+00, -2.8345e-01,
              4.8906e-01, -1.3836e+00,  9.0326e-01, -1.9636e-01],
            [ 7.0022e-01,  8.4105e-01,  1.2776e+00, -5.7347e-01,  4.5093e-01,
              2.8287e-02, -3.5367e-01,  1.4242e+00,  8.3518e-01,  1.9604e-01,
             -1.1902e+00, -7.8059e-01,  8.9422e-01,  1.3408e+00, -1.4054e+00,
              6.5918e-01, -4.2354e-01,  8.5976e-01,  7.7381e-01, -5.6624e-02,
              2.5318e-01,  9.4953e-01,  2.2229e+00, -1.5446e+00,  3.1552e+00,
              1.0755e+00,  2.5981e-01,  3.1997e-01, -3.7525e-01,  1.4404e+00,
              3.4912e-01, -2.5857e-01,  3.8624e-01, -1.9970e-01, -6.8511e-01,
             -9.5036e-01,  6.5627e-01,  6.4647e-01,  2.1521e-01,  7.3283e-01,
              6.5568e-01,  6.5587e-01, -1.5103e+00,  5.4824e-01, -8.8896e-01,
              1.0728e+00,  5.7586e-02,  3.8735e-01,  3.1898e-01,  4.5736e-01,
             -6.9813e-02,  1.2815e+00, -4.6582e-01,  8.1180e-01, -1.0776e-01,
              3.6080e-02, -7.4211e-01, -2.0570e+00,  2.5289e-01, -1.0930e+00,
              9.5386e-01, -1.1871e+00,  1.2682e+00, -1.7724e-04],
            [ 2.2209e+00, -3.6109e-01, -8.3713e-01, -6.8234e-01, -6.9216e-01,
             -9.4394e-01, -1.1351e+00,  1.3705e-02,  1.5266e-01, -6.5989e-02,
              8.7300e-01,  8.2864e-01, -1.6141e+00,  2.1174e+00,  1.1286e+00,
              1.0643e+00, -1.1649e+00, -2.2526e+00,  5.2665e-01, -6.5348e-01,
              4.3073e-01, -9.9332e-01, -8.4257e-01, -1.9335e-01,  7.0475e-01,
              7.0156e-02, -4.0648e-01,  6.2741e-01, -3.5083e-02,  2.9190e-02,
             -1.0805e+00, -1.9750e-02,  9.7785e-02,  7.7487e-01, -9.5052e-01,
             -1.0999e+00,  5.4854e-01, -1.6213e-01,  3.8821e-01, -1.7526e+00,
             -8.2344e-01,  6.2815e-01, -2.9670e-02,  8.1209e-01,  5.5605e-02,
              1.3945e+00, -8.9312e-01, -9.7363e-01, -4.2161e-01, -1.1026e+00,
             -1.8687e-01, -3.3740e-01, -8.6791e-01,  3.9583e-01, -2.0613e+00,
             -1.0527e-01, -9.1413e-01, -7.0248e-01,  3.3742e-01,  2.4582e+00,
              2.8557e+00, -4.3148e-01, -1.3120e+00, -1.5450e+00],
            [-1.5549e+00, -1.1406e+00,  5.1126e-01, -1.8057e+00, -9.5861e-01,
             -1.3170e-01,  9.2797e-02, -1.7434e-03,  6.4342e-01, -1.0257e+00,
              3.2177e-01, -3.3760e-01,  2.5204e-01, -4.6751e-01, -5.3467e-01,
             -6.7907e-01, -1.4553e+00,  1.4292e+00,  5.6030e-02,  1.7337e-01,
             -3.0357e-01,  6.2371e-01, -1.1419e+00,  1.2490e+00,  5.0464e-01,
              9.8252e-01,  1.4045e+00, -1.8143e+00,  6.5289e-01, -5.0173e-01,
              7.8742e-01,  1.5790e-01, -2.2307e-01,  5.7874e-01,  9.9586e-01,
              1.4665e+00, -7.4934e-02, -7.0671e-01, -8.6875e-01, -9.8476e-01,
             -9.5957e-01,  2.7299e-01,  1.4057e-01,  7.0796e-01,  9.7144e-02,
              2.0742e-01, -7.6660e-01, -9.0187e-01,  1.2054e+00, -4.5695e-01,
             -1.8195e-01,  5.5942e-02,  1.4575e+00, -2.0068e+00, -4.0691e-01,
             -2.2452e-01,  4.4170e-01, -1.1923e+00,  1.5058e-01,  2.0991e+00,
             -5.3677e-01, -2.3617e-01,  5.1852e-01, -4.8177e-01]])




```python
with torch.no_grad():
    pred = model.decode(z).cpu().numpy()
```


```python
pred[1]
```




    array([ 1.5000362 ,  1.4009571 ,  1.0334438 ,  0.14170583, -0.21413791,
           -0.743145  ,  1.5566754 ], dtype=float32)




```python
fake_data = scaler.inverse_transform(pred)
pd.DataFrame(fake_data)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.658290</td>
      <td>3.714365</td>
      <td>615.972717</td>
      <td>444.102570</td>
      <td>3.468233</td>
      <td>362.094727</td>
      <td>7.372141</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.775800</td>
      <td>3.828723</td>
      <td>638.225952</td>
      <td>463.833893</td>
      <td>3.376383</td>
      <td>319.129486</td>
      <td>7.625655</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.780164</td>
      <td>3.897336</td>
      <td>599.615417</td>
      <td>511.592346</td>
      <td>0.830701</td>
      <td>303.205597</td>
      <td>7.671512</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.678336</td>
      <td>2.682931</td>
      <td>623.772583</td>
      <td>430.854340</td>
      <td>2.470432</td>
      <td>366.023712</td>
      <td>5.357521</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.123950</td>
      <td>3.102459</td>
      <td>642.759888</td>
      <td>426.775299</td>
      <td>2.900143</td>
      <td>353.580048</td>
      <td>6.243179</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.643203</td>
      <td>3.722982</td>
      <td>591.546021</td>
      <td>470.808807</td>
      <td>6.740404</td>
      <td>354.595917</td>
      <td>7.343347</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2.979379</td>
      <td>3.004214</td>
      <td>633.870117</td>
      <td>423.768921</td>
      <td>1.637007</td>
      <td>367.192413</td>
      <td>5.974944</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3.293342</td>
      <td>3.377847</td>
      <td>588.772400</td>
      <td>480.983154</td>
      <td>5.393816</td>
      <td>343.471497</td>
      <td>6.673167</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3.111947</td>
      <td>3.213244</td>
      <td>583.475403</td>
      <td>498.851959</td>
      <td>5.819061</td>
      <td>328.099152</td>
      <td>6.311371</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3.390304</td>
      <td>3.434218</td>
      <td>606.538513</td>
      <td>403.388885</td>
      <td>33.454300</td>
      <td>399.425568</td>
      <td>6.829868</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3.447525</td>
      <td>3.497851</td>
      <td>623.776978</td>
      <td>444.948883</td>
      <td>1.317011</td>
      <td>352.726898</td>
      <td>6.947545</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3.464022</td>
      <td>3.565243</td>
      <td>643.925476</td>
      <td>487.048401</td>
      <td>0.920130</td>
      <td>284.147003</td>
      <td>7.018982</td>
    </tr>
    <tr>
      <th>12</th>
      <td>3.140067</td>
      <td>3.166206</td>
      <td>637.777161</td>
      <td>464.003479</td>
      <td>-0.024734</td>
      <td>316.960236</td>
      <td>6.297874</td>
    </tr>
    <tr>
      <th>13</th>
      <td>3.211455</td>
      <td>3.262667</td>
      <td>589.543396</td>
      <td>463.550934</td>
      <td>5.402682</td>
      <td>363.792908</td>
      <td>6.488429</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2.950759</td>
      <td>2.962238</td>
      <td>614.745300</td>
      <td>422.735748</td>
      <td>3.211672</td>
      <td>388.157715</td>
      <td>5.908453</td>
    </tr>
    <tr>
      <th>15</th>
      <td>3.323076</td>
      <td>3.442024</td>
      <td>592.784973</td>
      <td>501.746002</td>
      <td>3.007922</td>
      <td>315.990295</td>
      <td>6.756650</td>
    </tr>
    <tr>
      <th>16</th>
      <td>3.218014</td>
      <td>3.270338</td>
      <td>613.629028</td>
      <td>431.350311</td>
      <td>2.906749</td>
      <td>378.878387</td>
      <td>6.482038</td>
    </tr>
    <tr>
      <th>17</th>
      <td>3.297865</td>
      <td>3.390033</td>
      <td>591.776428</td>
      <td>487.965302</td>
      <td>3.807047</td>
      <td>334.130951</td>
      <td>6.678604</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2.913397</td>
      <td>3.060686</td>
      <td>592.994629</td>
      <td>491.221680</td>
      <td>2.869703</td>
      <td>327.627838</td>
      <td>5.994271</td>
    </tr>
    <tr>
      <th>19</th>
      <td>3.045898</td>
      <td>3.087197</td>
      <td>590.834900</td>
      <td>420.717682</td>
      <td>13.150869</td>
      <td>407.105469</td>
      <td>6.154039</td>
    </tr>
  </tbody>
</table>
</div>



#### Note: Input the value of "n" for the number of reconstructed samples we need


```python
n = 10 # Input the value of n for the number of samples that need to be generated
recon_batch[0:n].cpu().numpy().shape # This generates a batch of n+1 reconstructed samples

```




    (10, 7)




```python
scaler = trainloader.dataset.standardizer
recon_row = scaler.inverse_transform(recon_batch[0:n].cpu().numpy())
real_row = scaler.inverse_transform(testloader.dataset.x[0:n].cpu().numpy())
```


```python
recon_row.shape, real_row.shape
```




    ((10, 7), (10, 7))




```python
np.stack((recon_row, real_row), axis = 1).reshape((2*n,7)).shape
```




    (20, 7)



### Two samples are reconstructed using Decoder of the VAE (Compare with the original data samples)


```python
recon_df = pd.DataFrame(np.stack((recon_row, real_row), axis = 1).reshape((2*n,7)))
recon_df.columns = ["Normalised_Left_HIPPO","Normalised_Right_HIPPO", "Normalised_GM","Normalised_WM","Normalised_WMH","Normalised_CSF", "Normalised_HIPPO"]
```


```python
only_recon_df = pd.DataFrame(recon_row)
only_recon_df.columns = ["Normalised_Left_HIPPO","Normalised_Right_HIPPO", "Normalised_GM","Normalised_WM","Normalised_WMH","Normalised_CSF", "Normalised_HIPPO"]
```


```python
only_recon_df # Only reconstructed samples dataframe
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Normalised_Left_HIPPO</th>
      <th>Normalised_Right_HIPPO</th>
      <th>Normalised_GM</th>
      <th>Normalised_WM</th>
      <th>Normalised_WMH</th>
      <th>Normalised_CSF</th>
      <th>Normalised_HIPPO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.295215</td>
      <td>3.374730</td>
      <td>605.966309</td>
      <td>479.775330</td>
      <td>3.352345</td>
      <td>328.901825</td>
      <td>6.676765</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.249777</td>
      <td>3.323651</td>
      <td>602.573669</td>
      <td>460.694885</td>
      <td>4.064610</td>
      <td>355.638641</td>
      <td>6.559394</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.115930</td>
      <td>3.169504</td>
      <td>587.403625</td>
      <td>465.757233</td>
      <td>4.888006</td>
      <td>359.106903</td>
      <td>6.300127</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.341771</td>
      <td>3.433910</td>
      <td>631.655396</td>
      <td>487.568970</td>
      <td>2.618053</td>
      <td>296.845825</td>
      <td>6.779849</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.521149</td>
      <td>3.564056</td>
      <td>612.857849</td>
      <td>402.475281</td>
      <td>50.237633</td>
      <td>363.139435</td>
      <td>7.077423</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.369879</td>
      <td>3.441562</td>
      <td>592.080383</td>
      <td>442.557404</td>
      <td>17.712931</td>
      <td>368.885773</td>
      <td>6.806715</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3.103010</td>
      <td>3.066947</td>
      <td>656.844727</td>
      <td>422.060364</td>
      <td>2.857587</td>
      <td>348.377838</td>
      <td>6.171407</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3.032931</td>
      <td>3.098542</td>
      <td>609.733887</td>
      <td>470.137146</td>
      <td>2.733951</td>
      <td>340.055176</td>
      <td>6.127314</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3.047361</td>
      <td>3.116353</td>
      <td>609.942322</td>
      <td>470.380341</td>
      <td>2.563208</td>
      <td>339.141693</td>
      <td>6.157194</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3.741351</td>
      <td>3.842099</td>
      <td>615.631042</td>
      <td>486.632843</td>
      <td>2.634791</td>
      <td>319.231812</td>
      <td>7.576368</td>
    </tr>
  </tbody>
</table>
</div>



### Note: Recon_df contains the data of n reconstructed and n data samples from the original dataset.


```python
recon_df # The first n samples are reconstructed using VAE and the next n samples are from the original dataset
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Normalised_Left_HIPPO</th>
      <th>Normalised_Right_HIPPO</th>
      <th>Normalised_GM</th>
      <th>Normalised_WM</th>
      <th>Normalised_WMH</th>
      <th>Normalised_CSF</th>
      <th>Normalised_HIPPO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.295215</td>
      <td>3.374730</td>
      <td>605.966309</td>
      <td>479.775330</td>
      <td>3.352345</td>
      <td>328.901825</td>
      <td>6.676765</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.010453</td>
      <td>3.179425</td>
      <td>588.671082</td>
      <td>481.663666</td>
      <td>10.713317</td>
      <td>320.548950</td>
      <td>6.189878</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.249777</td>
      <td>3.323651</td>
      <td>602.573669</td>
      <td>460.694885</td>
      <td>4.064610</td>
      <td>355.638641</td>
      <td>6.559394</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.238097</td>
      <td>3.143342</td>
      <td>631.221619</td>
      <td>463.954651</td>
      <td>2.640509</td>
      <td>311.572662</td>
      <td>6.381440</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.115930</td>
      <td>3.169504</td>
      <td>587.403625</td>
      <td>465.757233</td>
      <td>4.888006</td>
      <td>359.106903</td>
      <td>6.300127</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.172531</td>
      <td>3.401141</td>
      <td>603.727905</td>
      <td>460.706482</td>
      <td>0.186141</td>
      <td>350.657318</td>
      <td>6.573673</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3.341771</td>
      <td>3.433910</td>
      <td>631.655396</td>
      <td>487.568970</td>
      <td>2.618053</td>
      <td>296.845825</td>
      <td>6.779849</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3.219062</td>
      <td>3.226120</td>
      <td>662.387878</td>
      <td>490.052582</td>
      <td>2.364141</td>
      <td>279.084534</td>
      <td>6.445182</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3.521149</td>
      <td>3.564056</td>
      <td>612.857849</td>
      <td>402.475281</td>
      <td>50.237633</td>
      <td>363.139435</td>
      <td>7.077423</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3.581865</td>
      <td>3.890969</td>
      <td>611.389587</td>
      <td>386.356110</td>
      <td>69.504646</td>
      <td>384.306793</td>
      <td>7.472834</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3.369879</td>
      <td>3.441562</td>
      <td>592.080383</td>
      <td>442.557404</td>
      <td>17.712931</td>
      <td>368.885773</td>
      <td>6.806715</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2.954243</td>
      <td>3.377693</td>
      <td>591.144958</td>
      <td>417.393311</td>
      <td>8.789750</td>
      <td>431.077972</td>
      <td>6.331936</td>
    </tr>
    <tr>
      <th>12</th>
      <td>3.103010</td>
      <td>3.066947</td>
      <td>656.844727</td>
      <td>422.060364</td>
      <td>2.857587</td>
      <td>348.377838</td>
      <td>6.171407</td>
    </tr>
    <tr>
      <th>13</th>
      <td>3.644325</td>
      <td>3.278724</td>
      <td>650.829712</td>
      <td>383.037292</td>
      <td>1.304775</td>
      <td>419.726807</td>
      <td>6.923048</td>
    </tr>
    <tr>
      <th>14</th>
      <td>3.032931</td>
      <td>3.098542</td>
      <td>609.733887</td>
      <td>470.137146</td>
      <td>2.733951</td>
      <td>340.055176</td>
      <td>6.127314</td>
    </tr>
    <tr>
      <th>15</th>
      <td>3.128363</td>
      <td>3.069726</td>
      <td>586.275757</td>
      <td>487.723999</td>
      <td>0.829597</td>
      <td>339.987305</td>
      <td>6.198089</td>
    </tr>
    <tr>
      <th>16</th>
      <td>3.047361</td>
      <td>3.116353</td>
      <td>609.942322</td>
      <td>470.380341</td>
      <td>2.563208</td>
      <td>339.141693</td>
      <td>6.157194</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2.948063</td>
      <td>3.108797</td>
      <td>620.992676</td>
      <td>478.187500</td>
      <td>0.705342</td>
      <td>317.269409</td>
      <td>6.056860</td>
    </tr>
    <tr>
      <th>18</th>
      <td>3.741351</td>
      <td>3.842099</td>
      <td>615.631042</td>
      <td>486.632843</td>
      <td>2.634791</td>
      <td>319.231812</td>
      <td>7.576368</td>
    </tr>
    <tr>
      <th>19</th>
      <td>3.818749</td>
      <td>4.140560</td>
      <td>584.180420</td>
      <td>475.261566</td>
      <td>4.848103</td>
      <td>357.739075</td>
      <td>7.959309</td>
    </tr>
  </tbody>
</table>
</div>




```python
healthy_normalized_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Normalised_Left_HIPPO</th>
      <th>Normalised_Right_HIPPO</th>
      <th>Normalised_GM</th>
      <th>Normalised_WM</th>
      <th>Normalised_WMH</th>
      <th>Normalised_CSF</th>
      <th>Normalised_HIPPO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.906594</td>
      <td>3.937245</td>
      <td>600.810654</td>
      <td>495.498178</td>
      <td>1.816761</td>
      <td>312.457857</td>
      <td>7.843839</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.692551</td>
      <td>3.702927</td>
      <td>604.911773</td>
      <td>468.931468</td>
      <td>1.224921</td>
      <td>349.979947</td>
      <td>7.395478</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.572854</td>
      <td>3.570298</td>
      <td>593.160610</td>
      <td>447.228614</td>
      <td>1.505301</td>
      <td>368.909511</td>
      <td>7.143152</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.539053</td>
      <td>3.640690</td>
      <td>603.567502</td>
      <td>451.664686</td>
      <td>1.317372</td>
      <td>373.248906</td>
      <td>7.179743</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.481908</td>
      <td>3.402540</td>
      <td>611.570743</td>
      <td>420.812093</td>
      <td>1.370832</td>
      <td>401.630613</td>
      <td>6.884448</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2046</th>
      <td>3.727488</td>
      <td>3.673359</td>
      <td>643.091260</td>
      <td>464.149711</td>
      <td>0.050438</td>
      <td>284.434370</td>
      <td>7.400847</td>
    </tr>
    <tr>
      <th>2047</th>
      <td>3.583230</td>
      <td>3.501180</td>
      <td>653.800079</td>
      <td>496.135716</td>
      <td>3.482072</td>
      <td>288.688287</td>
      <td>7.084410</td>
    </tr>
    <tr>
      <th>2048</th>
      <td>3.334138</td>
      <td>3.519456</td>
      <td>672.356246</td>
      <td>412.315592</td>
      <td>2.244508</td>
      <td>371.936767</td>
      <td>6.853594</td>
    </tr>
    <tr>
      <th>2049</th>
      <td>3.092036</td>
      <td>3.163253</td>
      <td>677.594203</td>
      <td>458.076967</td>
      <td>4.667049</td>
      <td>294.305059</td>
      <td>6.255301</td>
    </tr>
    <tr>
      <th>2050</th>
      <td>3.513670</td>
      <td>3.250810</td>
      <td>672.835632</td>
      <td>464.558631</td>
      <td>0.969365</td>
      <td>319.271484</td>
      <td>6.764480</td>
    </tr>
  </tbody>
</table>
<p>2051 rows × 7 columns</p>
</div>



#### We can Inspect the values if they are similar to the original data distribution.

### Calculating mean and variance 

### Data Loading step for diseased samples


```python
df = pd.DataFrame(pd.read_excel("ADNI_sheet_for_VED.xlsx"))
# Data Loading step
disease_indexes = df.index[df['CDGLOBAL'] == 1].tolist()
disease_df = df[df.index.isin(healthy_indexes)]
disease_df
# Preprocessing step
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>COLPROT</th>
      <th>RID</th>
      <th>VISCODE_x</th>
      <th>VISCODE2_x</th>
      <th>EXAMDATE_MRI</th>
      <th>PTDOB</th>
      <th>MRI_AGE</th>
      <th>MRI_AGE_YEARS</th>
      <th>PTGENDER</th>
      <th>PTEDUCAT</th>
      <th>...</th>
      <th>CDGLOBAL</th>
      <th>CDRSB</th>
      <th>CDSOB</th>
      <th>VSWEIGHT</th>
      <th>VSWTUNIT</th>
      <th>VSBPSYS</th>
      <th>VSBPDIA</th>
      <th>APGEN1</th>
      <th>APGEN2</th>
      <th>APOE4_STATUS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ADNIGO</td>
      <td>21</td>
      <td>nv</td>
      <td>m60</td>
      <td>40459</td>
      <td>12056</td>
      <td>77.766667</td>
      <td>77</td>
      <td>2</td>
      <td>18</td>
      <td>...</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>161.0</td>
      <td>1.0</td>
      <td>138.0</td>
      <td>74.0</td>
      <td>2</td>
      <td>3</td>
      <td>others</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ADNI2</td>
      <td>21</td>
      <td>v06</td>
      <td>m72</td>
      <td>40829</td>
      <td>12056</td>
      <td>78.780556</td>
      <td>78</td>
      <td>2</td>
      <td>18</td>
      <td>...</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>172.0</td>
      <td>1.0</td>
      <td>142.0</td>
      <td>70.0</td>
      <td>2</td>
      <td>3</td>
      <td>others</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ADNI2</td>
      <td>21</td>
      <td>v11</td>
      <td>m84</td>
      <td>41186</td>
      <td>12056</td>
      <td>79.755556</td>
      <td>79</td>
      <td>2</td>
      <td>18</td>
      <td>...</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>165.0</td>
      <td>1.0</td>
      <td>106.0</td>
      <td>60.0</td>
      <td>2</td>
      <td>3</td>
      <td>others</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ADNI2</td>
      <td>21</td>
      <td>v21</td>
      <td>m96</td>
      <td>41564</td>
      <td>12056</td>
      <td>80.791667</td>
      <td>80</td>
      <td>2</td>
      <td>18</td>
      <td>...</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>165.0</td>
      <td>1.0</td>
      <td>106.0</td>
      <td>60.0</td>
      <td>2</td>
      <td>3</td>
      <td>others</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ADNI2</td>
      <td>21</td>
      <td>v41</td>
      <td>m120</td>
      <td>42311</td>
      <td>12056</td>
      <td>82.836111</td>
      <td>82</td>
      <td>2</td>
      <td>18</td>
      <td>...</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>158.0</td>
      <td>1.0</td>
      <td>98.0</td>
      <td>58.0</td>
      <td>2</td>
      <td>3</td>
      <td>others</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5087</th>
      <td>ADNI3</td>
      <td>6822</td>
      <td>sc</td>
      <td>sc</td>
      <td>43741</td>
      <td>21187</td>
      <td>61.752778</td>
      <td>61</td>
      <td>2</td>
      <td>14</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>197.6</td>
      <td>1.0</td>
      <td>127.0</td>
      <td>93.0</td>
      <td>3</td>
      <td>3</td>
      <td>others</td>
    </tr>
    <tr>
      <th>5093</th>
      <td>ADNI3</td>
      <td>6831</td>
      <td>sc</td>
      <td>sc</td>
      <td>43776</td>
      <td>15713</td>
      <td>76.833333</td>
      <td>76</td>
      <td>2</td>
      <td>18</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>160.0</td>
      <td>1.0</td>
      <td>127.0</td>
      <td>80.0</td>
      <td>4</td>
      <td>4</td>
      <td>E4</td>
    </tr>
    <tr>
      <th>5097</th>
      <td>ADNI3</td>
      <td>6834</td>
      <td>sc</td>
      <td>sc</td>
      <td>43780</td>
      <td>13522</td>
      <td>82.844444</td>
      <td>82</td>
      <td>2</td>
      <td>16</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>127.0</td>
      <td>1.0</td>
      <td>124.0</td>
      <td>54.0</td>
      <td>3</td>
      <td>3</td>
      <td>others</td>
    </tr>
    <tr>
      <th>5116</th>
      <td>ADNI3</td>
      <td>6853</td>
      <td>sc</td>
      <td>sc</td>
      <td>43865</td>
      <td>19360</td>
      <td>67.091667</td>
      <td>67</td>
      <td>2</td>
      <td>12</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>132.0</td>
      <td>1.0</td>
      <td>161.0</td>
      <td>71.0</td>
      <td>3</td>
      <td>3</td>
      <td>others</td>
    </tr>
    <tr>
      <th>5125</th>
      <td>ADNI3</td>
      <td>6872</td>
      <td>sc</td>
      <td>sc</td>
      <td>44035</td>
      <td>19727</td>
      <td>66.555556</td>
      <td>66</td>
      <td>2</td>
      <td>16</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>165.8</td>
      <td>1.0</td>
      <td>127.0</td>
      <td>87.0</td>
      <td>3</td>
      <td>4</td>
      <td>E4</td>
    </tr>
  </tbody>
</table>
<p>2051 rows × 47 columns</p>
</div>



## Implementing the Deviation maps( Including mu and sigma calculations)


```python

```



### References: 

```
sayantan.k (2022) NormVAE: Normative modelling on neuroimaging data using Variational Autoencoders, arXiv:2110.04903v2 [eess.IV] 30 Jan 2022
```
