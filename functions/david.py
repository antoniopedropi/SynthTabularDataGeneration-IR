# Script path: functions/david.py

# This script was adapted from an orginal that is part of the 'DAVID' repository, which was developed by: 
# Stocksieker, S. (2024). DAVID: Data Augmentation with Variational Autoencoder for Imbalanced Dataset [Computer software]. GitHub. Retrieved from https://github.com/sstocksieker/DAVID.

# The 'DAVID' algorithm was implemented based on the following papers: 
# Stocksieker, S., Pommeret, D., & Charpentier, A. (2024). Data Augmentation with Variational Autoencoder for Imbalanced Dataset. arXiv preprint arXiv:2412.07039. Retrieved from https://arxiv.org/abs/2412.07039.

# This script contains a function to generate synthetic data using a Variational Autoencoder (VAE) for imbalanced datasets.


## load dependencies - third party
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA
from scipy.stats import gaussian_kde
from tqdm import trange
from scipy.linalg import LinAlgError

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 3, 3
sns.set(rc={'figure.figsize':(3,3)})


## load dependencies - internal
from functions import aux_functions


# Set the random seed for reproducibility
np.random.seed(4040)
seed_sample = np.random.randint(1000);


# Set the random seed for PyTorch
torch.manual_seed(4040)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Set the random seed for CUDA (if available)
if torch.cuda.is_available():
    torch.cuda.manual_seed(4040)
    torch.cuda.manual_seed_all(4040)


# Set the device to GPU if available, otherwise use CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# Declare global variables
graine = 0

encDim_S = 5
encDim_M = 10
encDim_L = 15

hp_fae_w_corr = None
hp_fae_sparsity_factor = None
hp_fae_sparsity_target=0.05
hp_fae_sparsity_mode=None
hp_fae_wNorm=False
hp_fae_mirror=False
hp_fae_encDim=1
hp_fae_epochs=1000
hp_fae_batch_size=128
hp_fae_lr=10e-3
hp_fae_lossfunc="wMSE"
hp_fae_dimHL=5
hp_fae_w_Y = None
hp_dimLat = 10 
hp_betaKLD=1e-6 
hp_betaX=1 
hp_betaY=3

hp_power=1
hp_penalRank=1

plotModel=False


# Function to transform the data
# This function scales the quantitative features and applies one-hot encoding to the categorical features
# It returns the transformed data
def transformer(inputs, mode_quanti="MinMax", mode_quali="OHE"):
    quanti = inputs.select_dtypes(include=['number'])
    quali = inputs.select_dtypes(include=['object', 'category'])
    name_quanti = quanti.columns
    if mode_quanti=="MinMax" :
      scaler = MinMaxScaler()
      quanti_t = pd.DataFrame(scaler.fit_transform(quanti), columns=name_quanti, index=quanti.index)
    else:
      scaler = StandardScaler()
      quanti_t = pd.DataFrame(scaler.fit_transform(quanti))
      quanti_t.columns = name_quanti
    if quali.shape[1]==0:
        return quanti_t
    else:
        quali_t = pd.get_dummies(quali).astype(int)
        return pd.concat([quanti_t, quali_t], axis=1)

  
# Function to inverse transform the data
# This function scales back the quantitative features and applies inverse one-hot encoding to the categorical features
# It returns the original data
def inv_transformer(inputs, outputs, mode_quanti="MinMax", mode_quali="OHE"):
  quanti = inputs.select_dtypes(include=['number'])
  quali = inputs.select_dtypes(include=['object', 'category'])
  name_quanti = quanti.columns
  if quanti.shape[1] > 0 :
    if mode_quanti=="MinMax" :
      scaler = MinMaxScaler()
      scaler.fit(quanti)
      res_quanti =  pd.DataFrame(scaler.inverse_transform(outputs[name_quanti]))
    elif mode_quanti=="SC" :
      scaler = StandardScaler()
      scaler.fit(quanti)
      res_quanti =  pd.DataFrame(scaler.inverse_transform(outputs[name_quanti]))
    else:
      res_quanti =  pd.DataFrame(outputs[name_quanti])
    res_quanti.columns = name_quanti
    res_quanti.index = outputs.index
  if quali.shape[1] > 0 :
    quali_t = pd.get_dummies(quali)
    name_quali_t = quali_t.columns
    res_quali = outputs.copy()
    res_quali.drop(name_quanti, axis=1, inplace=True)
    res_quali.index = quali_t.index
    res_quali = np.round(res_quali,0)
    if quanti.shape[1] > 0 :
      return pd.concat([res_quanti, res_quali], axis=1)
    else:
      return res_quali
  else:
    return res_quanti


# Function to calculate the imbalanced weighting
# This function uses Gaussian KDE to estimate the density of the data and returns the weights
def IR_weighting(Y, plot=False, alpha=1/2):
    w= 1/gaussian_kde(Y).evaluate(Y)**(alpha)
    w=w/sum(w)
    if plot==True:
        sns.set(rc={'figure.figsize':(3,3)})
        sns.histplot(Y)
        plt.show()
        sns.scatterplot(x=Y,y=w)
    return w


# Function to split the data into training and testing sets
# This function randomly selects a subset of the data for training and testing
# It returns a dictionary containing the training and testing sets
def trainTest(data, test_size=None,np_seed=None,w=None, train_size=0.6):
    if np_seed is None:
        np.random.seed()
    else:
        np.random.seed(np_seed)
    n = data.shape[0]
    if w is None:w=np.repeat(1,n)/n
    n_train=round(train_size*n)
    id_train = np.random.choice(n, size=n_train, p=w , replace=False)
    X_train = data.iloc[id_train,:]
    X_test = data.drop(index=id_train)
    
    return{'X_train':X_train,'X_test':X_test}


# Class for the Autoencoder (AE) model
# This class defines the architecture of the autoencoder
# It includes the encoder and decoder networks
# The encoder maps the input data to a lower-dimensional representation
# The decoder maps the lower-dimensional representation back to the original input space
class AEy(nn.Module):   
    def __init__(self, p,dimHL=hp_fae_dimHL, power = hp_power,penal_rank=hp_penalRank, dimLat=hp_dimLat,connect=True, betaKLD=None):
        super().__init__()
        self.p = p
        self.dimLat = dimLat
        q = int(p/10)+1
        self.encoders = nn.Sequential(
                    nn.Linear(p, 2*p),
                    nn.Tanh(),
                    nn.Linear(2*p, p-1*q),
                    nn.Tanh(),
                    nn.Linear(p-1*q, p-2*q),
                    nn.Tanh(),
                    nn.Linear(p-2*q, p-3*q))

        self.decoders = nn.Sequential(
                    nn.Linear(p-3*q, p-2*q),
                    nn.Tanh(),
                    nn.Linear(p-2*q, p-1*q),
                    nn.Tanh(),
                    nn.Linear(p-1*q, 2*p),
                    nn.Tanh(),
                    nn.Linear(2*p, p))
        
    def encode(self, x) :
        encoded=self.encoders(x)
        return encoded
    
    def decode(self, z):
        decoded = self.decoders(z)
        return decoded[:,:-1],decoded[:,[-1]]
    
    def forward(self, x,y):
        encoded = self.encode(torch.cat((x,y),dim=1))  
        decoded_x, decode_y = self.decode(encoded)
        return decoded_x, decode_y, encoded, encoded*0, encoded*0 


# Class for the Variational Autoencoder (VAE) model
# This class defines the architecture of the VAE
# It includes the encoder and decoder networks
# The encoder maps the input data to a lower-dimensional representation
# The decoder maps the lower-dimensional representation back to the original input space
# The VAE also includes a reparameterization trick to sample from the latent space
# The VAE is trained using a combination of reconstruction loss and KL divergence loss
# The KL divergence loss encourages the latent space to follow a Gaussian distribution
class VAEy(nn.Module):
    def __init__(self, p,dimHL=hp_fae_dimHL,sparsity_factor=None, power = hp_power,penal_rank=hp_penalRank, dimLat=hp_dimLat,connect=True, betaKLD=None):
        super().__init__()
        self.p = p
        self.dimLat = p-3
        self.betaKLD = betaKLD
        q = int(p/10)+1
        self.fc1 = nn.Linear(p+1, 2*p+1)
        self.fc2 = nn.Linear(2*p+1, p-1*q) 
        self.fc3 = nn.Linear(p-1*q, p-2*q) 
        self.fc41 = nn.Linear(p-2*q, p-3*q)
        self.fc42 = nn.Linear(p-2*q,p-3*q) 

        self.fc5 = nn.Linear(p-3*q, p-2*q)
        self.fc6 = nn.Linear(p-2*q, p-1*q) 
        self.fc7 = nn.Linear(p-1*q, 2*p+1) 
        self.fc81 = nn.Linear(2*p+1, p)
        self.fc82 = nn.Linear(2*p+1, 1) 
               
    def encode(self, x, a) :
        h1 = F.tanh(self.fc1(torch.cat([x.view(-1, x.shape[1]),a.view(-1, 1)],1)))  # on utilise une simple ReLu pour la premiere couche de 400 neurones et ID pour la deuxieme couche pour u et std
        h2 = F.tanh(self.fc2(h1))
        h3 = F.tanh(self.fc3(h2))
        mu = self.fc41(h3)
        logvar = self.fc42(h3)
        return mu, logvar
              
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)  # racine de la variance ==> std
        eps = torch.randn_like(std)   # une loi normal entre 0 et 1 avec une taille de std soit de 20 !!
        return mu + eps*std  # z = u + sigm * eps
  
    def decode(self, z):
        h5 = F.tanh(self.fc5(z))
        h6 = F.tanh(self.fc6(h5))
        h7 = F.tanh(self.fc7(h6))
        decoded_x = self.fc81(h7)
        decoded_y = self.fc82(h7)
        return decoded_x, decoded_y
    
    def forward(self, x, a):
        mu,logvar = self.encode(x.view(-1, x.shape[1]), a) 
        encoded = self.reparameterize(mu, logvar)
        decoded_x, decode_y = self.decode(encoded)
        return decoded_x, decode_y, encoded, mu, logvar 
        

# Function to calculate the balanced mean squared error loss
# This function calculates the mean squared error between the reconstructed and original data
# It also applies a weighting factor to balance the loss
# The weighting factor is used to give more importance to certain samples in the dataset
# This is useful for imbalanced datasets where some classes/values may be underrepresented
# The function returns the weighted mean squared error loss
def balanced_mse_loss(recon_y, y, w):   
    # print("torch.nn.functional.mse_loss(recon_y, y, reduction='sum')",torch.nn.functional.mse_loss(recon_y, y, reduction='sum'))
    # print("torch.sum(w*(recon_y-y).T**2",torch.sum(w*(recon_y-y).T**2))
    return torch.sum(w*(recon_y-y).T**2)


# This function calculates the loss function for the Variational Autoencoder (VAE)
# The loss function consists of three components:
# 1. Binary Cross-Entropy (BCE) loss for the reconstructed input data
# 2. BCE loss for the reconstructed output data
# 3. KL divergence loss for the latent space
# The function takes the reconstructed input data, latent space representation, reconstructed output data,
# original output data, original input data, mean and log variance of the latent space as inputs
# It returns the total loss, BCE loss for the input data, BCE loss for the output data, and KL divergence loss
# The function also allows for weighting the output data loss using a weighting factor
def loss_function_VAE(recon_x, z, recon_y, y, x, mu, logvar, epoch,i, betaX = 1,betaY = 0,  betaKLD = 0.5, w_Y=None):
    BCE_X = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
    if w_Y is None:
        BCE_Y = torch.nn.functional.mse_loss(recon_y, y, reduction='sum')
    else:
        BCE_Y = balanced_mse_loss(recon_y, y, w=w_Y)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # ici c'est la KL pour une loi normal 0 & 1 uniquement
    return betaX*BCE_X + betaKLD*KLD + betaY*BCE_Y , betaY*BCE_Y , betaX*BCE_X  , betaKLD*KLD  #+ mmd  # ici on additionne tout simplement les deux pour la loss function avec BCE pour retrouver X et KLD pour ne pas s'éloigner d'une loi normale


# Function to train the Variational Autoencoder (VAE) model
# This function takes the training data, labels, and various hyperparameters as inputs
# It initializes the model, optimizer, and loss function
# It trains the model using mini-batch gradient descent with AdamW optimizer
# The function returns the trained model, reconstructed input data, reconstructed output data,
# latent space representation, mean and log variance of the latent space, and loss history
# The function also allows for different types of models (VAE or AE) and connections between layers
def train_VAEy(X_train,y_train, seed=seed_sample, lr=hp_fae_lr, batch_size=hp_fae_batch_size, epochs=hp_fae_epochs, dimHL=hp_fae_dimHL,dimLat=hp_dimLat,stacked=True, 
              power=hp_power,penal_rank=hp_penalRank, betaX = hp_betaX,betaY = hp_betaY,  betaKLD=hp_betaKLD, connect=False,type_model="VAE", w_Y=None):
        
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        print("betaX",betaX)
        print("betaKLD",betaKLD)
        print("betaY",betaY)
        X_train = transformer(X_train)
        y_train = transformer(y_train)
        inputsX=torch.FloatTensor(X_train.to_numpy()).to(device)
        inputsY=torch.FloatTensor(y_train.to_numpy()).to(device)
        if w_Y is not None:
            w_Y=torch.FloatTensor(w_Y)*y_train.shape[0]
        else:
            w_Y=torch.FloatTensor(np.repeat(1,y_train.shape[0]))
        p = X_train.shape[1]
        if type_model=="VAE":
            print("type model : VAEy")
            model = VAEy(p,dimHL=dimHL, power = power,penal_rank=penal_rank, dimLat=dimLat, connect=connect).to(device)
        elif type_model=="AE":
            print("type model : AEy")
            model = AEy(p+1,dimHL=dimHL, power = power,penal_rank=penal_rank, dimLat=dimLat, connect=connect).to(device)
            
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
        batch_idx=0
        # log_interval=50
        batch_no = len(X_train) // batch_size

        decod=[]
        losses=[]
        losses_VAE=[]
        losses_X=[]
        losses_Y=[]
        losses_KL=[]
        
        for epoch in trange(epochs, desc="Proving P=NP", unit="carrots"):
          # x_train,  ytrain = shuffle(X_train, np.expand_dims(y_train,axis = 1))
          xtrain, ytrain, wtrain = shuffle(inputsX, inputsY, w_Y)
          # Mini batch learning
          for i in range(batch_no):
            start = i * batch_size
            end = start + batch_size
            x_var = Variable(xtrain[start:end]).to(device)
            y_var = Variable(ytrain[start:end]).to(device)
            w_var = Variable(wtrain[start:end]).to(device)
            # plt.scatter(y_var.cpu().detach().numpy(),w_var.cpu().detach().numpy())
            # plt.show()

            # model.train()
            train_loss = 0
            optimizer.zero_grad()
            decoded_x, decoded_y, encoded, mu, logvar = model(x_var, y_var)
            loss_VAE, loss_Y, loss_X, loss_KL = loss_function_VAE(decoded_x, encoded, decoded_y, y_var, x_var, mu, logvar,epoch,i, betaX = betaX,betaY = betaY,  
                                                                  betaKLD = betaKLD, w_Y=w_var)
            loss = loss_VAE
            loss.backward()
            train_loss += loss.item()
            losses.append(loss.cpu().detach().numpy()+0)
            losses_VAE.append(loss_VAE.cpu().detach().numpy()+0)
            losses_X.append(loss_X.cpu().detach().numpy()+0)
            losses_Y.append(loss_Y.cpu().detach().numpy()+0)
            losses_KL.append(loss_KL.cpu().detach().numpy()+0)
            optimizer.step()
          decoded_x, decoded_y, encoded, mu, logvar = model(inputsX, inputsY)
        if type_model=="AE":
                mu=encoded
        return {'model':model, "decoded_x":decoded_x, "decoded_y":decoded_y, "encoded":encoded, "mu":mu, "logvar":logvar,
                'losses':losses, 'losses_VAE':losses_VAE, 'losses_X':losses_X, 'losses_KL':losses_KL, 'losses_Y':losses_Y}
    

# Function to generate synthetic data using the trained VAE model
# This function takes the trained model, input data, labels, and various parameters as inputs
# It generates synthetic data by sampling from the latent space and decoding it back to the original input space
# The function also allows for different modes of generation (VAE, kVAE, kPCA, kKPCA)
# It returns the generated synthetic data for both input and output   
def generationXy(res, X, y, w=None, N=None, seed=None, mode="VAE", hmult=0.1):
    n = X.shape[0]
    if N is None:N=n
    if w is None:w=np.repeat(1,n)/n
    if seed is None: seed = np.random.choice(n, size=N, replace=True)
    inputsX0=X.copy()
    inputsy0=y.copy()
    if res is not None:
        inputsX=transformer(X)
        inputsy=transformer(y)
        if mode == "VAE":
            inputsX = np.array(inputsX)[seed,:]
            inputsy = np.array(inputsy)[seed,:]
            decoded_x, decoded_y, encoded, mu, logvar = res['model'](torch.FloatTensor(inputsX).to(device),torch.FloatTensor(inputsy).to(device))
            outputs_x=pd.DataFrame(decoded_x.cpu().detach())
            outputs_y=pd.DataFrame(decoded_y.cpu().detach())
            outputs_x.columns=X.columns
            outputs_y.columns=y.columns
        elif mode in ["kVAE","kAE"]:
            mu=res['mu']
            try:
                kde = gaussian_kde(mu.cpu().detach().numpy().T, bw_method = "silverman")
                #kde = gaussian_kde(mu.cpu().detach().numpy().T, bw_method = "silverman")
                H = kde.factor**2 * kde.covariance * hmult
                sim = np.random.multivariate_normal(mean=np.repeat(0,mu.shape[1]),cov=H,size=N)
                encoded = mu[seed,:] + torch.FloatTensor(sim).to(device)
            except LinAlgError:
                print("⚠ Covariance matrix of latent space is singular. Applying PCA to decorrelate latent variables...")
                from sklearn.decomposition import PCA

                mu_np = mu.cpu().detach().numpy()
                n_components = min(20, mu_np.shape[1], mu_np.shape[0] - 1)
                pca = PCA(n_components=n_components)
                mu_pca = pca.fit_transform(mu_np)

                kde = gaussian_kde(mu_pca.T, bw_method="silverman")

                H = kde.factor**2 * kde.covariance * hmult

                # Ajustar o tamanho do vetor de médias conforme o que foi usado no KDE
                dim = H.shape[0]
                sim = np.random.multivariate_normal(mean=np.zeros(dim), cov=H, size=N)


                sim_original_space = pca.inverse_transform(sim)
                encoded = torch.FloatTensor(sim_original_space).to(device)


            
            decoded_x, decoded_y = res['model'].decode(encoded)
            outputs_x=pd.DataFrame(decoded_x.cpu().detach())
            outputs_y=pd.DataFrame(decoded_y.cpu().detach())
            outputs_x.columns=X.columns
            outputs_y.columns=y.columns
        outputs_x = inv_transformer(X,outputs_x)
        outputs_y = inv_transformer(y,outputs_y)
    else:
        if mode == "kPCA":
            Xytrain=pd.concat([y,X],axis=1)
            pca = PCA(n_components=Xytrain.shape[1])
            scaler = StandardScaler()
            Xytrain_sc = scaler.fit_transform(Xytrain)
            Xyfact = pca.fit_transform(Xytrain_sc)
            inputs = Xyfact[graine,:]
            kde = gaussian_kde(Xyfact.T, bw_method = "silverman")
            H = kde.factor**2 * kde.covariance * hmult
            sim = np.random.multivariate_normal(mean=np.repeat(0,inputs.shape[1]),cov=H,size=N)
            synth = pd.DataFrame(inputs + sim)
            synth = pca.inverse_transform(synth)
            synth = pd.DataFrame(scaler.inverse_transform(synth))
            synth.columns=Xytrain.columns
            synth.set_index(Xytrain.iloc[graine,:].index,inplace=True)
            outputs_x = synth[X.columns]
            outputs_y = synth[y.columns]
        elif mode == "kKPCA":
            Xytrain=pd.concat([y,X],axis=1)
            pca = KernelPCA(n_components=Xytrain.shape[1], kernel='poly', fit_inverse_transform=True)
            scaler = StandardScaler()
            Xytrain_sc = scaler.fit_transform(Xytrain)
            Xyfact = pca.fit_transform(Xytrain_sc)
            inputs = Xyfact[graine,:]
            kde = gaussian_kde(Xyfact.T, bw_method = "silverman")
            H = kde.factor**2 * kde.covariance * hmult
            sim = np.random.multivariate_normal(mean=np.repeat(0,inputs.shape[1]),cov=H,size=N)
            synth = pd.DataFrame(inputs + sim)
            synth = pca.inverse_transform(synth)
            synth = pd.DataFrame(scaler.inverse_transform(synth))
            synth.columns=Xytrain.columns
            synth.set_index(Xytrain.iloc[graine,:].index,inplace=True)
            outputs_x = synth[X.columns]
            outputs_y = synth[y.columns]
        else:
            inputs0=pd.concat([inputsX0,inputsy0],axis=1).to_numpy()
            inputs=pd.concat([inputsX0.iloc[seed,:],inputsy0.iloc[seed,:]],axis=1).to_numpy()
            kde = gaussian_kde(inputs0.T, bw_method = "silverman")
            H = kde.factor**2 * kde.covariance * hmult
            sim = np.random.multivariate_normal(mean=np.repeat(0,inputs.shape[1]),cov=H,size=N)
            outputs = pd.DataFrame(inputs + sim)
            outputs_x = outputs.iloc[:,:-1]
            outputs_y = pd.DataFrame(outputs.iloc[:,-1])
            outputs_x.columns=X.columns
            outputs_y.columns=y.columns
    print(outputs_x)
    print(X)
    outputs_x.index=X.iloc[graine,:].index
    outputs_y.index=y.iloc[graine,:].index
    # outputs_x['seed']=seed
    return outputs_x,outputs_y


# Function to generate synthetic data using the DAVID algorithm
# This function takes the input data, label for the target variable, alpha value for weighting,
# and proportion of data to be used for training as inputs
# It performs data preprocessing, including removing non-numeric features and handling missing values
# It generates synthetic data using the trained VAE model and returns the generated data
# The function also visualizes the distribution of the target variable and the generated data
# It uses the IR weighting method to balance the dataset
def david(data, y_label, alfa=1, proportion=0.6, drop_na_col=True, drop_na_row=True):
    
    data = data.copy()  # make an internal copy to garantee that the original data is not modified (especially
    # when the "inplace" option is used in the dropna function)

    #data[y_label].describe()
    
    #data.info()
    
    lab_Y = y_label

    if drop_na_col:
        data.dropna(axis=1, inplace=True)
    if drop_na_row:
        data.dropna(axis=0, inplace=True)
    
    #data
    
    #data.info()
    
    #data.describe()
    
    #data = aux_functions.remove_non_numeric_features(data)
    
    sns.histplot(data[lab_Y])

    # Estimation of the probability density function (PDF) of the given data using Kernel Density Estimation (KDE) #
    # kdem = gaussian_kde(np.array(data.T),bw_method="scott")   # The parameter <bw_method="scott"> sets the bandwidth (smoothing parameter) using Scott's Rule, which automatically selects a bandwidth based on the data size and variance, affecting the smoothness of the KDE curve
    # kdem_covariance = pd.DataFrame(kdem.covariance)

        # Estimation of the probability density function (PDF) of the given data using Kernel Density Estimation (KDE) #
    try:
        kdem = gaussian_kde(np.array(data.T), bw_method="scott")
    except LinAlgError:
        print("⚠ Covariance matrix singular or not positive definite. Applying PCA to decorrelate variables...")

        from sklearn.decomposition import PCA

        X = data.drop(columns=[lab_Y]).to_numpy()

        # Número de componentes — no máximo 20 ou menos que número de variáveis
        n_components = min(20, X.shape[1], X.shape[0] - 1)

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)

        # Reconstruir o dataframe com as componentes + target
        data_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
        data_pca[lab_Y] = data[lab_Y].values

        # Tentar KDE novamente
        kdem = gaussian_kde(np.array(data_pca.T), bw_method="scott")
    
    # Imbalanced Weighting based on Gaussian KDE #
    matplotlib.rc_file_defaults()
    ax1 = sns.set_style(style=None, rc=None )

    fig, ax1 = plt.subplots(figsize=(12,6))

    sns.lineplot(x=data[lab_Y],y=IR_weighting(data[lab_Y], plot=False, alpha=2),ax=ax1,label="a=2")
    sns.lineplot(x=data[lab_Y],y=IR_weighting(data[lab_Y], plot=False, alpha=1),ax=ax1,label="a=1")
    sns.lineplot(x=data[lab_Y],y=IR_weighting(data[lab_Y], plot=False, alpha=1/2),ax=ax1,label='a=1/2')
    sns.lineplot(x=data[lab_Y],y=IR_weighting(data[lab_Y], plot=False, alpha=1/3), ax=ax1, label='a=1/3')
    # ax1.legend(['alpha=1',  'alpha=1/3', "alpha=1/3","alpha=1/3"])
    plt.legend()
    ax2 = ax1.twinx()
    sns.histplot(data=data,x=lab_Y, alpha=0.5, ax=ax2,kde=True, stat="probability", color="salmon")
 #   plt.savefig('graph_Alpha.png')
    
    w_Y=IR_weighting(data[lab_Y], plot=True, alpha=alfa)
    
    
    train_size = proportion
    test_size = 1 - proportion
    
    split = trainTest(data,test_size,seed_sample,w=w_Y,train_size=train_size)
    X_train = split['X_train']
    X_test = split['X_test']
    print(X_train.shape)
    print(X_test.shape)
    
    train_df = pd.DataFrame(X_train[[lab_Y]])
    train_df['sample']='train'
    test_df = pd.DataFrame(X_test[[lab_Y]])
    test_df['sample']='test'
    sns.histplot(data=pd.concat([train_df,test_df],axis=0),x=lab_Y,hue='sample',kde=True, stat="density")
    
    print(X_train[[lab_Y]].describe())
    sns.histplot(X_train[[lab_Y]])
    
    colY = np.where(X_train.columns==lab_Y)[0][0] ; 
    print(colY)
    
    y_train =  X_train[[lab_Y]]
    X_train = X_train.drop(columns=lab_Y, axis=1)
    w_Y=IR_weighting(y_train[lab_Y], alpha=alfa)
    sns.scatterplot(x=y_train[lab_Y],y=w_Y)
    
    epoch_simu = 2000
    
    global plotModel
    plotModel =True
    
    res_BVAEw  = train_VAEy(X_train=X_train,y_train=y_train, seed=seed_sample, stacked=None, epochs=epoch_simu,type_model="VAE",w_Y=w_Y)
    
    N = X_train.shape[0]
    
    graine_local = np.random.choice(X_train.shape[0], size=N, p=w_Y , replace=True)
    
    global graine
    
    graine = graine_local
    
    hmult=0.01
    
    y_grain = y_train.iloc[graine,:]
    
    sns.set(rc={'figure.figsize':(5,5)})
    sns.histplot(y_grain, kde=True)
    plt.show()
    train_df = pd.DataFrame(y_train)
    train_df['sample']='train'
    tir_df = pd.DataFrame(y_grain)
    tir_df['sample']='Tirage'
    sns.histplot(data=pd.concat([train_df,tir_df],axis=0),x=lab_Y,hue='sample',kde=True, stat="density")
    
    synth_X, synth_y = generationXy(res_BVAEw, X_train, y_train, seed=graine, mode="kVAE",N=N, hmult=hmult)
    synth = pd.concat([synth_X,synth_y],axis=1)
    synth_kBVAEw=pd.concat([synth,X_test], axis=0)
    
    
    
    matplotlib.rc_file_defaults()
    
    return synth_kBVAEw