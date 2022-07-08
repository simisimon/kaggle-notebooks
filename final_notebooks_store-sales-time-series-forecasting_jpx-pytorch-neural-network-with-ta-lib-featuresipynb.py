#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[ ]:


# install talib without internet
get_ipython().system('pip install ../input/talib0419/talib_binary-0.4.19-cp37-cp37m-manylinux1_x86_64.whl')


# In[ ]:


import pandas as pd
import numpy as np
import talib
from pprint import pprint
import gc
import os
from tqdm.notebook import tqdm
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import jpx_tokyo_market_prediction


# In[ ]:


# Set random seed
seed = 30
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# # Load Data

# In[ ]:


df_prices = pd.read_csv('../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv')
df_prices.tail()


# In[ ]:


# Drop missing values
df_prices = df_prices[['RowId', 'Date', 'SecuritiesCode', 'Open', 'High', 'Low', 'Close', 'Volume', 'Target']]
df_prices = df_prices.dropna(axis=0).reset_index(drop=True)
df_prices.tail()


# # Get Talib Features

# All the TA functions sorted by group:

# In[ ]:


pprint(talib.get_function_groups())


# In[ ]:


def get_talib_features(df):
    """
    Get technical features from TA-Lib
    """
    op = df['Open']
    hi = df['High']
    lo = df['Low']
    cl = df['Close']
    vo = df['Volume']
    
    # Overlap Studies
    df['BBANDS_upper'], df['BBANDS_middle'], df['BBANDS_lower'] = talib.BBANDS(cl, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    df['DEMA'] = talib.DEMA(cl, timeperiod=30)
    df['EMA'] = talib.EMA(cl, timeperiod=30)
    df['HT_TRENDLINE'] = talib.HT_TRENDLINE(cl)
    df['KAMA'] = talib.KAMA(cl, timeperiod=30)
    df['MA'] = talib.MA(cl, timeperiod=30, matype=0)
    df['MIDPOINT'] = talib.MIDPOINT(cl, timeperiod=14)
    df['SAR'] = talib.SAR(hi, lo, acceleration=0, maximum=0)
    df['SAREXT'] = talib.SAREXT(hi, lo, startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)
    df['SMA'] = talib.SMA(cl, timeperiod=30)
    df['T3'] = talib.T3(df['Close'], timeperiod=5, vfactor=0)
    df['TEMA'] = talib.TEMA(df['Close'], timeperiod=30)
    df['TRIMA'] = talib.TRIMA(df['Close'], timeperiod=30)
    df['WMA'] = talib.WMA(df['Close'], timeperiod=30)
    
    # Momentum Indicators
    df['ADX'] = talib.ADX(hi, lo, cl, timeperiod=14)
    df['ADXR'] = talib.ADXR(hi, lo, cl, timeperiod=14)
    df['APO'] = talib.APO(cl, fastperiod=12, slowperiod=26, matype=0)
    df['AROON_down'], df['AROON_up'] = talib.AROON(hi, lo, timeperiod=14)
    df['AROONOSC'] = talib.AROONOSC(hi, lo, timeperiod=14)
    df['BOP'] = talib.BOP(op, hi, lo, cl)
    df['CCI'] = talib.CCI(hi, lo, cl, timeperiod=14)
    df['DX'] = talib.DX(hi, lo, cl, timeperiod=14)
    df['MACD_macd'], df['MACD_macdsignal'], df['MACD_macdhist'] = talib.MACD(cl, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MFI'] = talib.MFI(hi, lo, cl, vo, timeperiod=14)
    df['MINUS_DI'] = talib.MINUS_DI(hi, lo, cl, timeperiod=14)
    df['MINUS_DM'] = talib.MINUS_DM(hi, lo, timeperiod=14)
    df['MOM'] = talib.MOM(cl, timeperiod=10)
    df['PLUS_DI'] = talib.PLUS_DI(hi, lo, cl, timeperiod=14)
    df['PLUS_DM'] = talib.PLUS_DM(hi, lo, timeperiod=14)
    df['RSI'] = talib.RSI(cl, timeperiod=14)
    df['STOCH_slowk'], df['STOCH_slowd'] = talib.STOCH(hi, lo, cl, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['STOCHF_fastk'], df['STOCHF_fastd'] = talib.STOCHF(hi, lo, cl, fastk_period=5, fastd_period=3, fastd_matype=0)
    df['STOCHRSI_fastk'], df['STOCHRSI_fastd'] = talib.STOCHRSI(cl, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    df['TRIX'] = talib.TRIX(cl, timeperiod=30)
    df['ULTOSC'] = talib.ULTOSC(hi, lo, cl, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    df['WILLR'] = talib.WILLR(hi, lo, cl, timeperiod=14)
    
    # Volume Indicators
    df['AD'] = talib.AD(hi, lo, cl, vo)
    df['ADOSC'] = talib.ADOSC(hi, lo, cl, vo, fastperiod=3, slowperiod=10)
    df['OBV'] = talib.OBV(cl, vo)
    
    # Volatility Indicators
    df['ATR'] = talib.ATR(hi, lo, cl, timeperiod=14)
    df['NATR'] = talib.NATR(hi, lo, cl, timeperiod=14)
    df['TRANGE'] = talib.TRANGE(hi, lo, cl)
    
    # Cycle Indicators
    df['HT_DCPERIOD'] = talib.HT_DCPERIOD(cl)
    df['HT_DCPHASE'] = talib.HT_DCPHASE(cl)
    df['HT_PHASOR_inphase'], df['HT_PHASOR_quadrature'] = talib.HT_PHASOR(cl)
    df['HT_SINE_sine'], df['HT_SINE_leadsine'] = talib.HT_SINE(cl)
    df['HT_TRENDMODE'] = talib.HT_TRENDMODE(cl)
    
    # Statistic Functions
    df['BETA'] = talib.BETA(hi, lo, timeperiod=5)
    df['CORREL'] = talib.CORREL(hi, lo, timeperiod=30)
    df['LINEARREG'] = talib.LINEARREG(cl, timeperiod=14) - cl
    df['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(cl, timeperiod=14)
    df['LINEARREG_INTERCEPT'] = talib.LINEARREG_INTERCEPT(cl, timeperiod=14) - cl
    df['LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(cl, timeperiod=14)
    df['STDDEV'] = talib.STDDEV(cl, timeperiod=5, nbdev=1)   
    
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', "results = df_prices.groupby('SecuritiesCode').apply(get_talib_features)\nresults\n")


# In[ ]:


results = results.dropna(axis=0).reset_index(drop=True)
print(results.shape)
results.tail()


# # Pytorch NN Model

# In[ ]:


# Define nn model
class jpx_dnn(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.mlp_network = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Linear(feat_dim, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.mlp_network(x)
        return x


# # Create Dataset & Dataloader

# In[ ]:


class JPXDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        features = self.X[idx]
        labels = self.y[idx]
        return features, labels


# In[ ]:


# Split train and valid data
train_df = results[results.Date < '2021-01-01'].copy().reset_index(drop=True)
valid_df = results[results.Date >= '2021-01-01'].copy().reset_index(drop=True)
X_train = train_df.drop(['RowId', 'Date', 'SecuritiesCode', 'Target'], axis=1).values
y_train = train_df['Target'].values
X_valid = valid_df.drop(['RowId', 'Date', 'SecuritiesCode', 'Target'], axis=1).values
y_valid = valid_df['Target'].values


# In[ ]:


# Create dataset
train_data = JPXDataset(X_train, y_train)
valid_data = JPXDataset(X_valid, y_valid)


# In[ ]:


# Create dataloader
train_dataloader = DataLoader(train_data, batch_size=1024, shuffle=False)
valid_dataloader = DataLoader(valid_data, batch_size=1024, shuffle=False)


# # Train Model

# In[ ]:


# Initialize model
model = jpx_dnn(feat_dim=69)

# Get cpu or gpu device for training
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")
model.to(device)


# In[ ]:


# Set training config
epochs = 5
learning_rate = 0.001
weight_decay = 1.0e-05
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


# In[ ]:


# Define train and test_loop - https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in tqdm(enumerate(dataloader)):
        X, y = X.to(device), y.to(device)
        X, y = X.to(torch.float32), y.to(torch.float32)
        
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred.view(-1), y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 500 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            X, y = X.to(torch.float32), y.to(torch.float32)
            pred = model(X)
            test_loss += loss_fn(pred.view(-1), y).item()

    test_loss /= num_batches
    print(f"Test avg loss: {test_loss:>8f} \n")


# In[ ]:


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(valid_dataloader, model, loss_fn)
print("Done!")


# # Model Evaluation

# ## Define evaluation metrics

# In[ ]:


## JPX Competition Metric - https://www.kaggle.com/code/smeitoma/jpx-competition-metric-definition
def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    """
    Args:
        df (pd.DataFrame): predicted results
        portfolio_size (int): # of equities to buy/sell
        toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
    Returns:
        (float): sharpe ratio
    """
    def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
        """
        Args:
            df (pd.DataFrame): predicted results
            portfolio_size (int): # of equities to buy/sell
            toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
        Returns:
            (float): spread return
        """
        assert df['Rank'].min() == 0
        assert df['Rank'].max() == len(df['Rank']) - 1
        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
        purchase = (df.sort_values(by='Rank')['Target'][:portfolio_size] * weights).sum() / weights.mean()
        short = (df.sort_values(by='Rank', ascending=False)['Target'][:portfolio_size] * weights).sum() / weights.mean()
        return purchase - short

    buf = df.groupby('Date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio

## MSE
def calc_mse(y_test, y_pred):
    return np.mean((y_pred - y_test)**2)

## MAE
def calc_mae(y_test, y_pred):
    return np.mean(np.abs(y_pred - y_test))

## IC & RankIC
def _correlation(df: pd.DataFrame, truth_col: str, pred_col: str, method: str, groupby: str = None) -> float:
    if groupby:
        corr_df = df.groupby(groupby)[[truth_col, pred_col]].corr(method=method)
        return corr_df.loc[(slice(None), truth_col), pred_col].mean()
    return df[[truth_col, pred_col]].corr(method=method).iloc[0, 1]

def spearman_corr(df: pd.DataFrame, truth_col: str, pred_col: str, groupby: str = None) -> float:
    """Spearman (Rank) correlation for regression problem

    Args:
        df (pd.DataFrame): contains truth_col, pred_col and (optinally) groupby columns
        truth_col (str): truth column
        pred_col (str): prediction column
        groupby (str, optional): groupby column. Defaults to None.

    Returns:
        float: correlation
    """
    return _correlation(df, truth_col, pred_col, 'spearman', groupby)

def pearson_corr(df: pd.DataFrame, truth_col: str, pred_col: str, groupby: str = None) -> float:
    """Pearson correlation for regression problem

    Args:
        df (pd.DataFrame): contains truth_col, pred_col and (optinally) groupby columns
        truth_col (str): truth column
        pred_col (str): prediction column
        groupby (str, optional): groupby column. Defaults to None.

    Returns:
        float: correlation
    """
    return _correlation(df, truth_col, pred_col, 'pearson', groupby)


# # Get model predictions

# In[ ]:


pred_df = valid_df[['Date', 'SecuritiesCode', 'Target']].copy()
preds = []
for X, _ in tqdm(valid_dataloader):
    X = X.to(device).to(torch.float32)
    pred = model(X)
    preds.append(pred)
preds = torch.cat(preds).cpu().detach().numpy()


# In[ ]:


pred_df['Target_Pred'] = preds
pred_df['Rank'] = pred_df.groupby('Date')['Target_Pred'].rank(ascending=False, method='first').astype(int) - 1
pred_df


# In[ ]:


sharpe_ratio = calc_spread_return_sharpe(pred_df, portfolio_size=1)
print('Sharpe ratio: ', sharpe_ratio)
mse = calc_mse(pred_df['Target'], pred_df['Target_Pred'])
print('MSE: ', mse)
mae = calc_mae(pred_df['Target'], pred_df['Target_Pred'])
print('MAE: ', mae)
ic = pearson_corr(pred_df, 'Target', 'Target_Pred', 'Date')
print('IC(Pearson correlation):', ic)
rankic = spearman_corr(pred_df, 'Target', 'Target_Pred', 'Date')
print('RankIC(Spearman correlation)', rankic)


# # Save Model

# In[ ]:


torch.save(model.state_dict(), 'pytorch_DNN_weights.pth')


# # Submission

# In[ ]:


df_history = df_prices[df_prices['Date']>= '2021-10-01'].copy().reset_index(drop=True)
df_history


# In[ ]:


# Make predictions and submission
env = jpx_tokyo_market_prediction.make_env()   # initialize the environment
iter_test = env.iter_test()    # an iterator which loops over the test files
for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
    # Combine history data with incoming new data
    display(prices)
    df_current = prices[['RowId', 'Date', 'SecuritiesCode', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df_history = pd.concat([df_history, df_current], ignore_index=True)
    
    # Get processed test data
    training_cutoff = prices['Date'].values[0]
    all_data = df_history.groupby('SecuritiesCode').apply(get_talib_features)
    test_data = all_data[all_data['Date'] == training_cutoff].copy().reset_index(drop=True).fillna(0)
    display(test_data)
    X_test = torch.from_numpy(test_data.drop(['RowId', 'Date', 'SecuritiesCode', 'Target'], axis=1).values).to(device).to(torch.float32)

    # Make predictions
    sample_prediction['target_pred'] = model(X_test).cpu().detach().numpy()
    sample_prediction = sample_prediction.sort_values(by="target_pred", ascending=False)
    sample_prediction['Rank'] = np.arange(2000)
    sample_prediction = sample_prediction.sort_values(by="SecuritiesCode", ascending=True)
    display(sample_prediction)
    sample_prediction.drop(['target_pred'], axis=1, inplace=True)
    env.predict(sample_prediction)  # register your predictions


# # Reference
# * Web Resources:
#     * [**TA-Lib : Technical Analysis Library**](https://www.ta-lib.org/)
#     * [**TA-Lib Documentation**](https://mrjbq7.github.io/ta-lib/)
#     * [**TA-Lib Document in Chinese**](https://github.com/HuaRongSAO/talib-document)
# * Kaggle notebooks:
#     * [***How to install Ta-Lib***](https://www.kaggle.com/code/tera555/how-to-install-ta-lib) by @tera555
#     * [***LGBM baseline (with technical indicator)***](https://www.kaggle.com/code/tera555/lgbm-baseline-with-technical-indicator) by @tera555
#     * [***Feature Engineering + training 📊🤓 with TA***](https://www.kaggle.com/code/metathesis/feature-engineering-training-with-ta) by @metathesis
#     * [***JPX Tokyo: simple LSTM Network***](https://www.kaggle.com/code/aboriginal3153/jpx-tokyo-simple-lstm-network) by @aboriginal3153
