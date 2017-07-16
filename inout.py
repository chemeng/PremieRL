###########################################################
'''
    Copyright 2017, MIT

    This file is part of PremieRL.
    
'''
############################
# Written by Tim Ioannidis #
############################

# ****************************************** #
# ********* Handles Input/Output  ********** #
# ****************************************** #

import joblib, os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

### grab training states matrix S[s] and rewards R(s,a)
def get_states(datafile,no_actions,one_game_per_season=True):
    df = pd.read_csv(datafile,index_col=False) # read data file into dataframe
    ### get parameters
    no_all_states = df.shape[0]     # states in training
    ### get parameters
    seasons = df.iloc[:,1].unique() # get unique seasons
    no_seasons = len(seasons) if one_game_per_season else 1
    msg = "A total of "+str(len(seasons))+" seasons with "+str(no_all_states)+" matches was loaded.\n"
    print msg
    ### initialize list of states/rewards/game info
    S = []
    R = []
    I = []
    #no_games
    for i,season in enumerate(seasons):
        ### get current season
        df_local = df[df.season==season]
        ### get parameters
        no_states = df_local.shape[0]
        ### get states matrix
        Slocal = df_local.iloc[:,6:].values
        res = df_local.iloc[:,5].values       # get results list
        ### build rewards matrix
        Rlocal = np.zeros((no_states,no_actions))-1 # initialize to -1
        # construct payoff matrix
        for i,s in enumerate(res):
            if s=='1':
                Rlocal[i,0] += Slocal[i,0] # reward for 1
            elif s=='X':
                Rlocal[i,1] += Slocal[i,1] # reward for X
            elif s=='2':
                Rlocal[i,2] += Slocal[i,2] # reward for 2
            Rlocal[i,3] = 0  # reward for not playing
        ### append stuff
        R.append(Rlocal)
        S.append(Slocal)
        I.append(df_local.iloc[:,0:9])
    # construct aggregate match state
    Rag = np.concatenate((R[:]),axis=0)
    Sag = np.concatenate((S[:]),axis=0)
    ### merge if one game total
    if not one_game_per_season:
        ### convert to list of (1,states,actions)
        R = [Rag]
        S = [Sag]
        seasons = ['All']
    season_per_game = 1 if one_game_per_season else len(S)
    msg = "Training with "+str(season_per_game)+" season(s) per game and \n"+'/'.join(str(s.shape[0]) for s in S)
    msg += " matches per corresponding season."
    print msg
    ### scale state data based on aggregate Sag
    scalerX = False
    if os.path.isfile('model/scalerx.pkl'):
        scalerX = joblib.load('model/scalerx.pkl')
    S, scalerX = scale_data(S,Sag,scalerX)
    ### save scaler for future use
    joblib.dump(scalerX, 'model/scalerx.pkl')
    return S,R,I,seasons


##############################
##### Standardize Input ######
##############################
def scale_data(X,Xag,scalerX):
    ### get scaler
    if not scalerX:
        # flatten array to aggregate different observations
        scalerX = StandardScaler().fit(Xag)
    # apply scaler
    X_scaled = []
    for i in range(0,len(X)):
        X_scaled.append(scalerX.transform(X[i]))
    return X_scaled,scalerX

###############################
##### INVERSE STANDARDIZE #####
###############################
def scale_back_data(X,scalerX=None):
    # reshape X for accounting for all samples
    num_stocks = len(X[:,0,0])
    num_timesteps = len(X[0,:,0])
    num_features = len(X[0,0,:])
    # apply scaler for each stock
    X_scaled = np.zeros((num_stocks,num_timesteps,num_features))
    for i in range(0,num_stocks):
        X_scaled[i,:,:] = scalerX.inverse_transform(X[i,:,:])
    return X_scaled,scalerX





