###########################################################
'''
    Copyright 2017, MIT

    This file is part of PremieRL.
    
'''
############################
# Written by Tim Ioannidis #
############################

# ******************************************* #
# ********* Class of betting game  ********** #
# ******************************************* #

import numpy as np

class Bet():
    ##########################
    #### Initialize class ####
    #### load S,R, capital ###
    ##########################
    def __init__(self,S_all,R_all,available_capital=10,include_money=False,dynamic_bet=False,validation_seasons=False):
        self.S_all = S_all
        self.R_all = R_all
        self.include_money = include_money
        self.available_capital_init = available_capital
        self.dynamic_bet = dynamic_bet
        self.total_seasons = len(S_all) if not validation_seasons else (len(S_all)-validation_seasons)
        self.reset()    # start playing from beginning of one season
    
    ########################
    #### Start new game ####
    ########################
    def reset(self,season_to_play=-1):
        ### reset profit to 0 (for scaling), and available capital
        self.profit = 0
        self.available_capital = self.available_capital_init
        ### get random number for game/season
        if season_to_play==-1:
            season_to_play = np.random.randint(0,self.total_seasons)
        self.season = season_to_play
        ### states for current game
        self.S = self.S_all[season_to_play]
        ### rewards for current game
        self.R = self.R_all[season_to_play]
        ### initialize match counter and total matches
        self.match = 0 
        self.total_matches = self.S.shape[0]
        ### start with 1st game for current season
        self.state = self.S[self.match]
        ### if current profit is to be included augment the state vector
        if self.include_money:
            self.state = np.insert(self.state,0,self.profit)
        ### get sizes
        self.state_size = self.state.shape[0]
        self.action_size = self.R.shape[1]
        self.state = self.state[:,np.newaxis].T # add axis and correct shape (1,No_Vars)
        self.rewards = self.R[self.match]

    ### get game name
    def name(self):
        return "Bet"

    ### get number of actions
    def nb_actions(self):
        # actions are the following 4: bet 1, bet X, bet 2, dont' bet
        return 4 

    ### get number of state variables
    def nb_statevars(self):
        return self.state.shape[1]

    ###################################
    #### Update state after action ####
    ###################################
    def play(self, action):
        ### get reward and update total
        r = self.rewards[action]
        ### adjust if dynamic bet used
        r = r*self.dynamic_bet*self.available_capital if self.dynamic_bet else r
        self.profit += r
        ### update available capital
        self.available_capital += r
        ### increment action counter
        self.match += 1
        ### update state and check if game is over
        if not self.is_over():
            self.state = self.S[self.match]
            self.rewards = self.R[self.match]
            if self.include_money:
                self.state = np.insert(self.state,0,self.profit)
            self.state = self.state[:,np.newaxis].T # add axis and correct shape (1,No_Vars)
        return r

    ##############################
    #### Return current state ####
    ##############################
    def get_state(self):
        return self.state

    #######################################
    ### Check if we are out of money or ###
    ###  season has ended and terminate ###
    #######################################
    def is_over(self):
        if (self.available_capital <= 0):
            self.available_capital = 0 
            self.profit = -self.available_capital_init
            return True
        elif (self.match == self.total_matches):
            return True
        else:
            return False

    ##############################
    ### Check if we made money ###
    ##############################
    def is_won(self):
        if self.profit > 0:
            return True
        else:
            return False

    ###########################
    ### Print class methods ###
    ###########################
    def __repr__(self):
        ss = "\nClass Game has the following methods:\n"
        for method in dir(self):
            if callable(getattr(self, method)):
                ss += method +'\n'
        return ss






