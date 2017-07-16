#!/usr/bin/env python -B
###########################################################
'''
    Copyright 2017, MIT

    This file is part of PremieRL.
    
'''
############################
# Written by Tim Ioannidis #
############################

# ********************************************* #
# ********* Main script of PremieRL  ********** #
# ********************************************* #

import os, argparse, glob
import pandas as pd
from inout import *
from game import Bet
from dqn import DQN
from agent import Agent
from logger import Logger


msg =  "\n***************************************************************"
msg += "\n*********** Welcome to PremieRL. Let's get started! ***********\n"
msg += "***************************************************************\n"
print msg
###################################
###### SPORTSBET PARAMETERS #######
###################################
datafile = 'game_data.csv'      # file with data for RL
available_capital = 10          # total capital available, bet is $1 here (multiply accordingly)
dynamic_bet = 0.1             # if not false, bets proportional to available capital
no_actions = 4                  # actions: 1,X,2,0 (what to bet)
one_game_per_season = True      # each game is only 1 season (F) or all together (T)
include_money  = True           # include money as additional variable in the state
training_seasons = 100         # how many seasons to train for (maximum)
validation_seasons = 3          # how many seasons to use for validation/early stopping
###################################
#######   END PARAMETERS   ########
###################################

###################
### Parse input ###
###################
parser = argparse.ArgumentParser()
parser.add_argument("-t","--t",help="train model",action='store_true')
parser.add_argument("-p","--p",help="play game based on trained model",action='store_true')
parser.add_argument("-r","--r",help="reset Deep Q Network and delete state file",action='store_true')
parser.add_argument("-rt","--rt",help="reset Deep Q Network and train",action='store_true')
args=parser.parse_args()
### print help if nothing
if '-h' in args or (args.p+args.t+args.r+args.rt)==False:
    parser.print_help()
    exit()

### check if old model exists
trained_model = True if os.path.isfile("./model/bet_DDQN.h5") else False
### clean if requested
if args.r or args.rt:
    if os.path.isfile("./model/bet_DDQN.h5"):
        os.remove("./model/bet_DDQN.h5")
    if os.path.isfile("./model/scalerx.pkl"):
        os.remove("./model/scalerx.pkl")
    if not args.rt:
        print "State and scalers reset. Rerun training..\n"
        exit()
    else:
        args.t = True
    trained_model = False

################################
### Load game info from file ###
################################
### get training states S[g,s] (game g) 
### and corresponding action-rewards R[g,s,a] for game g
S_all, R_all, I, seasons = get_states(datafile, no_actions, one_game_per_season)

##################################
### create playing environment ###
##################################
bet = Bet(S_all,R_all,available_capital,include_money,dynamic_bet,validation_seasons)

#############################
### build neural networks ###
#############################
dqn = DQN(bet.nb_statevars(),no_actions)
target_dqn = DQN(bet.nb_statevars(),no_actions)
### load dqn if existing
if trained_model:
    dqn.model.load_weights("./model/bet_DDQN.h5")
target_dqn.update_model(dqn)

############################
### create playing agent ###
############################
agent = Agent(dqn.model,target_dqn.model,validation_seasons)

### create logger
logRL = Logger(I,seasons)

###########################
### Train agent to play ###
###########################
if args.t:
    ### empty logs
    logfiles = glob.glob('train_data/*')
    for f in logfiles:
        os.remove(f)
    ### train net
    agent.train(bet, logRL, training_seasons)
    trained_model = True

##########################
### Let the agent play ###
##########################
if args.p:
    if trained_model:
        ### empty logs
        logfiles = glob.glob('results/*')
        for f in logfiles:
            os.remove(f)
        profits = agent.play(bet, logRL)
    else:
        print "You need to train your model first with the -t flag before playing."

msg =  "\n***************************************************************"
msg += "\n** Thank you for using PremieRL. I hope you made some money! **\n"
msg += "***************************************************************\n"
msg += "\nDeveloped by T.Ioannidis\nMassachusetts Institute of Technology, 2017\n"
print msg




