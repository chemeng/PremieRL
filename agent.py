###########################################################
'''
    Copyright 2017, MIT

    This file is part of PremieRL.
    
'''
############################
# Written by Tim Ioannidis #
############################

# ******************************************** #
# ********* Class of playing agent  ********** #
# ******************************************** #

import numpy as np
import pandas as pd
import random, time
from collections import deque

class Agent:
    def __init__(self,model,target_model,validation_seasons):
        ### create memory object
        self.model = model
        self.target_model = target_model
        ###########################################
        ### Hyper parameters for the Double DQN ###
        ###########################################
        self.gamma = 0.90             # gamma
        self.epsilon = 1.0            # initial epsilon
        self.epsilon_decay = 0.999    # decay per episode
        self.epsilon_min = 0.05       # minimum value
        self.batch_size = 128         # how many old memories to train on per batch
        self.train_start = 256        # after how many random steps to start training
        self.max_memory = 4096        # max memory to train for
        self.validation_seasons = validation_seasons # how many seasons to use for validation
        self.init_memory()  # initialize memory

    def init_memory(self):
        self.memory = deque(maxlen=self.max_memory)

   # save sample <s,a,r,s'> to the replay memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    ### get current state
    def get_game_data(self, game):
        s = game.get_state()
        return s

    ### get action from model using epsilon-greedy policy
    def get_action(self, state, nb_actions):
        if np.random.rand() <= self.epsilon:
            return random.randrange(nb_actions),['random']*nb_actions
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0]),q_value[0]

    ### update epsilon
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def early_stopping(self,val_profits):
        ### convert to list
        profits = np.array(val_profits)
        index_max = np.argmax(profits) # get max index
        ### check if improvement over past 5 values
        if index_max == 0 :
            return True
        else:
            return False

    #################################
    ### Train using replay memory ###
    #################################
    def train_replay(self,game):
        # if very early don't train
        if len(self.memory) < self.train_start:
            return
        # pick samples randomly from replay memory (with batch_size)
        batch_size = min(self.batch_size, len(self.memory)) # maximum batch size
        mini_batch = random.sample(self.memory, batch_size) # get mini-batch randomly
        ### initialize placeholders for input/target to DDQN
        update_input = np.zeros((batch_size, game.state_size))
        update_target = np.zeros((batch_size, game.action_size))
        ### loop over batches to construct input/targets
        for i in range(batch_size):
            s, a, r, s_prime, game_over = mini_batch[i]
            target = self.model.predict(s)[0] # get q values
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if game_over:
                target[a] = r # reward the last one
            else:
                ##############################################
                #          the key point of Double DDQN       # 
                # selection of action is from model (action) #
                #  update is from target model (evaluation)  #
                ##############################################
                ### calculate argmax_a' Q(s',a'), which action maximizes Q
                action_pred = np.argmax(self.model.predict(s_prime)[0])
                ### calculate r + gamma*max_a' Q(s',a'), but Q is from target NN
                target[a] = r + self.gamma*(self.target_model.predict(s_prime)[0][action_pred])
            ### create minibatch which includes target q value and predicted q value
            update_input[i] = s
            update_target[i] = target
        ### train on the mini-batch
        self.model.fit(update_input, update_target, batch_size=batch_size, epochs=1, verbose=0)

    #############################
    #### Train agent to play ####
    #############################
    def train(self, game, logRL, episodes=1000):
        ### logger for training/test error
        self.df_profit = pd.DataFrame(columns=['Episode','Profit_Train','Stddev_Train',
            'Profit_Test','Stdev_Test','No_prof','Avg_profit','No_loss','Avg_loss'])
        profit_entries = 0
        ### set seed for debugging
        random.seed(666)
        print "Training started with available capital $"+str(game.available_capital_init)+"\n"
        ### set ANN model for D-Q-L
        model = self.model
        ### initialize validation profits
        val_profits = deque([-game.available_capital_init]*6,maxlen=6)
        ### loop over training epochs
        for episode in range(episodes):
            # reset game
            game.reset()
            # reset game over
            game_over = False
            # get initial state
            s = self.get_game_data(game)
            # initialize match counter
            match_count = 0
            # play until the end of the game
            while not game_over:
                ### get action
                a, q_value = self.get_action(s,game.nb_actions())
                ### advance game one step and update state, reward
                r = game.play(a)
                ### get new state
                s_prime = self.get_game_data(game)
                ### check game over
                game_over = game.is_over()
                ### save the sample <s, a, r, s'> to the replay memory
                self.remember(s,a,r,s_prime,game_over)
                ### update epsilon
                self.update_epsilon()
                ################################
                ### train using memory games ###
                ################################
                self.train_replay(game)
                ### set new state as current state
                s = s_prime
                ### log game
                logRL.log_match(game.season,match_count,r,a,q_value,game.profit,game.available_capital)
                match_count += 1
                if game_over:
                    ### Key idea behind double-Q: Decouple action selection and evaluation
                    ### update after each episode the target model
                    self.target_model.set_weights(self.model.get_weights())
                    ### print info
                    print("episode:", episode+1,"season:",logRL.season_labels[game.season],"profit:", "{:.1f}".format(game.profit), "matches played:",
                        str(match_count),"memory length:", len(self.memory),"epsilon:", self.epsilon)
                    ### save to log
                    logRL.log_season(episode,game.season,game.profit,match_count,self.epsilon,'train')
                        # save the model
                    if episode % 5 == 0:
                        self.model.save_weights("./model/bet_DDQN.h5")
                    ### write to csv file current seasons log
                    logRL.write_seasons('train')
                    if (episode > 0):
                        start = time.time()
                        cepsilon = self.epsilon ### backup epsilon
                        profits = self.play(game) # get profits for validation set
                        training_profits = profits[:-self.validation_seasons]
                        test_profits = profits[-self.validation_seasons:]
                        val_profits.append(np.mean(test_profits)) # append profit
                        self.epsilon = cepsilon ### restore epsilon
                        ### check for early stopping
                        #if self.early_stopping(val_profits):
                        #    print "Early stopping.."
                        #    return
                        ### append profit of training/test for monitoring
                        avg_profit = np.mean(test_profits[test_profits > 0]) if len(test_profits[test_profits > 0]) else 0
                        avg_loss = np.mean(test_profits[test_profits < 0]) if len(test_profits[test_profits < 0]) else 0
                        entry = [episode,np.mean(training_profits),np.std(training_profits),
                        np.mean(test_profits),np.std(test_profits),np.sum(test_profits > 0),
                        avg_profit,np.sum(test_profits < 0),avg_loss]
                        self.df_profit.loc[profit_entries] = entry
                        # write log
                        self.df_profit.to_csv('test_profits.csv',index=False)
                        profit_entries += 1


    ###########################
    #### Play once trained ####
    ###########################
    def play(self, game, logRL=False):
        if logRL:
            print "\nPlaying started with available capital $"+str(game.available_capital_init)+"\n"
        ### set ANN model for D-Q-L
        model = self.model
        ### use only optimal policy
        self.epsilon = 0
        ### get total seasons and initialize profits
        tot_seasons = len(game.S_all)
        profits = np.zeros((tot_seasons,1))
        ### loop over training epochs
        season_count = 0
        for season in range(tot_seasons):
            if logRL:
                print "Playing season "+str(season+1)+" of "+str(tot_seasons)
            # reset game
            game.reset(season)
            # reset game over
            game_over = False
            # get initial state
            s = self.get_game_data(game)
            # initialize match counter
            match_count = 0
            # play until the end of the game
            while not game_over:
                ### get action
                a, q_value = self.get_action(s,game.nb_actions())
                ### advance game one step and update state, reward
                r = game.play(a)
                ### get new state
                s_prime = self.get_game_data(game)
                ### check game over
                game_over = game.is_over()
                ### set new state as current state
                s = s_prime
                ### log game
                if logRL:
                    logRL.log_match(game.season,match_count,r,a,q_value,game.profit,game.available_capital)
                match_count += 1
                if game_over:
                    ### print info
                    if logRL:
                        print "Season: "+logRL.season_labels[season]+"  profit $"+"{:.2f}".format(game.profit)
                        ### save to log
                        logRL.log_season(season,game.season,game.profit,match_count,self.epsilon,'play')
                        ### write to csv file current seasons log
                        logRL.write_seasons('play')
                    ### register profit for season
                    profits[season_count] = game.profit
                    season_count += 1
        return profits
 


