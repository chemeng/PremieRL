###########################################################
'''
    Copyright 2017, MIT

    This file is part of PremieRL.
    
'''
############################
# Written by Tim Ioannidis #
############################

# **************************************************** #
# ********* Class of logger for statistics  ********** #
# **************************************************** #
import pandas as pd

class Logger:
    def __init__(self,I,seasons):
        self.seasons = pd.DataFrame(columns=['Episode','Season','Profit','Matches','Epsilon'])
        self.season_labels = seasons
        self.info = I
        self.reset_matches()

    ### reset matches dataframe
    def reset_matches(self):
        mcols = ['season','match_id','date','home_team','away_team','result','action','predicted','game_profit',
        'total_profit','available_capital','odds_home','odds_tie','odds_away','q_1','q_X','q_2','q_0']
        self.matches = pd.DataFrame(columns=mcols)

    ####################
    ### single match ###
    ####################
    def log_match(self,season_id,match_count,r,a,q_values,profit,capital):
        avail_actions = ['1','X','2','0']
        # grab info from season and game
        game_info = self.info[season_id].iloc[match_count,:]
        predicted = 'YES' if game_info.result==avail_actions[a] else 'NO'
        predicted = 'NA' if avail_actions[a]=='0' else predicted
        ### log info
        entry = [game_info.season,str(game_info.match_id),game_info.date,game_info.home_team,game_info.away_team,
                game_info.result,avail_actions[a],predicted,"{:.1f}".format(r),"{:.1f}".format(profit),
                "{:.1f}".format(capital),"{:.2f}".format(game_info.home_odds),"{:.2f}".format(game_info.tie_odds)
                ,"{:.2f}".format(game_info.away_odds)]+[str(q) for q in q_values]
        self.matches.loc[match_count] = entry

    ####################### 
    ### log season/game ###
    #######################
    def log_season(self, episode, season_no, money,matches,epsilon,flag):
        ### get matches into seasons dataframe
        self.seasons.loc[episode] = [str(episode+1), self.season_labels[season_no], money,str(matches),epsilon]
        ### write matches
        season_name = self.season_labels[season_no].replace('/20','')
        if 'train' in flag:
            fname = 'train_data/ep'+ str(episode+1).zfill(3)+'-'+season_name+'.csv'
        elif 'play' in flag or 'test' in flag:
            fname = 'results/season'+ str(episode+1).zfill(2)+'-'+season_name+'.csv'
        fname = fname.replace('-20','-')
        ### increase index to start at 1
        self.matches.index += 1
        self.matches.to_csv(fname,index=False)
        ### reset matches
        self.reset_matches()

    #################
    ### write log ###
    #################
    def write_seasons(self,flag):
        if 'train' in flag:
            self.seasons.to_csv('train_data/training_seasons.csv',index=False)
        elif 'play' in flag or 'test' in flag:
            self.seasons.to_csv('results/playing_seasons.csv',index=False)


