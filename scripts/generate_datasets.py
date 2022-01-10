import os
import numpy as np
import pandas as pd
from datetime import date 
from tqdm import tqdm
import pickle
import json
import requests
from urllib.error import HTTPError
import urllib.request
from bs4 import BeautifulSoup
from basketball_reference_web_scraper import client
import warnings
warnings.filterwarnings('ignore')

class GenerateDatasets(object):
    def __init__(self):
        self.data_path = os.path.dirname(os.getcwd()) + '/data'

    #load team to abbreviation mapping
    with open(os.path.dirname(os.getcwd()) + '/scripts' + '/team_to_abbreviations.json') as f:
        team_to_abbreviations = json.load(f)

    def extract_mvp_candidates(year):
        url = f"https://www.basketball-reference.com/awards/awards_{year}.html#mvp"
        try:
            mvp_candidate_table = pd.read_html(url)[0].droplevel(level=0, axis = 1)
            mvp_candidate_table['year'] = year
        except HTTPError as err:
            print(f'no mvp race has been found for year {year}')
        return mvp_candidate_table

    def extract_team_stats(year):
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_standings.html#all_confs_standings_E%22"

        #east
        team_east_standing_table = pd.read_html(url)[0]
        team_east_standing_table = team_east_standing_table.rename({'Eastern Conference': 'team'}, axis=1)
        #remove 'Division' in team column (e.g. Atlantic Division)
        team_east_standing_table = team_east_standing_table[team_east_standing_table['team'].str.contains('Division')==False]
        team_east_standing_table['seed'] = team_east_standing_table['W'].rank(ascending=False)

        #west
        team_west_standing_table = pd.read_html(url)[1]
        team_west_standing_table = team_west_standing_table.rename({'Western Conference': 'team'}, axis=1)
        #remove 'Division' in team column (e.g. Atlantic Division)
        team_west_standing_table = team_west_standing_table[team_west_standing_table['team'].str.contains('Division')==False]
        team_west_standing_table['seed'] = team_west_standing_table['W'].rank(ascending=False)

        #combine east and west 
        team_standing_table = pd.concat([team_east_standing_table, team_west_standing_table])
        
        #remove * in team column
        team_standing_table.team = team_standing_table.team.str.replace('*', '')
        
        #change player name string if current year (different formatting)
        if year == max_year:
            team_name_lst = []
            seeds = team_standing_table['seed']
            for seed, team in zip(list(seeds), list(team_standing_table['team'])): 
                if len(str(seed)) != 3:
                    team_name = team[:-5]
                else:
                    team_name = team[:-4]
                team_name_lst.append(team_name)
            
            team_standing_table['team'] = team_name_lst
        
        #map abbreviation to full team name
        team_standing_table['Tm'] = team_standing_table['team'].map(team_to_abbreviations)
        
        #filter only needed columns
        team_standing_table_sub = team_standing_table[['Tm', 'team', 'W', 'W/L%', 'seed']]
        
        return team_standing_table_sub

        filter_advanced = [
            'name',
            'player_efficiency_rating',
            'true_shooting_percentage',
            'three_point_attempt_rate',
            'free_throw_attempt_rate',
            'offensive_rebound_percentage',
            'defensive_rebound_percentage',
            'total_rebound_percentage',
            'assist_percentage',
            'steal_percentage',
            'block_percentage',
            'turnover_percentage',
            'usage_percentage',
            'offensive_win_shares',
            'defensive_win_shares',
            'win_shares',
            'win_shares_per_48_minutes',
            'offensive_box_plus_minus',
            'defensive_box_plus_minus',
            'box_plus_minus',
            'value_over_replacement_player'
        ]

    def extract_advanced_stats(year):
        advanced_stats_df = pd.DataFrame(client.players_advanced_season_totals(season_end_year=year))
        advanced_stats_df['year'] = year
        
        advanced_stats_df = advanced_stats_df[filter_advanced]
        advanced_stats_df = advanced_stats_df.rename(columns={'name':'Player'})
        return advanced_stats_df

    # MAIN FUNCTION TO EXTRACT HISTORICAL TABLE
    def extract_historical_table(self):
        tables = []
        print('extracting historical data of NBA MVP candidates..')
        for year in tqdm(years):
            mvp_candidate_table = extract_mvp_candidates(year)
            team_standing_table_sub = extract_team_stats(year)
            
            #left merge mvp candidate with team standings table on team abbreviation
            table = pd.merge(mvp_candidate_table, team_standing_table_sub, how='left', on='Tm')
            
            #add advanced stats
            advanced_stats_df = extract_advanced_stats(year)
            
            #left merge mvp candidate with team standings table on team abbreviation
            table = pd.merge(table, advanced_stats_df, how='left', on='Player')

            #append to list of tables
            tables.append(table)
        master_table = pd.concat(tables)
        #drop rows of players who were traded mid season
        master_table = master_table[master_table['Tm'].str.contains('TOT')==False]

        #fill na in 3P%
        master_table['3P%'] = master_table['3P%'].fillna(0)
        print('complete')

        #save to data folder
        full_path_historic = self.data_path + '/master_table.csv'
        print(f"Historical MVP candidate master table has been saved to {full_path_historic}")

    ## 2022 MVP Candidate forecasting
    def load_current_mvp_candidates(self):
        year = 2022
        basic_stats_df = pd.DataFrame(client.players_season_totals(season_end_year=year))
        advanced_stats_df = pd.DataFrame(client.players_advanced_season_totals(season_end_year=year))
        advanced_stats_df = advanced_stats_df[filter_advanced]
        advanced_stats_df = advanced_stats_df.rename(columns={'name':'Player'})
        return basic_stats_df, advanced_stats_df

    def filter_basic_stats(self): 
        filter_basic = [
            'name',
            'games_played',
            'team',
            'points',
            'assists',
            'offensive_rebounds',
            'defensive_rebounds',
            'steals',
            'blocks',
            'made_field_goals',
            'attempted_field_goals',
            'made_three_point_field_goals',
            'attempted_three_point_field_goals'
        ]
        df = df[filter_basic]
        df = df.rename(columns={'name':'Player',
                                    'points':'PTS',
                                    'assists':'AST',
                                    'steals':'STL',
                                    'blocks':'BLK',})

        df['PTS'] = df['PTS'] / df['games_played']
        df['AST'] = df['AST'] / df['games_played']
        df['STL'] = df['STL'] / df['games_played']
        df['BLK'] = df['BLK'] / df['games_played']
        df['FG%'] = df['made_field_goals'] / df['attempted_field_goals']
        df['3P%'] = df['made_three_point_field_goals'] / df['attempted_three_point_field_goals']
        df['TRB'] = (df['offensive_rebounds'] + df['defensive_rebounds']) / df['games_played']

        df = df[['Player', 'games_played', 'team', 'PTS', 'AST', 'STL', 'BLK', 'FG%' ,'3P%' , 'TRB']]
        df['team'] = df['team'].astype(str).str.slice(5,)
        df['team'] = df['team'].astype(str).str.replace('_', ' ').str.lower()
        return df
    
    def find_mvp_candidate_names(self, url):
        #2022 candidate table
        url = url
        html = requests.get(url).content

        soup = BeautifulSoup(html)
        remove_line = 'Last week’s ranking'

        top_five = []
        next_five = [] 

        for line in soup.find_all("h3")[1:-1]:
            if remove_line not in str(line):
                name_raw = str(line).split(',')[0]
                name_raw = name_raw.split('.')[1]
                name = name_raw[1:]
                top_five.append(name)

        for line in soup.find_all("p"):
            if 'week: ' in str(line):
                name_raw = str(line).split(',')[0]
                name_raw = name_raw.split('.')[1]
                name = name_raw[1:]
                name = name.split('>')[1][1:]
                next_five.append(name)    
        top_ten = top_five + next_five
        return top_ten

    def adjust_vorp(self, df):
        # adjust VORP at the current pace and project to rest of the season
        df_sub = df[['Player', 'games_played', 'value_over_replacement_player']]
        df_sub['games'] = 82
        df_sub['games_left'] = (df_sub['games'] - df_sub['games_played'])
        df_sub['vorp/games_played'] = (df_sub['value_over_replacement_player'] / df_sub['games_played'])
        df_sub['adjusted_vorp'] = (df_sub['vorp/games_played'] * df_sub['games_left']) + df_sub['value_over_replacement_player']
        df['value_over_replacement_player'] = df_sub['adjusted_vorp']
        return df

    # MAIN FUNCTION TO EXTRACT CURRENT SEASON MVP CANDIDATE TABLE
    def extract_current_mvp_candidates(self):
        basic_stats_df, advanced_stats_df = load_current_mvp_candidates()
        basic_stats_df = filter_basic_stats(df=basic_stats_df)

        #left merge mvp candidate with team standings table on team abbreviation
        team_standing_table_sub = extract_team_stats(2022)
        team_standing_table_sub['team'] = team_standing_table_sub['team'].str.lower()
        joined_table_2022 = pd.merge(basic_stats_df, team_standing_table_sub, how='left', on='team')
        joined_table_2022 = pd.merge(joined_table_2022, advanced_stats_df, how='left', on='Player')
        #fix character in name Jokić'
        joined_table_2022.loc[joined_table_2022['Player'] == 'Nikola Jokić', 'Player'] = 'Nikola Jokic'

        top_ten = find_mvp_candidate_names(url= 'https://www.nba.com/news/kia-mvp-ladder-jan-7-2022-edition')

        joined_table_2022 = joined_table_2022[joined_table_2022['Player'].isin(top_ten)]
        joined_table_2022 = adjust_vorp(df=joined_table_2022)

        #save to data folder
        full_path_current_candidates = self.data_path + '/data_2022.csv'
        print(f"Current season's MVP candidate table has been saved to {full_path_current_candidates}")
