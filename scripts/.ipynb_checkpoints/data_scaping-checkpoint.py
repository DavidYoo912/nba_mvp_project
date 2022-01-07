import os
import pandas as pd
from datetime import date 
from tqdm import tqdm
import pickle

import requests
from urllib.error import HTTPError
import urllib.request
from bs4 import BeautifulSoup

import warnings
warnings.filterwarnings('ignore')

max_year = date.today().year + 1
years = [year for year in range(1980, max_year)]

#load team name to abbreviations json
with open(os.path.dirname(os.getcwd()) + '/scripts' + '/team_to_abbreviations.json') as f:
      team_to_abbreviations = json.load(f)

def extract_mvp_candidates(year):
    url = f"https://www.basketball-reference.com/awards/awards_{year}.html#mvp"
    try:
        mvp_candidate_table = pd.read_html(url)[0].droplevel(level=0, axis = 1)
        mvp_candidate_table['year'] = year
        #mvp_candidate_history.append(mvp_candidate_table)
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
    
    #map abbreviation to full team name
    team_standing_table['Tm'] = team_standing_table['team'].map(team_to_abbreviations)
    
    #filter only needed columns
    team_standing_table_sub = team_standing_table[['Tm', 'team', 'W', 'W/L%', 'seed']]

    return team_standing_table_sub

tables = []
print('extracting raw data..')
for year in tqdm(years):
    mvp_candidate_table = extract_mvp_candidates(year)
    team_standing_table_sub = extract_team_stats(year)
    
    #left merge mvp candidate with team standings table on team abbreviation
    table = pd.merge(mvp_candidate_table, team_standing_table_sub, how='left', on='Tm')
    
    #append to list of tables
    tables.append(table)
print('complete')

data_path = os.path.dirname(os.getcwd()) + '/data' + '/master_table.csv'
master_table.to_csv(data_path, index=False)