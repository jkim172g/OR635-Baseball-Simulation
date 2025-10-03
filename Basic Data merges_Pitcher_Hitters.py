# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 09:33:57 2025

@author: wjmor
"""

import pandas as pd
import os
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

print(os.getcwd())  # Shows your current working directory
path = r'C:\Users\wjmor\OneDrive\Documents\MS OR\OR635_Simulations\Project'
if os.path.exists(path):
    os.chdir(path)
    print("Directory changed to:", os.getcwd())
else:
    print("Path does not exist.")
print(os.listdir('Data'))

#Rudimentary Batting Data
B_Dash = pd.read_csv(r'Data\Batter\Batter_Dash_23_25.csv')
B_Dash.head
B_BattedBall = pd.read_csv(r'Data\Batter\Batter_BattedBall_23_25.csv')
B_Discipline = pd.read_csv(r'Data\Batter\Batter_Discipline_23_25.csv')
B_Field = pd.read_csv(r'Data\Batter\Batter_Fielding_Standard.csv')

#Rudimentary Pitching Data
P_Dash = pd.read_csv(r'Data\Pitcher\Pitching_Dash_23_25.csv')
P_Discipline = pd.read_csv(r'Data\Pitcher\Pitching_Discipline_23_25.csv')
P_PitchType = pd.read_csv(r'Data\Pitcher\Pitching_PitchType_23_25.csv')


#Combining Data Frames
from functools import reduce
import pandas as pd

# Example list of DataFrames
Batter_DFs = [B_Dash, B_BattedBall,B_Discipline]

# Merge all on 'player_id'
Batter_Merge = reduce(
    lambda left, right: pd.merge(left, right, on='PlayerId', how='outer', suffixes=('', '_dup')),
    Batter_DFs
)
Batter_Merge = Batter_Merge.drop(columns=[col for col in Batter_Merge.columns if '_dup' in col])


Pitcher_DFs = [P_Dash,P_Discipline,P_PitchType]
Pitcher_Merge = reduce(
    lambda left, right: pd.merge(left, right, on='PlayerId', how='outer', suffixes=('', '_dup')),
    Pitcher_DFs
)
Pitcher_Merge = Pitcher_Merge.drop(columns=[col for col in Pitcher_Merge.columns if '_dup' in col])


#Pitch Selection Helper

P_PitchType.columns



# Step 1: Filter by PlayerId
player_id = 3137  # Replace with your target PlayerId
filtered_df = P_PitchType[P_PitchType['PlayerId'] == player_id]

# Step 2: Get the player's name
player_name = filtered_df['Name'].iloc[0].replace(" ", "_")  # Replace spaces with underscores

# Step 3: Keep only % columns + 'Name'
cols_to_keep = [col for col in filtered_df.columns if '%' in col or col == 'Name']
filtered_df = filtered_df[cols_to_keep].dropna(axis=1, how='all')

# Step 4: Store in a dictionary with dynamic name
pitch_type_dfs = {}
pitch_type_dfs[f"{player_name}_pitchTypes_df"] = filtered_df
