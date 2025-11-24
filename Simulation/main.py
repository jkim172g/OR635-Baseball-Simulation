from SeasonSimulator import SeasonSimulator, SeasonStats
from baseball_sim_classes import Game, Team, Batter, Pitcher, read_data, normalize_values, perturb_values
import numpy as np
import pandas as pd

# Read data
batter_df, fb_disc_df, ch_disc_df, cu_disc_df, sl_disc_df, pitcher_df, hit_traj_df = read_data()

# Forward-fill team names (replace "- - -" with the team name from above)
batter_df['Team'] = batter_df['Team'].replace('- - -', pd.NA).ffill()
pitcher_df['Team'] = pitcher_df['Team'].replace('- - -', pd.NA).ffill()

# Drop any rows that still have NA team names (first row issues)
batter_df = batter_df.dropna(subset=['Team'])
pitcher_df = pitcher_df.dropna(subset=['Team'])

# Consolidate team name variants (ATH is likely Athletics/OAK)
team_name_map = {
    'ATH': 'OAK',  # Athletics
    # Add other mappings if needed
}

batter_df['Team'] = batter_df['Team'].replace(team_name_map)
pitcher_df['Team'] = pitcher_df['Team'].replace(team_name_map)

# Organize players by team
teams = {}

fb_names = set(fb_disc_df.index)
cu_names = set(cu_disc_df.index)
sl_names = set(sl_disc_df.index)
ch_names = set(ch_disc_df.index)

for idx, row in batter_df.iterrows():
    team_name = row["Team"]
    if team_name not in teams:
        teams[team_name] = {"batters": [], "pitchers": []}
    
    # Create batter data dictionary
    batter_data = {
        'name': row["Name"],
        'id': row["PlayerId"],
        'team': row["Team"],
        'zone_prob': {
            "na": row["Zone%"],
            "fastball": fb_disc_df.loc[row["Name"], "Zone%"] if row["Name"] in fb_names else None, 
            "curveball": cu_disc_df.loc[row["Name"], "Zone%"] if row["Name"] in cu_names else None,
            "slider": sl_disc_df.loc[row["Name"], "Zone%"] if row["Name"] in sl_names else None,
            "changeup": ch_disc_df.loc[row["Name"], "Zone%"] if row["Name"] in ch_names else None 
        },
        'swing_prob': {
            "na": {'strike': row["Z-Swing%"], 'ball': row["O-Swing%"]},
            "fastball": {'strike': fb_disc_df.loc[row["Name"], "Z-Swing%"] if row["Name"] in fb_names else None,
                         'ball': fb_disc_df.loc[row["Name"], "O-Swing%"] if row["Name"] in fb_names else None},
            "curveball": {'strike': cu_disc_df.loc[row["Name"], "Z-Swing%"] if row["Name"] in cu_names else None,
                         'ball': cu_disc_df.loc[row["Name"], "O-Swing%"] if row["Name"] in cu_names else None},
            "slider": {'strike': sl_disc_df.loc[row["Name"], "Z-Swing%"] if row["Name"] in sl_names else None,
                         'ball': sl_disc_df.loc[row["Name"], "O-Swing%"] if row["Name"] in sl_names else None},
            "changeup": {'strike': ch_disc_df.loc[row["Name"], "Z-Swing%"] if row["Name"] in ch_names else None,
                         'ball': ch_disc_df.loc[row["Name"], "O-Swing%"] if row["Name"] in ch_names else None},
        },
        'contact_prob': {
            "na": {'strike': row["Z-Contact%"], 'ball': row["O-Contact%"]},
            "fastball": {'strike': fb_disc_df.loc[row["Name"], "Z-Contact%"] if row["Name"] in fb_names else None,
                         'ball': fb_disc_df.loc[row["Name"], "O-Contact%"] if row["Name"] in fb_names else None},
            "curveball": {'strike': cu_disc_df.loc[row["Name"], "Z-Contact%"] if row["Name"] in cu_names else None,
                         'ball': cu_disc_df.loc[row["Name"], "O-Contact%"] if row["Name"] in cu_names else None},
            "slider": {'strike': sl_disc_df.loc[row["Name"], "Z-Contact%"] if row["Name"] in sl_names else None,
                         'ball': sl_disc_df.loc[row["Name"], "O-Contact%"] if row["Name"] in sl_names else None},
            "changeup": {'strike': ch_disc_df.loc[row["Name"], "Z-Contact%"] if row["Name"] in ch_names else None,
                         'ball': ch_disc_df.loc[row["Name"], "O-Contact%"] if row["Name"] in ch_names else None},
        },
        'foul_prob': {'strike': 0.22, 'ball': 0.22},
        'contact_cat_prob': {
            "ground_ball": row["GB%"],
            "line_drive": row["LD%"],
            "fly_ball": row["FB%"],
        },
        'outcome_prob': {
            "ground_ball": {
                "single": hit_traj_df.loc["Ground Balls", "1B%"],
                "double": hit_traj_df.loc["Ground Balls", "2B%"],
                "triple": hit_traj_df.loc["Ground Balls", "3B%"],
                "home_run": hit_traj_df.loc["Ground Balls", "HR%"],
                "out": hit_traj_df.loc["Ground Balls", "Out%"]
            },
            "line_drive": {
                "single": hit_traj_df.loc["Line Drives", "1B%"],
                "double": hit_traj_df.loc["Line Drives", "2B%"],
                "triple": hit_traj_df.loc["Line Drives", "3B%"],
                "home_run": hit_traj_df.loc["Line Drives", "HR%"],
                "out": hit_traj_df.loc["Line Drives", "Out%"]
            },
            "fly_ball": {
                "single": hit_traj_df.loc["Fly Balls", "1B%"],
                "double": hit_traj_df.loc["Fly Balls", "2B%"],
                "triple": hit_traj_df.loc["Fly Balls", "3B%"],
                "home_run": row["HR/FB"],
                "out": hit_traj_df.loc["Fly Balls", "Out%"]
            },
        },
        'outcome_power_prob': {
            "soft": row["Soft%"],
            "medium": row["Med%"],
            "hard": row["Hard%"]
        },
    }
    
    teams[team_name]["batters"].append(Batter(batter_data))

base_velocity_dist = {95: 0.5, 100: 0.5}
base_movement_prob = {'straight': 0.25, 'down': 0.25, 'side': 0.25, 'fade': 0.25}

for idx, row in pitcher_df.iterrows():
    team_name = row["Team"]
    if team_name not in teams:
        teams[team_name] = {"batters": [], "pitchers": []}
    
    pitcher_data = {
        'name': row["Name"],
        'id': row["PlayerId"],
        'team': row["Team"],
        'pitch_type_prob': {
            'fastball': row["FA%"],
            'curveball': row["CU%"],
            'slider': row["SL%"],
            'changeup': row["CH%"]
        },
        'swing_prob': {'strike': row["Z-Swing%"], 'ball': row["O-Swing%"]},
        'contact_prob': {'strike': row["Z-Contact%"], 'ball': row["O-Contact%"]},
        'contact_cat_prob': {
            "ground_ball": row["GB%"],
            "line_drive": row["LD%"],
            "fly_ball": row["FB%"],
        },
        'outcome_power_prob': {
            "soft": row["Soft%"],
            "medium": row["Med%"],
            "hard": row["Hard%"]
        },
        'velocity_dist': perturb_values(base_velocity_dist, 0.05),
        'movement_prob': perturb_values(base_movement_prob, 0.05),
        'strike_prob': row["Zone%"],
        'starter': False
    }
    
    teams[team_name]["pitchers"].append(Pitcher(pitcher_data))

# Mark first pitcher as starter for each team
for team_name in teams:
    if teams[team_name]["pitchers"]:
        teams[team_name]["pitchers"][0].starter = True

# Filter out teams with insufficient rosters
teams_to_remove = []
for team_name in teams:
    if len(teams[team_name]["batters"]) < 9 or len(teams[team_name]["pitchers"]) < 5:
        teams_to_remove.append(team_name)
        print(f"Removing {team_name}: insufficient roster ({len(teams[team_name]['batters'])} batters, {len(teams[team_name]['pitchers'])} pitchers)")

for team_name in teams_to_remove:
    del teams[team_name]

print(f"\nFound {len(teams)} valid teams:")
for team_name in sorted(teams.keys()):
    print(f"  {team_name}: {len(teams[team_name]['batters'])} batters, {len(teams[team_name]['pitchers'])} pitchers")

# Create team objects
team_objects = []
for tname, roster in teams.items():
    team_objects.append(Team(roster["batters"], roster["pitchers"]))
    team_objects[-1].team_name = tname  # Add team name attribute

print(f"\nCreated {len(team_objects)} team objects")

# Run season simulation
sim = SeasonSimulator(team_objects, Game)
print("\nStarting season simulation...")
sim.run_season(games_per_matchup=1)  # Start with 1 game per matchup for testing
print("\nExporting statistics...")
sim.export_stats("season_stats")
print("\nSeason simulation complete!")