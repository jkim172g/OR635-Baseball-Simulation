## Project for GMU OR 635: Discrete System Simulation
## Authors: Jacob Kim, Noah Mecikalski, Bennett Miller, W.J. Moretz

import pandas as pd
import numpy as np
import pdb

np.random.seed(17) # TODO need to set any random streams/substreams?

# TODO do we want to track stats of what happen in the Batters and Pitchers themselves as well?

class Batter:
    
    def __init__(self, batter_data):
        
        #Save basic information about batter
        self.name = batter_data['name']
        self.id = batter_data['id']
        self.team = batter_data['team']
        
        #Save this batter's probability of swinging, making contact, and result
        self.zone_prob = batter_data['zone_prob']
        self.swing_prob = batter_data['swing_prob']
        self.contact_prob = batter_data['contact_prob']
        self.contact_cat_prob = batter_data['contact_cat_prob']
        self.outcome_prob = batter_data['outcome_prob']
        self.outcome_power_prob = batter_data['outcome_power_prob']
        self.foul_prob = batter_data['foul_prob']
        # self.int_walk_prob = batter_data['int_walk_prob']
        # self.hit_by_pitch_prob = batter_data['hit_by_pitch_prob']
        # self.sac_fly_prob = batter_data['sac_fly_prob']
        # self.sac_bunt_prob = batter_data['sac_bunt_prob']             
        
    def __str__(self):
        return f"Batter: {self.name}, {self.team}"
        

class Pitcher:
    
    def __init__(self, pitcher_data):
        
        #Save basic information about pitcher
        self.name = pitcher_data['name']
        self.id = pitcher_data['id']
        self.team = pitcher_data['team']
        
        #Save statistics about pitching probabilities
        self.pitch_type_prob = pitcher_data['pitch_type_prob']
        self.swing_prob = pitcher_data['swing_prob']
        self.contact_prob = pitcher_data['contact_prob']
        self.contact_cat_prob = pitcher_data['contact_cat_prob']
        self.outcome_power_prob = pitcher_data['outcome_power_prob']
        self.velocity_dist = pitcher_data['velocity_dist']
        self.movement_prob = pitcher_data['movement_prob']
        self.strike_prob = pitcher_data['strike_prob']
        self.num_pitch = 0
        self.starter = pitcher_data['starter']
        
        def __str__(self):
            return f"Pitcher: {self.name}, {self.team}"
        

class Team:
    
    def __init__(self, batters, pitchers):
        
        self.batters = batters
        self.pitchers = pitchers
        self.score = 0
        self.batter_index = 0
        self.num_batters = len(batters)
        self.pitcher_index = 0
        self.num_pitchers = len(pitchers)
    
    def __str__(self):
        return "Team"


class Game:
    
    def __init__(self, team1, team2):
        
        #Init game stats
        self.inning = 1
        self.inning_half = 'top'
        self.outs = 0
        self.strikes = 0
        self.balls = 0
        self.bases = [0,0,0]
        self.baserunners = [0,0,0] # TODO something other than 0 when empty? Dict instead with bases named?
        self.team1_batter_index = 0
        self.team2_batter_index = 0
        self.team1_pitcher_index = 0
        self.team2_pitcher_index = 0
        
        self.team1 = team1
        self.team2 = team2
        
        self.event_log = {'Inning':[self.inning], 'Inning Half':[self.inning_half],'Event':['Init'],
                          'Pitch Outcome':['Init'], 'Batter':['NA'], 'Batter Number':['NA'], 'Bases':[self.bases],
                          'Baserunners':[self.baserunners], 'Balls':[self.balls], 'Strikes':[self.strikes],
                          'Outs':[self.outs],'Team 1 Score':[0], 'Team 2 Score':[0], 'Pitcher':['NA'],
                          }
    
    def pitch(self, batter, pitcher):
        # Get pitch details

        pitcher.num_pitch += 1
        pitch_type = np.random.choice(list(pitcher.pitch_type_prob.keys()), p=normalize_values(pitcher.pitch_type_prob.values(), 1))
        pitch_velocity = np.random.choice(list(pitcher.velocity_dist.keys()), p=normalize_values(pitcher.velocity_dist.values(), 1))
        pitch_movement = np.random.choice(list(pitcher.movement_prob.keys()), p=normalize_values(pitcher.movement_prob.values(), 1))

        
        #Calculate result
        strike_prob = get_aligned_value(pitcher.strike_prob, batter.zone_prob) # Uses Zone% for each
        strike = True if np.random.uniform() < strike_prob else False
        if strike:
            pitch_result = 'strike'
            #If strike, check if batter swings
            #Adjust swing prob by half if 3 balls and less than two strikes
            # Uses Z-Swing%
            adj_batter_swing_prob = batter.swing_prob['strike']/2 if self.balls == 3 and self.strikes < 2 else batter.swing_prob['strike']
            swing_prob = get_aligned_value(pitcher.swing_prob['strike'], adj_batter_swing_prob)
            swing = True if np.random.uniform() < swing_prob else False
            if swing:
                #If batter swings, check if contact was made
                # Uses Z-Contact%
                contact_prob = get_aligned_value(pitcher.contact_prob['strike'], batter.contact_prob['strike'])
                contact = True if np.random.uniform() < contact_prob else False
                #If contact was made, calculate result
                if contact:
                    # If contact was made, check if foul
                    foul = True if np.random.uniform() < batter.foul_prob['strike'] else False
                    if foul:
                        result = 'foul'
                    else:
                        contact_cat_prob = get_aligned_value(
                            normalize_values(list(pitcher.contact_cat_prob.values()), 1),
                            normalize_values(list(batter.contact_cat_prob.values()), 1)
                            )
                        contact_cat = np.random.choice(list(batter.contact_cat_prob.keys()), p=normalize_values(contact_cat_prob, 1))
                        outcome_prob = batter.outcome_prob[contact_cat]                        
                        result = np.random.choice(list(outcome_prob.keys()), p=normalize_values(list(outcome_prob.values()), 1))
                else:
                    #If swung and missed, strike 
                    result = 'strike'
            else:
                #If strike not swung at, strike
                result = 'strike'
        else:
            #If ball was thrown, check if it was swung at
            # Uses O-Swing%
            adj_batter_swing_prob = batter.swing_prob['ball']/2 if self.balls == 3 and self.strikes < 2 else batter.swing_prob['ball']
            swing_prob = get_aligned_value(pitcher.swing_prob['ball'], adj_batter_swing_prob)
            swing = True if np.random.uniform() < swing_prob else False
            if swing:
                # Pitch now considered a strike regardless of outcome
                pitch_result= 'strike'
                #If batter swings, check if contact was made
                # Uses O-Contact%
                contact_prob = get_aligned_value(pitcher.contact_prob['ball'], batter.contact_prob['ball'])
                contact = True if np.random.uniform() < contact_prob else False
                #If contact was made, calculate result
                if contact:
                    # If contact, check if foul
                    foul = True if np.random.uniform() < batter.foul_prob['ball'] else False
                    if foul:
                        result = 'foul'
                    else:
                        contact_cat_prob = get_aligned_value(
                            normalize_values(list(pitcher.contact_cat_prob.values()), 1),
                            normalize_values(list(batter.contact_cat_prob.values()), 1)
                            )
                        contact_cat = np.random.choice(list(batter.contact_cat_prob.keys()), p=normalize_values(contact_cat_prob, 1))
                        outcome_prob = batter.outcome_prob[contact_cat]                        
                        result = np.random.choice(list(outcome_prob.keys()), p=normalize_values(list(outcome_prob.values()), 1))
                else:
                    #If swung and missed, strike 
                    result = 'strike'
            else:
                pitch_result= 'ball'
                #if ball not swung at, result is a ball
                result = 'ball'
               
        
        return result, pitch_result
        

    def move_bases(self, action, batting_team, current_batter):
        #Put logic here for moving bases based on the action
     
        ## How do we determine whether a player already on base was out?
        #pdb.set_trace()
        if action == 'home_run':
            batting_team.score += sum(self.bases) + 1
            self.bases = [0,0,0]
            self.baserunners = [0,0,0]
        elif action == 'walk':
            # First is empty, fill, no other advancing
            if self.bases[0] == 0:
                self.bases[0] = 1
                self.baserunners[0] = current_batter.name
            # Now always someone on first
            # Bases loaded, advance all and score
            elif self.bases == [1,1,1]:
                batting_team.score += 1
                self.baserunners[2] = self.baserunners[1]
                self.baserunners[1] = self.baserunners[0]
                self.baserunners[0] = current_batter.name
            # Only First and Second, advance runners and fill First, no score
            elif self.bases[0] == 1 and self.bases[1] == 1:
                self.bases[2] = 1 # now a runner on Third
                self.baserunners[2] = self.baserunners[1]
                self.baserunners[1] = self.baserunners[0]
                self.baserunners[0] = current_batter.name
            # First and Third or just First, advance to Second and fill First, Third stays if there
            else:
                self.bases[1] = 1
                self.baserunners[1] = self.baserunners[0]
                self.baserunners[0] = current_batter.name
        elif action == 'single': # TODO update baserunning logic later
            # Batter goes to First, any baserunners advance, score if on Third
            batting_team.score += self.bases[2]
            self.bases[2] = self.bases[1]
            self.bases[1] = self.bases[0]
            self.bases[0] = 1
            self.baserunners[2] = self.baserunners[1]
            self.baserunners[1] = self.baserunners[0]
            self.baserunners[0] = current_batter.name
        elif action == 'double': # TODO update baserunning logic later
            # Batter goes to second, any baserunners advance two bases (for now), score if on Second or Third
            batting_team.score += self.bases[2] + self.bases[1]
            self.bases[2] = self.bases[0]
            self.bases[1] = 1
            self.bases[0] = 0
            self.baserunners[2] = self.baserunners[0]
            self.baserunners[1] = current_batter.name
            self.baserunners[0] = 0        
        elif action == 'triple':
            # All runners clear bases, score, batter goes to Third
            batting_team.score += sum(self.bases)
            self.bases = [0,0,1]
            self.baserunners[2] = current_batter.name
            self.baserunners[1] = 0
            self.baserunners[0] = 0
        else:
            # TODO add tag-up logic
            self.outs += 1
    
    def determine_pitch_change(self, pitching_team, batting_team, pitcher, prev_pitches, prev_batting_score, starter):
        replace = False
        if starter:
            
            if pitcher.num_pitch > np.random.uniform(105,5)-3:
                if not (self.inning >= 9 and pitching_team.score > batting_team.score):
                    replace = True
            elif prev_pitches - pitcher.num_pitch >= 35:
                replace = True
            elif self.inning < 4 and batting_team.score - prev_batting_score > 5:
                replace = True
            elif 4 <= self.inning <= 6 and batting_team.score - prev_batting_score > 3:
                replace = True
            elif self.inning > 6 and batting_team.score - prev_batting_score > 2:
                replace = True
        else:
            if pitcher.num_pitch > np.random.uniform(25,5)-3:
                if self.outs < 2:
                    replace = True
            elif prev_pitches - pitcher.num_pitch >= 35:
                replace = True
            elif batting_team.score - prev_batting_score > 3:
                replace = True
            elif pitching_team.score - batting_team.score < 3 and batting_team.score - prev_batting_score > 2:
                replace = True
            elif 4 <= self.inning <= 6 and batting_team.score - prev_batting_score > 2:
                replace = True
            elif self.inning > 6 and batting_team.score - prev_batting_score > 1:
                replace = True
                    
        return replace

    def simulate_inning_half(self, batting_team, pitching_team):
        end_game = False
        prev_pitches =  [pitching_team.pitchers[pitching_team.pitcher_index].num_pitch][0]
        prev_batting_score = [batting_team.score][0]
        while self.outs < 3 and not end_game:
            #Set batter and pitcher
            self.strikes = 0
            self.balls = 0
            current_batter = batting_team.batters[batting_team.batter_index]
            current_pitcher = pitching_team.pitchers[pitching_team.pitcher_index]
            #Run through a pitch and update outcomes
            hit = False
            while self.strikes < 3 and self.balls < 4 and not hit and not end_game:
                #Get result of pitch
                event, pitch_result = self.pitch(current_batter, current_pitcher)
                #Update stats
                if event == 'strike':
                        self.strikes += 1
                        #Record an out if strike out
                        if self.strikes == 3:
                            self.outs += 1
                elif event == 'ball':
                        self.balls += 1
                        #Record a walk if ball 4
                        if self.balls == 4:
                            self.move_bases('walk', batting_team, current_batter)
                elif event == 'foul':
                    #Record a strike if foul and strikes < 2
                    if self.strikes < 2:
                        self.strikes += 1
                else:
                    #Move bases if a hit
                    self.move_bases(event, batting_team, current_batter)
                    hit = True
                #After each pitch, update current game state
                
                self.update_event_log(current_batter, batting_team.batter_index, current_pitcher, event, pitch_result)
                
                #If home team scored in the final inning to go ahead, end game
                if self.inning >= 9 and self.inning_half == 'bottom' and batting_team.score > pitching_team.score:
                    end_game = True
            #Update batter
            if batting_team.batter_index < batting_team.num_batters - 1:
                batting_team.batter_index += 1
            else:
                batting_team.batter_index = 0
            #Update pitcher
            result = self.determine_pitch_change(pitching_team, batting_team, current_pitcher, prev_pitches, 
                                                 prev_batting_score, current_pitcher.starter)
            if result:
                if pitching_team.pitcher_index < pitching_team.num_pitchers - 1:
                    pitching_team.batter_index += 1
                else:
                    #TODO: We shouldn't cycle back to the first pitcher, but this is to avoid an error
                    pitching_team.batter_index = 0
                #Reset previous number of pitches and previous score
                prev_pitches =  [pitching_team.pitchers[pitching_team.pitcher_index].num_pitch][0]
                prev_batting_score = [batting_team.score][0]
        
    def update_event_log(self, current_batter, batter_index, current_pitcher, event, pitch_result):
        #Add all current information to the event log
        self.event_log['Inning'].append(self.inning)
        self.event_log['Inning Half'].append(self.inning_half)
        self.event_log['Outs'].append(self.outs)
        self.event_log['Strikes'].append(self.strikes)
        self.event_log['Balls'].append(self.balls)
        self.event_log['Bases'].append(self.bases[:])
        self.event_log['Baserunners'].append(self.baserunners[:])
        self.event_log['Team 1 Score'].append(self.team1.score)
        self.event_log['Team 2 Score'].append(self.team2.score)
        self.event_log['Batter'].append(current_batter.name)
        self.event_log['Batter Number'].append(batter_index + 1) # TODO maybe replace with an added current_batter.lineup_spot?
        self.event_log['Pitcher'].append(current_pitcher.name)
        self.event_log['Pitch Outcome'].append(pitch_result)
        if self.strikes == 3:
            self.event_log['Event'].append('strikeout')
        elif self.balls == 4:
            self.event_log['Event'].append('walk')
        else:
            self.event_log['Event'].append(event)
            
    
    def play_ball(self):
        #pdb.set_trace()
        #Loop through innings
        while self.inning <= 9 or (self.inning >=9 and team1.score == team2.score):
            self.outs = 0
            self.strikes = 0
            self.balls = 0
            self.bases = [0,0,0]
            self.baserunners = [0,0,0]
            if self.inning_half == 'top':
                #If top of the inning, team1 is batting and team2 is pitching
                self.simulate_inning_half(team1, team2)    
                self.inning_half = 'bottom'
            else:
                #If bottom of the inning, team2 is batting and team1 is pitching
                if self.inning == 9 and team2.score > team1.score:
                    self.inning += 1
                    pass
                else:
                    self.simulate_inning_half(team2, team1) 
                    self.inning_half = 'top'
                    self.inning += 1
            
        return self.event_log
    

def perturb_values(orig_val_dict, range):
    # Perturb each value by up to 5% of its value
    # Note this will never return any values below 0, but can above 1,
        # thus normalization (outside the method) will return valid probabilities
    val_dict = orig_val_dict.copy()
    # perturb
    for k, v in val_dict.items():
        perturbed_val = np.random.uniform(v * (1 - range), v * (1 + range))
        val_dict[k] = perturbed_val

    # normalize back to 1
    denom = sum(val_dict.values())
    norm = 1/denom
    for k, v in val_dict.items():
        val_dict[k] = v*norm

    return val_dict


def make_box_score(event_log):
    batting_stats = {"Team1": {}, "Team2": {}}
    pitching_stats = {"Team1": {}, "Team2": {}}

    for _, row in event_log.iterrows():
        result = row["Event"]
        if result in ["Init"]:  # ignore non-play events
            continue
        
        pitch_outcome = row["Pitch Outcome"]  # "ball" or "strike"

        is_strike = True if pitch_outcome == "strike" else False
    
        if row["Inning Half"] == "top":
            # Team 1 is batting, Team 2 pitching
            update_batter(batting_stats["Team1"], row["Batter"], result)
            update_pitcher(pitching_stats["Team2"], row["Pitcher"], result, pitch_strike = is_strike)
        elif row["Inning Half"] == "bottom":
            # Team 2 is batting, Team 1 pitching
            update_batter(batting_stats["Team2"], row["Batter"], result)
            update_pitcher(pitching_stats["Team1"], row["Pitcher"], result, pitch_strike = is_strike)
    
    
    team1_bat = make_batting_box(batting_stats["Team1"])
    team2_bat = make_batting_box(batting_stats["Team2"])
    
    team1_bat["OBP"] = (team1_bat["H"] + team1_bat["BB"]) / team1_bat["PA"]
    team1_bat["SLG"] = team1_bat["TB"] / team1_bat["AB"].replace(0, pd.NA)
    
    team2_bat["OBP"] = (team2_bat["H"] + team2_bat["BB"]) / team2_bat["PA"]
    team2_bat["SLG"] = team2_bat["TB"] / team2_bat["AB"].replace(0, pd.NA)
    
    team1_bat = team1_bat.round({"OBP": 3, "SLG": 3})
    team1_bat = team1_bat.fillna(0)
    
    team2_bat = team2_bat.round({"OBP": 3, "SLG": 3})
    team2_bat = team2_bat.fillna(0)
    
    team1_pit = make_pitching_box(pitching_stats["Team1"])
    team2_pit = make_pitching_box(pitching_stats["Team2"])
    
    print("Team 1 Batting:")
    print(team1_bat)
    print("\nTeam 2 Batting:")
    print(team2_bat)
    print("\nTeam 1 Pitching:")
    print(team1_pit)
    print("\nTeam 2 Pitching:")
    print(team2_pit)

    with pd.ExcelWriter("box_score.xlsx", engine="openpyxl") as writer:
        team1_bat.to_excel(writer, sheet_name="Team1_Batting", index=False)
        team2_bat.to_excel(writer, sheet_name="Team2_Batting", index=False)
        team1_pit.to_excel(writer, sheet_name="Team1_Pitching", index=False)
        team2_pit.to_excel(writer, sheet_name="Team2_Pitching", index=False)


def update_batter(stats, player, result):
        
    if player not in stats:
        stats[player] = {"AB": 0, "H": 0, "HR": 0, "BB": 0, "K": 0, "TB": 0, "PA": 0}
        #at bats, hits, base on balls, strikeouts, total bases, plate appearances
    
    stats[player]["PA"] += 1  # every result counts as a plate appearance

    if result in ["single", "double", "triple", "home_run"]:
        stats[player]["AB"] += 1
        stats[player]["H"] += 1
        # assign total bases
        bases = {"single": 1, "double": 2, "triple": 3, "home_run": 4}[result]
        stats[player]["TB"] += bases
        if result == "home_run":
            stats[player]["HR"] += 1

    elif result in ["ground_out", "fly_out", "strikeout"]:
        stats[player]["AB"] += 1
        if result == "strikeout":
            stats[player]["K"] += 1

    elif result == "walk":
        stats[player]["BB"] += 1


def update_pitcher(stats, player, result, pitch_strike=None):
    if player not in stats:
        stats[player] = {"IP":0.0, "H": 0, "R": 0, "ER": 0, "BB": 0, "K": 0, "HR": 0, "P": 0, "S": 0}
        #innings pitched, hits, runs, earned runs, walks, strikeouts, home runs, total pitches, strikes
    
    stats[player]["P"] += 1
    if pitch_strike is True:
        stats[player]["S"] += 1
    
    # Event outcomes
    if result in ["single", "double", "triple", "home_run"]:
        stats[player]["H"] += 1
        if result == "home_run":
            stats[player]["HR"] += 1
            stats[player]["R"] += 1
    elif result == "walk":
        stats[player]["BB"] += 1
    elif result == "strikeout":
        stats[player]["K"] += 1
        stats[player]["IP"] += 1/3  # one out
    elif result in ["ground_out", "fly_out"]:
        stats[player]["IP"] += 1/3  # one out
    elif result == "run_scored":
        stats[player]["R"] += 1
        

def make_batting_box(team_dict):
    df = pd.DataFrame.from_dict(team_dict, orient="index").reset_index()
    df.rename(columns={"index": "Player"}, inplace=True)
    df["AVG"] = (df["H"] / df["AB"]).replace([float('inf'), pd.NA], 0).fillna(0).round(3)
    return df


def make_pitching_box(team_dict):
    df = pd.DataFrame.from_dict(team_dict, orient="index").reset_index()
    df.rename(columns={"index": "Pitcher"}, inplace=True)

    df["IP"] = df["IP"].round(1)
    df["ERA"] = (df["ER"] * 9 / df["IP"].replace(0, pd.NA)).fillna(0).round(2)
    df["PC-ST"] = df["P"].astype(str) + "-" + df["S"].astype(str)

    return df
    

def normalize_values(values, scale=1):
    # Handles list or dictionary, returning same format as given, but values normalized to the scale
    if isinstance(values, list):
        value_array = np.array(list(values), dtype=float)
        total = np.sum(value_array)
        new_values = (value_array / total) * scale
        return list(new_values)
    elif isinstance(values, dict):
        value_array = np.array(list(values.values()), dtype=float)
        total = np.sum(value_array)
        new_values = (value_array / total) * scale
        return dict(zip(values.keys(), new_values))
    else:
        raise TypeError("Unexpected variable type for values to be normalized. Getting ", type(values))
    

def get_aligned_value(pitcher_val, batter_val):
    # Currently returns the mean
    # With adjustments to the batter_val, like dropping swing rate on 3-0 counts,
    # it makes less sense to use a uniform or normal dist to pull a threshold value

    # np.random.uniform(min(pitcher_val, batter_val),
    #                   max(pitcher_val, batter_val))
    if isinstance(pitcher_val, list) and isinstance(batter_val, list):
        if len(pitcher_val) != len(batter_val):
            raise TypeError("Bad data for pitcher and batter. Attempting to align values: \n", pitcher_val, "\n", batter_val)
        new_values = []
        for i in range(len(pitcher_val)):
            new_values.append((pitcher_val[i] + batter_val[i]) / 2)
        return new_values
    elif isinstance(pitcher_val, float) and isinstance(batter_val, float):
        return (pitcher_val + batter_val) / 2
    else:   
        raise TypeError("Unexpected variable type aligning between pitcher and batter.\n",
                        "Pitcher is ", type(pitcher_val), " and batter is ", type(batter_val))
    

def read_data():
    # Read and prep data, returning dataframes
    batter_df = pd.read_excel("../Data/Merged_Data_C.xlsx", sheet_name="Batting Data")
    pitcher_df = pd.read_excel("../Data/Merged_Data_C.xlsx", sheet_name="Pitching Data")
    hit_traj_df = pd.read_excel("../Data/hit_trajectory.xlsx", sheet_name="Worksheet", index_col=0)

    pitcher_df.fillna(0, inplace=True) # TODO is this ok? Or need to be more selective like below?
    # pitcher_df[["FA%", "FT%", "FC%", "FS%", "FO%", "SI%", "SL%", "CU%", "KC%", "EP%", "CH%", "SC%", "KN%", "UN%"]] = pitcher_df[["FA%", "FT%", "FC%", "FS%", "FO%", "SI%", "SL%", "CU%", "KC%", "EP%", "CH%", "SC%", "KN%", "UN%"]].fillna(0)
    hit_traj_df["1B"] = hit_traj_df["H"] - hit_traj_df["2B"] - hit_traj_df["3B"] - hit_traj_df["HR"]
    hit_traj_df["Out"] = hit_traj_df["AB"] - hit_traj_df["H"]
    hit_traj_df["1B%"] = hit_traj_df["1B"] / hit_traj_df["AB"]
    hit_traj_df["2B%"] = hit_traj_df["2B"] / hit_traj_df["AB"]
    hit_traj_df["3B%"] = hit_traj_df["3B"] / hit_traj_df["AB"]
    hit_traj_df["HR%"] = hit_traj_df["HR"] / hit_traj_df["AB"]
    hit_traj_df["Out%"] = hit_traj_df["Out"] / hit_traj_df["AB"]
    
    return batter_df, pitcher_df, hit_traj_df


if __name__ == '__main__':
    # TODO add selection based on team? Data currently has lots of blanks bc of trades/moves
    # TODO don't have hard hit %s in merged_data_b, so can't calc GO/LO/FO from that
    
    batter_df, pitcher_df, hit_traj_df = read_data()

    batter_ids = list(batter_df["PlayerId"])
    pitcher_ids = list(pitcher_df["PlayerId"])
    # TODO add pitching changes, relief pitching. Currently just selecting a starting pitcher for each side to pitch the whole game
    selected_batter_ids = np.random.choice(batter_ids, 18, replace=False) # samples w/o replacement
    selected_pitcher_ids = np.random.choice(pitcher_ids, 2, replace=False) # samples w/o replacemnet
    
    # TODO update decision logic for new GB/LD/FB then 1B/2B/3B/HR/Out rates
    # TODO integrate new decision logic for other outcomes (IBB, HBP, SF, SH, BO)
    batter_data = [ 
        (lambda row:
            {'name': row["Name"],
             'id': row["PlayerId"],
             'team': row["Team"],
             'zone_prob': row["Zone%"],
             'swing_prob': {'strike': row["Z-Swing%"], 'ball': row["O-Swing%"]},
             'contact_prob': {'strike': row["Z-Contact%"], 'ball': row["O-Contact%"]},
             'foul_prob': {'strike': 0.22, 'ball': 0.22}, # TODO add this by-player? Also add foul-out chance/rate
            #  'int_walk_prob': row["IBB"] / row["PA"], # TODO no longer in merged data for these 4
            #  'hit_by_pitch_prob': row["HBP"] / row["PA"],
            #  'sac_fly_prob': row["SF"] / row["PA"],
            #  'sac_bunt_prob': row["SH"] / row["PA"],
             # TODO add GDP rate/logic
             'contact_cat_prob': {
                            "ground_ball": row["GB%"],
                            "line_drive": row["LD%"],
                            "fly_ball": row["FB%"],
                            # TODO eventually, add Bunts? Split off from GB% somehow?
                            },
             'outcome_prob': {
                            # TODO add way to track what type of single/double/triple/hr/out it was for event log
                            "ground_ball": {
                                            "single": hit_traj_df.loc["Ground Balls", "1B%"],
                                            "double": hit_traj_df.loc["Ground Balls", "2B%"],
                                            "triple": hit_traj_df.loc["Ground Balls", "3B%"],
                                            "home_run": hit_traj_df.loc["Ground Balls", "HR%"], # this should always be 0
                                            "out": hit_traj_df.loc["Ground Balls", "Out%"]
                                            },
                            "line_drive": {
                                            "single": hit_traj_df.loc["Line Drives", "1B%"],
                                            "double": hit_traj_df.loc["Line Drives", "2B%"],
                                            "triple": hit_traj_df.loc["Line Drives", "3B%"],
                                            "home_run": hit_traj_df.loc["Line Drives", "HR%"],
                                            "out": hit_traj_df.loc["Line Drives", "Out%"]
                                            },
                            "fly_ball": { # TODO incorporate IFFB% somehow? Also from pitcher side
                                            "single": hit_traj_df.loc["Fly Balls", "1B%"],
                                            "double": hit_traj_df.loc["Fly Balls", "2B%"],
                                            "triple": hit_traj_df.loc["Fly Balls", "3B%"],
                                            "home_run": row["HR/FB"], # Note, this will make sum likely not 1, which is normalized later
                                            "out": hit_traj_df.loc["Fly Balls", "Out%"]
                                            },
                            # "bunt": {
                            #                 "single": hit_traj_df.loc["Bunts", "1B%"],
                            #                 "double": hit_traj_df.loc["Bunts", "2B%"],
                            #                 "triple": hit_traj_df.loc["Bunts", "3B%"],
                            #                 "home_run": hit_traj_df.loc["Bunts", "HR%"], # this should always be 0
                            #                 "out": hit_traj_df.loc["Bunts", "Out%"] # this should always be 0
                            #                 }
                            },
             'outcome_power_prob': {
                                "soft": row["Soft%"],
                                "medium": row["Med%"],
                                "hard": row["Hard%"]
                              },
            # TODO add stolen bases, wild pitches, errors?
            # TODO pull from distribution for distance of a ball hit, or power, for chances of tagging up or multiple bases?
            })
        (batter_df[batter_df["PlayerId"] == bid].iloc[0])
            for bid in selected_batter_ids
    ]

    base_velocity_dist = {95:0.5,
                        100:0.5}
    base_movement_prob = {'straight': 0.25,
                          'down': 0.25,
                          'side': 0.25,
                          'fade': 0.25}
    pitcher_data = [
        (lambda row:
            {'name': row["Name"],
             'id': row["PlayerId"],
             'team': row["Team"],
             'pitch_type_prob': {'fastball': row["FA%"], # + row["FT%"], # Adding 4-seam, 2-seam, and unclassified tgr
                                 'curveball': row["CU%"],
                                 'slider': row["SL%"],
                                 'changeup': row["CH%"]},
                                 # TODO eventually, add Bunts? Split off from GB% somehow?
             'swing_prob': {'strike': row["Z-Swing%"], 'ball': row["O-Swing%"]},
             'contact_prob': {'strike': row["Z-Contact%"], 'ball': row["O-Contact%"]},
             'contact_cat_prob': {
                            "ground_ball": row["GB%"],
                            "line_drive": row["LD%"],
                            "fly_ball": row["FB%"],
                            # TODO eventually, add Bunts?
                            },
             'outcome_power_prob': {
                                "soft": row["Soft%"],
                                "medium": row["Med%"],
                                "hard": row["Hard%"]
                              },
             'velocity_dist': perturb_values(base_velocity_dist, 0.05), # TODO replace with params for some other RN pull on pitch
             'movement_prob': perturb_values(base_movement_prob, 0.05), # TODO replace with params for some other RN pull on pitch
             'strike_prob': row["Zone%"] # prob inside zone, not of being a strike bc of zone/swing/foul
            }
        )(pitcher_df[pitcher_df["PlayerId"] == pid].iloc[0])
            for pid in selected_pitcher_ids
    ]
    
    batters1 = [Batter(batter) for batter in batter_data[0:9]]
    batters2 = [Batter(batter) for batter in batter_data[9:]]
    
    pitcher1 = Pitcher(pitcher_data[0])
    pitcher2 = Pitcher(pitcher_data[1])
    team1 = Team(batters1,[pitcher1])
    team2 = Team(batters2,[pitcher2])
    
    game = Game(team1, team2)
    game.play_ball()
    
    event_log = pd.DataFrame(game.event_log)
    event_log.to_csv("event_log.csv", encoding='utf-8-sig', index=False)
    
    make_box_score(event_log)
        
    
    ### Some code below to run 100 games and compute team record, average score
    # scores = {i:[] for i in range(1,10)}
    # team1_record = [0,0,0]
    # team2_record = [0,0,0]
    # for i in range(1000):
    #     team1 = Team(batters1,[pitcher1])
    #     team2 = Team(batters2,[pitcher1])
        
    #     game = Game(team1, team2)   
    #     game.play_ball()
        
    #     event_log = pd.DataFrame(game.event_log)
        
    #     team1_score = event_log.at[len(event_log)-1,'Team 1 Score']
    #     team2_score = event_log.at[len(event_log)-1,'Team 2 Score']
        
    #     if team1_score > team2_score:
    #         team1_record[0] += 1
    #         team2_record[1] += 1
    #     elif team2_score > team1_score:
    #         team1_record[1] += 1
    #         team2_record[0] += 1
    #     else:
    #         team1_record[2] += 1
    #         team2_record[2] += 1
        
    #     for j in range(1,10):
    #         max_index = max(event_log[event_log['Inning'] == j].index)
    #         scores[j].append(event_log.at[max_index, 'Team 1 Score'])
    #         scores[j].append(event_log.at[max_index, 'Team 2 Score'])
            
    # avg_scores_by_inning = {i:np.average(scores[i]) for i in range(1,10)}    


# TODO do v&v by looking at values vs avg hits, runs, pitches, etc. per game, but also at BB%, R, AVG, etc. for players over many games
