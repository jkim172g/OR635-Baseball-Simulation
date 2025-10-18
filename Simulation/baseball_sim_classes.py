
import pandas as pd
import numpy as np
import random


import pdb

class Batter:
    
    def __init__(self, batter_data):
        
        #Save basic information about batter
        self.name = batter_data['name']
        self.team = batter_data['team']
        
        #Save this batter's probability of swinging, making contact, and result
        self.swing_prob = batter_data['swing_prob']
        self.contact_prob = batter_data['contact_prob']
        self.outcome_prob = batter_data['outcome_prob']
        
    def __str__(self):
        return f"Batter: {self.name}, {self.team}"
        
class Pitcher:
    
    def __init__(self, pitcher_data):
        
        #Save basic information about pitcher
        self.name = pitcher_data['name']
        self.team = pitcher_data['team']
        
        #Save statistics about pitching probabilities
        self.pitch_type_prob = pitcher_data['pitch_type_prob']
        self.velocity_dist = pitcher_data['velocity_dist']
        self.movement_prob = pitcher_data['movement_prob']
        self.strike_prob = pitcher_data['strike_prob']
        
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
        # self.bases = {"First": False, "Second": False, "Third": False}
        # TODO could also place the players on the bases, to track their stats?
        self.team1_batter_index = 0
        self.team2_batter_index = 0
        self.team1_pitcher_index = 0
        self.team2_pitcher_index = 0
        
        self.team1 = team1
        self.team2 = team2
        
        self.event_log = {'Inning':[self.inning], 'Inning Half':[self.inning_half],'Event':['Init'],
                          'Batter':['NA'], 'Bases':[self.bases], 'Balls':[self.balls],
                          'Strikes':[self.strikes], 'Outs':[self.outs],'Team 1 Score':[0], 'Team 2 Score':[0], 'Pitcher':['NA'],
                          }
    
    def pitch(self, batter, pitcher):
        # Get pitch details
        pitch_type = random.choices(list(pitcher.pitch_type_prob.keys()), list(pitcher.pitch_type_prob.values()))[0]
        pitch_velocity = random.choices(list(pitcher.velocity_dist.keys()), list(pitcher.velocity_dist.values()))[0]
        pitch_movement = random.choices(list(pitcher.movement_prob.keys()), list(pitcher.movement_prob.values()))[0]
        
        
        #Calculate result
        
        strike = True if np.random.uniform() < pitcher.strike_prob else False
        if strike:
            #If strike, check if batter swings
            swing = True if np.random.uniform() < batter.swing_prob['strike'] else False
            if swing:
                #If batter swings, check if contact was made
                contact = True if np.random.uniform() < batter.contact_prob['strike'] else False
                #If contact was made, calculate result
                if contact:
                    result = random.choices(list(batter.outcome_prob.keys()), list(batter.outcome_prob.values()))[0]
                
                else:
                    #If swung and missed, strike 
                    result = 'strike'
            else:
                #If strike not swung at, strike
                result = 'strike'
        else:
            #If ball was thrown, check if it was swung at
            swing = True if np.random.uniform() < batter.swing_prob['ball'] else False
            if swing:
                #If batter swings, check if contact was made
                contact = True if np.random.uniform() < batter.contact_prob['ball'] else False
                #If contact was made, calculate result
                if contact:
                    result = random.choices(list(batter.outcome_prob.keys()), list(batter.outcome_prob.values()))[0]
                
                else:
                    #If swung and missed, strike 
                    result = 'strike'
            else:
                #if ball not swung at, result is a ball
                result = 'ball'
               
        return result
        
    
    def move_bases(self, action, batting_team):
        #Put logic here for moving bases based on the action
     
        ## How do we determine whether a player already on base was out?
        #pdb.set_trace()
        if action == 'home_run':
            batting_team.score += sum(self.bases) + 1
            self.bases = [0,0,0]
        
        elif action == 'walk':
            if self.bases == [1,1,1]:
                batting_team.score += self.bases[2]
            if self.bases[0] == 1 and self.bases[1] == 1:
                self.bases[2] = 1
            
            self.bases[1] = 1 if self.bases[0] == 1 else self.bases[1]
            self.bases[0] = 1
        
        elif action == 'single':
            batting_team.score += self.bases[2]
            self.bases[2] = self.bases[1]
            self.bases[1] = self.bases[0]
            self.bases[0] = 1
        
        elif action == 'double':
            
            batting_team.score += self.bases[2] + self.bases[1]
            self.bases[2] = self.bases[0]
            self.bases[1] = 1
            self.bases[0] = 0
        
        elif action == 'triple':
            batting_team.score += sum(self.bases)
            self.bases = [0,0,1]
        
        else:
            self.outs += 1
    
    def simulate_inning_half(self, batting_team, pitching_team):
        
        while self.outs < 3:
            #Set batter and pitcher
            self.strikes = 0
            self.balls = 0
            current_batter = batting_team.batters[batting_team.batter_index]
            current_pitcher = pitching_team.pitchers[pitching_team.pitcher_index]
            #Run through a pitch and update outcomes
            hit = False
            while self.strikes < 3 and self.balls < 4 and not hit:
                #Get result of pitch
                pitch_result = self.pitch(current_batter, current_pitcher)
                #Update stats
                if pitch_result == 'strike':
                        self.strikes += 1
                        #Record an out if strike out
                        if self.strikes == 3:
                            self.outs += 1
                elif pitch_result == 'ball':
                        self.balls += 1
                        #Record a walk if ball 4
                        if self.balls ==4:
                            self.move_bases('walk',batting_team)
                else:
                    #Move bases if hit
                    self.move_bases(pitch_result,batting_team)
                    hit = True
                #After each pitch, update current game state
                self.update_event_log(current_batter, current_pitcher,pitch_result)

            #Update batter
            if batting_team.batter_index < batting_team.num_batters-1:
                batting_team.batter_index += 1
            else:
                batting_team.batter_index =0
        
    def update_event_log(self,current_batter, current_pitcher, event):
        #Add all current information to the event log
        self.event_log['Inning'].append(self.inning)
        self.event_log['Inning Half'].append(self.inning_half)
        self.event_log['Outs'].append(self.outs)
        self.event_log['Strikes'].append(self.strikes)
        self.event_log['Balls'].append(self.balls)
        self.event_log['Bases'].append(self.bases[:])
        self.event_log['Team 1 Score'].append(self.team1.score)
        self.event_log['Team 2 Score'].append(self.team2.score)
        self.event_log['Batter'].append(current_batter.name)
        self.event_log['Pitcher'].append(current_pitcher.name)
        self.event_log['Event'].append(event)
    
    def play_ball(self):
        #pdb.set_trace()
        #Loop through innings
        while self.inning <= 9:
            self.outs = 0
            self.strikes = 0
            self.balls = 0
            self.bases = [0,0,0]
            if self.inning_half == 'top':
                #If top of the inning, team1 is batting and team2 is pitching

                self.simulate_inning_half(team1, team2)    
                self.inning_half = 'bottom'
                
            else:
                #If bottom of the inning, team2 is batting and team1 is pitching

                self.simulate_inning_half(team2, team1) 
                self.inning_half = 'top'
                self.inning += 1
            
        return self.event_log
    

def perturb_values(orig_val_dict, range):
    val_dict = orig_val_dict.copy()
    # perturb
    for k, v in val_dict.items():
        direction = random.choice([-1, 1])
        scale = random.random()
        perturb = scale * direction * range
        val_dict[k] = v + perturb

    # normalize back to 1
    denom = sum(val_dict.values())
    norm = 1/denom
    for k, v in val_dict.items():
        val_dict[k] = v*norm

    return val_dict
    

if __name__ == '__main__':
    # TODO add selection based on team? Data currently has lots of blanks bc of trades/moves
    batter_df = pd.read_excel("Merged_Data.xlsx", sheet_name='Batting Data')
    pitcher_df = pd.read_excel("Merged_Data.xlsx", sheet_name='Pitching Data')

    batter_ids = batter_df["PlayerID"]
    pitcher_ids = pitcher_df["PlayerID"]
    # TODO add pitching changes, relief pitching. Currently just selecting a starting pitcher for each side to pitch the whole game

    selected_batter_ids = random.sample(batter_ids, 18) # samples w/o replacement
    selected_pitcher_ids = random.sample(pitcher_ids, 2) # samples w/o replacemnet
    
    base_outcome_prob = {'single': 0.175,
                        'double': 0.1,
                        'triple': 0.05,
                        'home_run': 0.025,
                        'ground_out': 0.35,
                        'fly_out': 0.3}
        
    batter_data = [
        (lambda row: 
            {'name': row["Name"],
                'team': row["Team"],
                'swing_prob': {'strike': row["Z-Swing%"], 'ball': row["O-Swing%"]},
                'contact_prob': {'strike': row["Z-Contact%"], 'ball': row["O-Contact%"]},
                'outcome_prob': perturb_values(base_outcome_prob, .05) # TODO replace with calculated %s, from power stats like SLG, LO, etc., still TBD
            }
        )(batter_df[batter_df["PlayerID"] == bid])
            for bid in selected_batter_ids
    ]
    i = 1
    for batter in batter_data:
        batter['name'] = str(i)
        i += 1

    base_velocity_dist = {95:0.5,
                        100:0.5}
    
    base_movement_prob = {'straight': 0.25,
                                  'down': 0.25,
                                  'side': 0.25,
                                  'fade': 0.25}

    pitcher_data = [
        (lambda row:
            {'name': row["Name"],
             'team': row["Team"],
             'pitch_type_prob': {'fastball': row["FA%"] + row["FT%"], # Adding 4-seam, 2-seam, and unclassified tgr
                                  'curveball': row["CU%"],
                                  'slider': row["SL%"],
                                  'changeup': row["CH%"]},
                                  # TODO what to do if they don't add to 1, bc of other pitch types? Just expand categories handled in sim?
             'velocity_dist': perturb_values(base_velocity_dist, 0.05), # TODO replace with params for some other RN pull on pitch
             'movement_prob': perturb_values(base_movement_prob, 0.05), # TODO replace with params for some other RN pull on pitch
             'strike_prob': row["Zone%"] # prob inside zone, not of being a strike bc of zone/swing/foul
            }
        )(pitcher_df[pitcher_df["PlayerID"] == pid])
            for pid in selected_pitcher_ids
    ]
    
    
    batters1 = [Batter(batter) for batter in batter_data[0:8]]
    batters2 = [Batter(batter) for batter in batter_data[9:]]
    
    pitcher1 = Pitcher(pitcher_data[0])
    pitcher2 = Pitcher(pitcher_data[1])
    team1 = Team(batters1,[pitcher1])
    team2 = Team(batters2,[pitcher2])
    
    game = Game(team1, team2)
    game.play_ball()
    
    event_log = pd.DataFrame(game.event_log)
    
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
      
        
        
        
        