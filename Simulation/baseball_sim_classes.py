## Project for GMU OR 635: Discrete System Simulation
## Authors: Jacob Kim, Noah Mecikalski, Bennett Miller, W.J. Moretz

import pandas as pd
import numpy as np
import pdb

np.random.seed(100) # TODO need to set any random streams/substreams?

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
        self.num_batters = 0
        self.runs_allowed = 0
        self.hits_allowed = 0
        self.outs = 0
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
        
        self.event_log = {'Inning':[], 'Inning Half':[],'Event':[], 'Detailed Event':[], 'Pitch Type':[],
                          'Pitch Outcome':[], 'Batter':[], 'Batter Number':[], 'Bases':[],
                          'Baserunners':[], 'Baserunning Event':[], 'Baserunning Result': [],'Balls':[], 'Strikes':[],
                          'Outs':[],'Team 1 Score':[], 'Team 2 Score':[], 'Pitcher':[],
                          }
    
    def pitch(self, batter, pitcher):
        # Get pitch details
        pitcher.num_pitch += 1
        pitch_type = np.random.choice(list(pitcher.pitch_type_prob.keys()), p=normalize_values(list(pitcher.pitch_type_prob.values()), 1))
        #pitch_velocity = np.random.choice(list(pitcher.velocity_dist.keys()), p=normalize_values(list(pitcher.velocity_dist.values()), 1))
        #pitch_movement = np.random.choice(list(pitcher.movement_prob.keys()), p=normalize_values(list(pitcher.movement_prob.values()), 1))
        contact_cat = None
        power = None

        #Calculate result
        batter_zone_prob = batter.zone_prob[pitch_type] or batter.zone_prob["na"]
        strike_prob = get_aligned_value(pitcher.strike_prob, batter_zone_prob) # Uses Zone% for each
        strike = True if np.random.uniform() < strike_prob else False
        if strike:
            pitch_result = 'strike'
            #If strike, check if batter swings
            #Adjust swing prob by half if 3 balls and less than two strikes
            # Uses Z-Swing%
            base_batter_swing_prob = batter.swing_prob[pitch_type]["strike"] or batter.swing_prob["na"]["strike"]
            adj_batter_swing_prob = base_batter_swing_prob/2 if self.balls == 3 and self.strikes < 2 else base_batter_swing_prob
            swing_prob = get_aligned_value(pitcher.swing_prob['strike'], adj_batter_swing_prob)
            swing = True if np.random.uniform() < swing_prob else False
            if swing:
                #If batter swings, check if contact was made
                # Uses Z-Contact%
                batter_contact_prob = batter.contact_prob[pitch_type]["strike"] or batter.contact_prob["na"]["strike"]
                contact_prob = get_aligned_value(pitcher.contact_prob['strike'], batter_contact_prob)
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
                        outcome_power_prob = batter.outcome_power_prob
                        power = np.random.choice(list(outcome_power_prob.keys()), p=normalize_values(list(outcome_power_prob.values()), 1))
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
            base_batter_swing_prob = batter.swing_prob[pitch_type]["ball"] or batter.swing_prob["na"]["ball"]
            adj_batter_swing_prob = base_batter_swing_prob/2 if self.balls == 3 and self.strikes < 2 else base_batter_swing_prob
            swing_prob = get_aligned_value(pitcher.swing_prob['ball'], adj_batter_swing_prob)
            swing = True if np.random.uniform() < swing_prob else False
            if swing:
                # Pitch now considered a strike regardless of outcome
                pitch_result= 'strike'
                #If batter swings, check if contact was made
                # Uses O-Contact%
                batter_contact_prob = batter.contact_prob[pitch_type]["ball"] or batter.contact_prob["na"]["ball"]
                contact_prob = get_aligned_value(pitcher.contact_prob['ball'], batter_contact_prob)
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
                        outcome_power_prob = batter.outcome_power_prob
                        power = np.random.choice(list(outcome_power_prob.keys()), p=normalize_values(list(outcome_power_prob.values()), 1))
                        result = np.random.choice(list(outcome_prob.keys()), p=normalize_values(list(outcome_prob.values()), 1))
                else:
                    #If swung and missed, strike 
                    result = 'strike'
            else:
                pitch_result= 'ball'
                #if ball not swung at, result is a ball
                result = 'ball'         
        
        return result, pitch_result, swing, pitch_type, contact_cat, power
        

    def move_bases(self, action, batting_team, current_batter, contact_cat, power, current_pitcher):
        #Put logic here for moving bases based on the action
        
        def first_to_third(contact_cat, power):
            # prob is (rate attempt, rate thrown out)
            first_to_third_probs = {('soft','ground_ball'):(0.65,0.1),
                                     ('soft','line_drive'):(0.75,0.05),
                                     ('soft','fly_ball'):(0.1,0.01),
                                     ('medium','ground_ball'):(0.66,0.05),
                                     ('medium','line_drive'):(0.33,0.07),
                                     ('medium','fly_ball'):(0.6,0.2),
                                     ('hard','ground_ball'):(0.25,0.1),
                                     ('hard','line_drive'):(0.66,0.05),
                                     ('hard','fly_ball'):(0.8,0.15)}
            result = []
            probs = first_to_third_probs[(power, contact_cat)]
            attempt = True if np.random.uniform() < probs[0] else False
            if attempt:
                result.append('first to third attempted')
                thrown_out = True if np.random.uniform() < probs[1] else False
                if thrown_out:
                    result.append('thrown out')
                    self.outs += 1
                    current_pitcher.outs += 1
                else:
                   result.append('safe')

            return result
        
        def single_to_double(contact_cat, power):
            # prob is (rate attempt, rate thrown out)
            single_to_double_probs = {('soft','ground_ball'):(0,0),
                                     ('soft','line_drive'):(0,0),
                                     ('soft','fly_ball'):(0,0),
                                     ('medium','ground_ball'):(0,0),
                                     ('medium','line_drive'):(0.05,0.25),
                                     ('medium','fly_ball'):(0,0),
                                     ('hard','ground_ball'):(0.1,0.4),
                                     ('hard','line_drive'):(0.1,0.4),
                                     ('hard','fly_ball'):(0.1,0.4)}
            result = []
            probs = single_to_double_probs[(power, contact_cat)]
            attempt = True if np.random.uniform() < probs[0] else False
            if attempt:
                result.append('single to double attempted')
                thrown_out = True if np.random.uniform() < probs[1] else False
                if thrown_out:
                    result.append('thrown out')
                    self.outs += 1
                    current_pitcher.outs += 1
                else:
                   result.append('safe')

            return result
        
        def first_to_home_double(contact_cat):
            # prob is (rate attempt, rate thrown out)
            num_outs = '2' if self.outs == 1 else '<2'
            run_diff = '>3' if abs(self.team1.score - self.team2.score) >3 else '<3'
            if num_outs != '2':
                game_state = num_outs + ' ' + run_diff
            else:
                game_state = '2'
            first_to_home_probs = {('<2 >3','ground_ball'):(0.5,0.1),
                                     ('<2 >3','line_drive'):(0.5,0.1),
                                     ('<2 >3','fly_ball'):(0.3,0.1),
                                     ('<2 <3','ground_ball'):(0.4,0.03),
                                     ('<2 <3','line_drive'):(0.6,0.03),
                                     ('<2 <3','fly_ball'):(0.2,0.03),
                                     ('2','ground_ball'):(0.9,0.1),
                                     ('2','line_drive'):(0.9,0.1),
                                     ('2','fly_ball'):(0.9,0.1)}
            result = []
            probs = first_to_home_probs[(game_state, contact_cat)]
            attempt = True if np.random.uniform() < probs[0] else False
            if attempt:
                result.append('first to home attempted')
                thrown_out = True if np.random.uniform() < probs[1] else False
                if thrown_out:
                    result.append('thrown out')
                    self.outs += 1
                    current_pitcher.outs += 1
                else:
                   result.append('safe')

            return result
        
        def double_to_triple(contact_cat, power):
            # prob is (rate attempt, rate thrown out)
            single_to_double_probs = {('soft','ground_ball'):(0,0),
                                     ('soft','line_drive'):(0,0),
                                     ('soft','fly_ball'):(0,0),
                                     ('medium','ground_ball'):(0,0),
                                     ('medium','line_drive'):(0,0),
                                     ('medium','fly_ball'):(0,0),
                                     ('hard','ground_ball'):(0.05,0.6),
                                     ('hard','line_drive'):(0.05,0.6),
                                     ('hard','fly_ball'):(0.05,0.6)}
            result = []
            probs = single_to_double_probs[(power, contact_cat)]
            attempt = True if np.random.uniform() < probs[0] else False
            if attempt:
                result.append('single to double attempted')
                thrown_out = True if np.random.uniform() < probs[1] else False
                if thrown_out:
                    result.append('thrown out')
                    self.outs += 1
                    current_pitcher.outs += 1
                else:
                   result.append('safe')

            return result
        
        def tag_up_second_to_third(contact_cat, power):
            # prob is (rate attempt, rate thrown out)
            tag_up_second_to_third_probs = {('soft','ground_ball'):(0,0),
                                     ('soft','line_drive'):(0,0),
                                     ('soft','fly_ball'):(0,0),
                                     ('medium','ground_ball'):(0,0),
                                     ('medium','line_drive'):(0,0),
                                     ('medium','fly_ball'):(0,0),
                                     ('hard','ground_ball'):(0,0),
                                     ('hard','line_drive'):(0,0),
                                     ('hard','fly_ball'):(0.6,0.1)}
            result = []
            probs = tag_up_second_to_third_probs[(power, contact_cat)]
            attempt = True if np.random.uniform() < probs[0] else False
            if attempt:
                result.append('tag up second to third attempted')
                thrown_out = True if np.random.uniform() < probs[1] else False
                if thrown_out:
                    result.append('thrown out')
                    self.outs += 1
                    current_pitcher.outs += 1
                else:
                    result.append('safe')
            
            return result
            
        def tag_up_third_to_home(contact_cat, power):
            # prob is (rate attempt, rate thrown out)
            tag_up_third_to_home_probs = {('soft','ground_ball'):(0,0),
                                     ('soft','line_drive'):(0,0),
                                     ('soft','fly_ball'):(0,0),
                                     ('medium','ground_ball'):(0,0),
                                     ('medium','line_drive'):(0.6,0.2),
                                     ('medium','fly_ball'):(0.6,0.2),
                                     ('hard','ground_ball'):(0,0),
                                     ('hard','line_drive'):(1,0),
                                     ('hard','fly_ball'):(1,0)}
            result = []
            probs = tag_up_third_to_home_probs[(power, contact_cat)]
            attempt = True if np.random.uniform() < probs[0] else False
            if attempt:
                result.append('tag up third to home attempted')
                thrown_out = True if np.random.uniform() < probs[1] else False
                if thrown_out:
                    result.append('thrown out')
                    self.outs += 1
                    current_pitcher.outs += 1
                else:
                    result.append('safe')
            
            return result
                 
        ## How do we determine whether a player already on base was out?
        #pdb.set_trace()
        baserunning_result = []
        if action == 'home_run':
            batting_team.score += sum(self.bases) + 1
            current_pitcher.runs_allowed += sum(self.bases) + 1
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
                current_pitcher.runs_allowed += 1
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
        elif action == 'single':
            # Batter goes to First, any baserunners advance, score if on Third
            batting_team.score += self.bases[2]
            current_pitcher.runs_allowed += self.bases[2]
            # Check if runner on first
            if self.bases[0] == 1:
                #If runner on first, check if runner attempts to third
                first_to_third_result = first_to_third(contact_cat, power)
                
                if len(first_to_third_result) == 0:
                    #If not attempted, all runners advance one base
                    self.bases[2] = self.bases[1]
                    self.bases[1] = self.bases[0]
                    self.bases[0] = 1
                    self.baserunners[2] = self.baserunners[1]
                    self.baserunners[1] = self.baserunners[0]
                    self.baserunners[0] = current_batter.name
                elif first_to_third_result[1] == 'safe':
                    baserunning_result.append(first_to_third_result)
                    # If safe, any runner on second scores and first goes to third
                    self.bases[2] = 1
                    batting_team.score += self.bases[1]
                    current_pitcher.runs_allowed += self.bases[1]
                    self.baserunners[2] = self.baserunners[0]
                    # Check single to double
                    single_to_double_result = single_to_double(contact_cat, power)
                    
                    # If not single to double, batter just goes one base
                    if len(single_to_double_result) == 0:
                        self.bases[1] = 0
                        self.bases[0] = 1
                        self.baserunners[1] = 0
                        self.baserunners[0] = current_batter.name
                    elif single_to_double_result[1] == 'safe':
                        #If single to double safe, move bases accordingly
                        baserunning_result.append(single_to_double_result)
                        self.bases[1] = 1
                        self.bases[0] = 0
                        self.baserunners[1] = current_batter.name
                        self.baserunners[0] = 0
                    else:
                        #If not safe update
                        baserunning_result.append(single_to_double_result)
                        self.bases[1] = 0
                        self.bases[0] = 0
                        self.baserunners[1] = 0
                        self.baserunners[0] = 0    
                else:
                    baserunning_result.append(first_to_third_result)
                    #If first to third not safe, then batter will NOT attempt to go to second. Update accordingly
                    self.bases[2] = 0
                    self.bases[1] = 0
                    self.bases[0] = 1
                    self.baserunners[2] = 0
                    self.baserunners[1] = 0
                    self.baserunners[0] = current_batter.name
            else:
                #If no runner on first, check if batter attempts to go to second
                single_to_double_result = single_to_double(contact_cat, power)
                
                if len(single_to_double_result) == 0:
                    #If no attempt to second, all baserunners move one base
                    self.bases[2] = self.bases[1]
                    self.bases[1] = self.bases[0]
                    self.bases[0] = 1
                    self.baserunners[2] = self.baserunners[1]
                    self.baserunners[1] = self.baserunners[0]
                    self.baserunners[0] = current_batter.name
                elif single_to_double_result[1] == 'safe':
                    baserunning_result.append(single_to_double_result)
                    #If attempted and safe, runner on second goes to third
                    self.bases[2] = self.bases[1]
                    self.bases[1] = 1
                    self.bases[0] = 0
                    self.baserunners[2] = self.baserunners[1]
                    self.baserunners[1] = current_batter.name
                    self.baserunners[0] = 0
                else:
                    #If attempted and out, update
                    baserunning_result.append(single_to_double_result)
                    self.bases[2] = self.bases[1]
                    self.bases[1] = 0
                    self.bases[0] = 0
                    self.baserunners[2] = self.baserunners[1]
                    self.baserunners[1] = 0
                    self.baserunners[0] = 0
        elif action == 'double':
            # Runners on third and second score
            batting_team.score += self.bases[2] + self.bases[1]
            current_pitcher.runs_allowed += self.bases[2] + self.bases[1]
            # Check if runner on first
            if self.bases[0] == 1:
                #If runner on first, check if runner attempts to home
                first_to_home_result = first_to_home_double(contact_cat)
                
                if len(first_to_home_result) == 0:
                    #If not attempted, all runners advance two bases
                    self.bases[2] = self.bases[0]
                    self.bases[1] = 1
                    self.bases[0] = 0
                    self.baserunners[2] = self.baserunners[0]
                    self.baserunners[1] = current_batter.name
                    self.baserunners[0] = 0
                elif first_to_home_result[1] == 'safe':
                    baserunning_result.append(first_to_home_result)
                    # If safe, add one to score
                    batting_team.score += 1
                    current_pitcher.runs_allowed += 1
                    
                    # Check double to triple
                    double_to_triple_result = double_to_triple(contact_cat, power)
                    
                    if len(double_to_triple_result) == 0:
                        # If not double to triple not attempted, batter goes two bases. First and third are empty
                        self.bases[2] = 0
                        self.bases[1] = 1
                        self.bases[0] = 0
                        self.baserunners[2] = 0
                        self.baserunners[1] = current_batter.name
                        self.baserunners[0] = 0
                    elif double_to_triple_result[1] == 'safe':
                        #If double to triple safe, only third is 
                        baserunning_result.append(double_to_triple_result)
                        self.bases[2] = 1
                        self.bases[1] = 0
                        self.bases[0] = 0
                        self.baserunners[2] = current_batter.name
                        self.baserunners[1] = 0
                        self.baserunners[0] = 0
                    else:
                        #If not safe, bases are empty
                        baserunning_result.append(double_to_triple_result)
                        self.bases[2] = 0
                        self.bases[1] = 0
                        self.bases[0] = 0
                        self.baserunners[2] = 0
                        self.baserunners[1] = 0
                        self.baserunners[0] = 0      
                else:
                    baserunning_result.append(first_to_home_result)
                    #If first to home not safe, then batter will NOT attempt to go to third. Update accordingly
                    self.bases[2] = 0
                    self.bases[1] = 1
                    self.bases[0] = 0
                    self.baserunners[2] = 0
                    self.baserunners[1] = current_batter.name
                    self.baserunners[0] = 0
            else:
                #If no runner on first, check if batter attempts to go to third
                double_to_triple_result = double_to_triple(contact_cat, power)
                
                if len(double_to_triple_result) == 0:
                    #If no attempt to third, batter ends at second. first and third empty
                    self.bases[2] = 0
                    self.bases[1] = 1
                    self.bases[0] = 0
                    self.baserunners[2] = 0
                    self.baserunners[1] = current_batter.name
                    self.baserunners[0] = 0
                elif double_to_triple_result[1] == 'safe':
                    #If attempted and safe, batter goes to third, all others empty
                    baserunning_result.append(double_to_triple_result)
                    
                    self.bases[2] = 1
                    self.bases[1] = 0
                    self.bases[0] = 0
                    self.baserunners[2] = current_batter.name
                    self.baserunners[1] = 0
                    self.baserunners[0] = 0
                else:
                    #If attempted and out, all bases empty
                    baserunning_result.append(double_to_triple_result)
                    self.bases[2] = 0
                    self.bases[1] = 0
                    self.bases[0] = 0
                    self.baserunners[2] = 0
                    self.baserunners[1] = 0
                    self.baserunners[0] = 0
        elif action == "triple":
            # All runners clear bases, score, batter goes to Third
            # Note there is no attempted extra bases with this, as triples are already very rare, so an inside the park HR is incredibly unlikely
            batting_team.score += sum(self.bases)
            current_pitcher.runs_allowed += sum(self.bases)
            self.bases = [0,0,1]
            self.baserunners[2] = current_batter.name
            self.baserunners[1] = 0
            self.baserunners[0] = 0
        elif action == "out":
            self.outs += 1
            current_pitcher.outs += 1
            if self.outs < 3:
                # inning not over, try tagging up from third
                if self.bases[2] == 1:
                    # runner on third, valid to try for home
                    tag_up_third_to_home_result = tag_up_third_to_home(contact_cat, power)

                    if len(tag_up_third_to_home_result) == 0:
                        # If no tag up attempt, no movement
                        pass
                    elif tag_up_third_to_home_result[1] == 'safe':
                        # If attempted and safe, runner on third scores, add score, clear third
                        baserunning_result.append(tag_up_third_to_home_result)
                        
                        batting_team.score += 1
                        current_pitcher.runs_allowed += 1

                        self.bases[2] = 0
                        self.baserunners[2] = 0
                    else:
                        # If attempted and out, clear third (out added in tag up method)
                        baserunning_result.append(single_to_double_result)

                        self.bases[2] = 0
                        self.baserunners[2] = 0
            if self.outs < 3:
                # inning still not over, try tagging up from second
                if self.bases[1] == 1 and self.bases[2] == 0:
                    # runner on second and not on third, valid to try for third
                    tag_up_second_to_third_result = tag_up_second_to_third(contact_cat, power)

                    if len(tag_up_second_to_third_result) == 0:
                        # If no tag up attempt, no movement
                        pass
                    elif tag_up_second_to_third_result[1] == 'safe':
                        # If attempted and safe, runner on second moves to third, second empty
                        baserunning_result.append(tag_up_third_to_home_result)

                        self.bases[2] = self.bases[1]
                        self.bases[1] = 0
                        self.baserunners[2] = self.baserunners[1]
                        self.baserunners[1] = 0
                    else:
                        # If attempted and out, clear second (out added in tag up method)
                        baserunning_result.append(single_to_double_result)

                        self.bases[1] = 0
                        self.baserunners[1] = 0
        else:
            # Shouldn't get here unless new outcomes are added
            raise NotImplementedError("Logic for this event action is not implemented yet.")

        return baserunning_result
    
    def determine_pitch_change(self, pitching_team, batting_team, pitcher, prev_pitches,
                               prev_batting_score, runs_allowed, baserunner_count, starter):
        replace = False
        if starter:
            # logic for starters
            if pitcher.num_pitch > np.random.normal(105,5)-3:
                # reached high pitch count
                if not (self.inning >= 9 and pitching_team.score > batting_team.score):
                    # not winning in the 9th or after
                    replace = True
            elif pitcher.num_pitch - prev_pitches >= 35:
                # threw 35+ pitches in an inning
                replace = True
            elif batting_team.score - prev_batting_score > 5:
                # gave up over 5 runs in an inning
                replace = True
            elif self.inning < 4 and runs_allowed > 5:
                # gave up over 5 runs total, and it's inside the first three innings
                replace = True
            elif 4 <= self.inning <= 6 and runs_allowed > 3:
                # gave up over 3 runs total, and it's inside the middle three innings
                replace = True
            elif self.inning > 6 and baserunner_count > 2:
                # gave up over 2 baserunners, and it's inside the final three innings (or extra innings)
                replace = True
            # elif self.inning > 6 and runs_allowed > 2:
            #     # gave up over 2 runs total, and it's inside the last three innings (or extra innings)
            #     replace = True
        else:
            # logic for bullpen
            if pitcher.num_pitch > np.random.normal(25,5)-3:
                # reached high pitch count
                if self.outs < 2:
                    # not one out away from ending the inning
                    replace = True
            elif pitcher.num_pitch - prev_pitches >= 35:
                # threw 35+ pitches in an inning
                replace = True
            elif self.outs == 3 and pitcher.outs > 1:
                # end of the inning, relief pitcher was in for more than one out
                # intended to prevent (or at least mostly) pitching more than 4 outs
                # new pitcher will start next inning
                replace = True
            elif batting_team.score - prev_batting_score > 3:
                # gave up over 3 runs
                replace = True
            elif (4 <= pitching_team.score - prev_batting_score <= 6) and batting_team.score - prev_batting_score > 2:
                # lead was 4-6 and gave up down up over 2 runs
                replace = True
            elif baserunner_count > 2 and pitching_team.score - batting_team.score < 3:
                # gave up over 2 baserunners, with the lead currently less than 3 (after baserunners have possibly scored)
                replace = True
            # elif pitching_team.score - batting_team.score < 3 and batting_team.score - prev_batting_score > 2:
            #     # lead down to less than 3 runs and has given up over 2 runs
            #     replace = True
            # elif 4 <= self.inning <= 6 and batting_team.score - prev_batting_score > 2:
            #     # gave up over 2 runs, and it's inside the middle three innings
            #     replace = True
            # elif self.inning > 6 and batting_team.score - prev_batting_score > 1:
            #     # gave up more than 1 run, and it's inside the last three innings (or extra innings)
            #     replace = True
                    
        return replace

    def simulate_inning_half(self, batting_team, pitching_team):
        end_game = False

        # Reset at start of half inning, track for pitching changes based only on this inning's events
        prev_pitches = [pitching_team.pitchers[pitching_team.pitcher_index].num_pitch][0] # pitch count at start of inning
        prev_batting_score = [batting_team.score][0] # batting team's score at start of inning
        baserunner_count = 0 # baserunners this inning, updated on hit or walk
    
        while self.outs < 3 and not end_game:
            #Set batter and pitcher
            self.strikes = 0
            self.balls = 0
            current_batter = batting_team.batters[batting_team.batter_index]
            current_pitcher = pitching_team.pitchers[pitching_team.pitcher_index]
            #Run through a pitch and update outcomes
            hit = False
            while self.strikes < 3 and self.balls < 4 and not hit and not end_game:
                baserunning_result = []
                #Get result of pitch
                event, pitch_result, swing, pitch_type, contact_cat, power = self.pitch(current_batter, current_pitcher)
                #Update stats
                if event == 'strike':
                        self.strikes += 1
                        #Record an out if strike out
                        if self.strikes == 3:
                            self.outs += 1
                            current_pitcher.outs += 1
                elif event == 'ball':
                        self.balls += 1
                        #Record a walk if ball 4
                        if self.balls == 4:
                            baserunning_result = self.move_bases('walk', batting_team, current_batter, contact_cat, power, current_pitcher)
                            baserunner_count += 1
                elif event == 'foul':
                    #Record a strike if foul and strikes < 2
                    if self.strikes < 2:
                        self.strikes += 1
                else:
                    #Move bases if a hit
                    baserunning_result = self.move_bases(event, batting_team, current_batter, contact_cat, power, current_pitcher)
                    if event != "out":
                        baserunner_count += 1
                    hit = True
                #After each pitch, update current game state
                
                self.update_event_log(current_batter, batting_team.batter_index, current_pitcher, event,
                                      swing, contact_cat, pitch_result, pitch_type, baserunning_result)
                
                #If home team scored in the final inning to go ahead, end game
                if self.inning >= 9 and self.inning_half == 'bottom' and batting_team.score > pitching_team.score:
                    end_game = True
            #Update batter
            if batting_team.batter_index < batting_team.num_batters - 1:
                batting_team.batter_index += 1
            else:
                batting_team.batter_index = 0
            #Update pitcher, possibly
            result = self.determine_pitch_change(pitching_team, batting_team, current_pitcher, prev_pitches, 
                                                 prev_batting_score, current_pitcher.runs_allowed,
                                                 baserunner_count, current_pitcher.starter)
            if result:
                if pitching_team.pitcher_index < pitching_team.num_pitchers - 1:
                    pitching_team.pitcher_index += 1
                else:
                    #TODO: We shouldn't cycle back to the first pitcher, but this is to avoid an error
                    pitching_team.pitcher_index = 0
                #Reset previous number of pitches and previous score
                prev_pitches = [pitching_team.pitchers[pitching_team.pitcher_index].num_pitch][0] # this will be 0 for new pitcher
                prev_batting_score = [batting_team.score][0] # reset batting team's score for when the new pitcher entered
                baserunner_count = 0 # reset baserunner count for new pitcher
        
    def update_event_log(self, current_batter, batter_index, current_pitcher, event, swing,
                         contact_cat, pitch_result, pitch_type, baserunning_result):
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
        self.event_log['Batter Number'].append(batter_index + 1)
        self.event_log['Pitcher'].append(current_pitcher.name)
        self.event_log['Pitch Type'].append(pitch_type)
        self.event_log['Pitch Outcome'].append(pitch_result)
        if len(baserunning_result) == 0:
            self.event_log['Baserunning Event'].append('None')
            if sum(self.bases) == 0:
                self.event_log['Baserunning Result'].append('NA')
            else:
                self.event_log['Baserunning Result'].append('safe')
        else:
            b_event = ''
            b_result = ''
            for base_event in baserunning_result:
                b_event += base_event[0] + ','
                b_result += base_event[1] + ','
            self.event_log['Baserunning Event'].append(b_event[:-1])
            self.event_log['Baserunning Result'].append(b_result[:-1])
        if self.strikes == 3:
            self.event_log['Event'].append('strikeout')
            if swing:
                self.event_log['Detailed Event'].append('strikeout swinging')
            else:
                self.event_log['Detailed Event'].append('strikeout looking')
        elif self.balls == 4:
            self.event_log['Event'].append('walk')
            self.event_log['Detailed Event'].append('walk')
        else:
            self.event_log['Event'].append(event)
            if event == "strike":
                if swing:
                    self.event_log['Detailed Event'].append('swinging strike')
                else:
                    self.event_log['Detailed Event'].append('called strike')
            elif event in ["out", "single", "double", "triple", "home_run"]:
                if event == "out":
                    contact = contact_cat.split("_")[0]
                else:
                    contact = contact_cat.replace("_", " ")
                self.event_log['Detailed Event'].append(contact + " " + event.replace("_", " "))
            else:
                self.event_log['Detailed Event'].append(event)
            
    
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

    fb_disc_df_raw = pd.read_csv("../Data/Batter_FB_Discipline.csv") # fastball
    fb_disc_df = fb_disc_df_raw.groupby("Name", as_index=False).mean(numeric_only=True).set_index("Name", drop=False)
    ch_disc_df_raw = pd.read_csv("../Data/Batter_CH_Discipline.csv") # changeup
    ch_disc_df = ch_disc_df_raw.groupby("Name", as_index=False).mean(numeric_only=True).set_index("Name", drop=False)
    cu_disc_df_raw = pd.read_csv("../Data/Batter_CU_Discipline.csv") # curveball
    cu_disc_df = cu_disc_df_raw.groupby("Name", as_index=False).mean(numeric_only=True).set_index("Name", drop=False)
    sl_disc_df_raw = pd.read_csv("../Data/Batter_SL_Discipline.csv") # slider
    sl_disc_df = sl_disc_df_raw.groupby("Name", as_index=False).mean(numeric_only=True).set_index("Name", drop=False)

    pitcher_df.fillna(0, inplace=True) # TODO is this ok? Or need to be more selective like below?
    # pitcher_df[["FA%", "FT%", "FC%", "FS%", "FO%", "SI%", "SL%", "CU%", "KC%", "EP%", "CH%", "SC%", "KN%", "UN%"]] = pitcher_df[["FA%", "FT%", "FC%", "FS%", "FO%", "SI%", "SL%", "CU%", "KC%", "EP%", "CH%", "SC%", "KN%", "UN%"]].fillna(0)
    hit_traj_df["1B"] = hit_traj_df["H"] - hit_traj_df["2B"] - hit_traj_df["3B"] - hit_traj_df["HR"]
    hit_traj_df["Out"] = hit_traj_df["AB"] - hit_traj_df["H"]
    hit_traj_df["1B%"] = hit_traj_df["1B"] / hit_traj_df["AB"]
    hit_traj_df["2B%"] = hit_traj_df["2B"] / hit_traj_df["AB"]
    hit_traj_df["3B%"] = hit_traj_df["3B"] / hit_traj_df["AB"]
    hit_traj_df["HR%"] = hit_traj_df["HR"] / hit_traj_df["AB"]
    hit_traj_df["Out%"] = hit_traj_df["Out"] / hit_traj_df["AB"]
    
    return batter_df, fb_disc_df, ch_disc_df, cu_disc_df, sl_disc_df, pitcher_df, hit_traj_df


if __name__ == '__main__':
    
    batter_df, fb_disc_df, ch_disc_df, cu_disc_df, sl_disc_df, pitcher_df, hit_traj_df = read_data()

    fb_names = set(fb_disc_df.index)
    cu_names = set(cu_disc_df.index)
    sl_names = set(sl_disc_df.index)
    ch_names = set(ch_disc_df.index)

    batter_ids = list(batter_df["PlayerId"])
    pitcher_ids = list(pitcher_df["PlayerId"])
    # TODO add pitching changes, relief pitching. Currently just selecting a starting pitcher for each side to pitch the whole game
    selected_batter_ids = np.random.choice(batter_ids, 18, replace=False) # samples w/o replacement
    selected_pitcher_ids = np.random.choice(pitcher_ids, 26, replace=False) # samples w/o replacemnet
    
    batter_data = [ 
        (lambda row:
            {'name': row["Name"],
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
             'strike_prob': row["Zone%"], # prob inside zone, not of being a strike bc of zone/swing/foul
             'starter': False # Whether this pitcher is a starter, is updated after assigned to teams
            }
        )(pitcher_df[pitcher_df["PlayerId"] == pid].iloc[0])
            for pid in selected_pitcher_ids
    ]
    
    batters1 = [Batter(batter) for batter in batter_data[0:9]]
    batters2 = [Batter(batter) for batter in batter_data[9:]]

    
    pitchers1 = [Pitcher(pitcher) for pitcher in pitcher_data[0:13]]
    #Label first pitcher as the starter
    pitchers1[0].starter = True

    
    pitchers2 = [Pitcher(pitcher) for pitcher in pitcher_data[13:]]
    #Label first pitcher as the starter
    pitchers2[0].starter = True

        
    team1 = Team(batters1,pitchers1)
    team2 = Team(batters2,pitchers2)
    
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
