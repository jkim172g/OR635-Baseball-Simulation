
import pandas as pd
import numpy as np
import random



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
        self.team1_score = 0
        self.team2_score = 0
        self.bases = [False, False, False]
        self.team1_batter_index = 0
        self.team2_batter_index = 0
        self.team1_pitcher_index = 0
        self.team2_pitcher_index = 0
        
        self.team1 = team1
        self.team2 = team2
    
    def pitch(self, batter, pitcher):
        # Get pitch details
        pitch_type = random.choices(list(pitcher.pitch_type_prob.keys()), list(pitcher.pitch_type_prob.values()))[0]
        pitch_velocity = random.choices(list(pitcher.velocity_dist.keys()), list(pitcher.velocity_dist.values()))[0]
        pitch_movement = random.choices(list(pitcher.movement_prob.keys()), list(pitcher.movement_prob.values()))[0]
        
        
        #Calculate result
        
        strike = True if np.random.uniform() < pitcher.strike_prob else False
        if strike:
            #If strike, check if batter swings
            swing = True if np.random.uniform() < batter.swing_prob else False
            if swing:
                #If batter swings, check if contact was made
                contact = True if np.random.uniform() < batter.contact_prob else False
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
            swing = True if np.random.uniform() < batter.swing_prob else False
            if swing:
                #If batter swings, check if contact was made
                contact = True if np.random.uniform() < batter.contact_prob else False
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
        
    
    def move_bases(self, action):
        #Put logic here for moving bases based on the action
        #Calculate score as needed
        #This logic may be messy - trying to think of a more elegant way to do it than a million if statements
        ## How do we determine whether a player already on base was out?
        
        pass
    
    def simulate_inning_half(self, batting_team, batter_up, pitching_team, pitcher_up):
        
        
        
        
        while self.outs < 3:
            #Set batter and pitcher
            current_batter = batting_team.batters[batter_up]
            current_pitcher = pitching_team.pitcher[pitcher_up]
            #Run through a pitch and update outcomes
            hit = False
            while self.strikes < 3 and self.balls < 4 and not hit:
                #Get result of pitch
                pitch_result = self.pitch(self,current_batter, current_pitcher)
                #Update stats
                if pitch_result == 'strike':
                        self.strikes += 1
                if pitch_result == 'ball':
                        self.balls += 1
                else:
                    self.move_bases(self,pitch_result)
                    hit = True
            # Add an out or walk depending on result of at-bat
            if self.strikes == 3:
                self.outs += 1
            if self.balls ==4:
                self.move_bases(self,'walk')
            batter_up +=1
            
            
            
                    
            
    
    def play_ball(self):
        
        #Loop through innings
        while self.inning <= 9:
            if self.inning_half == 'top':
                #If top of the inning, team1 is batting and team2 is pitching
                batter_index = self.team1_batter_index
                pitcher_index = self.team2_pitcher_index
                self.simulate_inning_half(self, team1, batter_index, team2, pitcher_index)    
            else:
                #If bottom of the inning, team2 is batting and team1 is pitching
                batter_index = self.team2_batter_index
                pitcher_index = self.team1_pitcher_index
                self.simulate_inning_half(self, team2, batter_index, team1, pitcher_index) 
     
    

if __name__ == '__main__':
    batter1_data= {'name':'Joe', 'team':'Nationals', 'swing_prob':0.5, 'contact_prob':0.7,
                   'outcome_prob':{'single':0.25,
                                   'double':0.1,
                                   'triple':0.05,
                                   'home_run':0.05,
                                   'ground_out':0.3,
                                   'fly_out':0.3}}
    pitcher1_data= {'name':'Pete', 'team':'Nationals', 
               'pitch_type_prob':{'fastball':0.25,
                                  'curveball':0.25,
                                  'slider':0.25,
                                  'changeup':0.25},
               'velocity_dist':{95:0.5, 
                                100:0.5},
               'movement_prob':{'straight':0.25,
                                  'down':0.25,
                                  'side':0.25,
                                  'fade':0.25},
               'strike_prob':0.5}
        
    batter1 = Batter(batter1_data)  
    pitcher1 = Pitcher(pitcher1_data)
    
    team1 = Team([batter1],[pitcher1])
    team2 = Team([batter1],[pitcher1])
        
        
        
        
        
        
        
        