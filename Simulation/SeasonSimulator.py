"""
Season Simulator for Baseball - 2025 MLB Schedule
Runs a complete 162-game season for all teams
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from baseball_sim_classes import Game, Team, Batter, Pitcher, read_data, normalize_values, perturb_values


class SeasonStats:
    """Stores cumulative season-long batting & pitching statistics."""
    
    def __init__(self):
        self.batter_stats = {}
        self.pitcher_stats = {}
        self.team_stats = {}

    def register_batter(self, batter_name, team):
        if batter_name not in self.batter_stats:
            self.batter_stats[batter_name] = {
                "Team": team,
                "PA": 0, "AB": 0, "H": 0, "1B": 0, "2B": 0,
                "3B": 0, "HR": 0, "BB": 0, "SO": 0, "R": 0, "RBI": 0,
                "SB": 0, "CS": 0  # Stolen bases, Caught stealing
            }

    def register_pitcher(self, pitcher_name, team):
        if pitcher_name not in self.pitcher_stats:
            self.pitcher_stats[pitcher_name] = {
                "Team": team,
                "BF": 0, "Outs": 0, "H": 0, "R": 0, "BB": 0, "SO": 0, "W": 0, "L": 0
            }
    
    def register_team(self, team_abbr):
        if team_abbr not in self.team_stats:
            self.team_stats[team_abbr] = {"W": 0, "L": 0, "R": 0, "RA": 0}


class SeasonSimulator:
    """Runs a full 162-game MLB season."""
    
    def __init__(self, teams_dict):
        self.teams_dict = teams_dict
        self.season_stats = SeasonStats()
        self.schedule = []
        
        # Register all teams
        for abbr in teams_dict.keys():
            self.season_stats.register_team(abbr)

    def load_schedule_from_file(self, schedule_file_path):
        """Load the 2025 MLB schedule from a CSV file."""
        print(f"\nLoading schedule from {schedule_file_path}...")
        
        try:
            # Read exactly 8 columns
            schedule_df = pd.read_csv(schedule_file_path, header=None, 
                                     names=['Date', 'Day', 'LocalTime', 'ETTime', 'Matchup', 'AwayTeam', 'HomeTeam', 'Venue'],
                                     usecols=[0,1,2,3,4,5,6,7])
            
            # Team name mapping
            team_mapping = {
                'Dodgers': 'LAD', 'Cubs': 'CHC', 'Mets': 'NYM', 'Astros': 'HOU',
                'Orioles': 'BAL', 'Blue Jays': 'TOR', 'Twins': 'MIN', 'Cardinals': 'STL',
                'D-backs': 'ARI', 'Tigers': 'DET', 'Athletics': 'OAK', 'Mariners': 'SEA',
                'Pirates': 'PIT', 'Marlins': 'MIA', 'Phillies': 'PHI', 'Nationals': 'WSN',
                'Braves': 'ATL', 'Padres': 'SDP', 'Red Sox': 'BOS', 'Rangers': 'TEX',
                'Giants': 'SFG', 'Reds': 'CIN', 'Guardians': 'CLE', 'Royals': 'KCR',
                'Angels': 'LAA', 'White Sox': 'CHW', 'Brewers': 'MIL', 'Yankees': 'NYY',
                'Rockies': 'COL', 'Rays': 'TBR'
            }
            
            schedule = []
            for _, row in schedule_df.iterrows():
                away = team_mapping.get(row['AwayTeam'])
                home = team_mapping.get(row['HomeTeam'])
                
                if away and home and away in self.teams_dict and home in self.teams_dict:
                    schedule.append((home, away))
            
            self.schedule = schedule
            
            # Verify game counts
            games_count = {team: 0 for team in self.teams_dict.keys()}
            for home, away in schedule:
                games_count[home] += 1
                games_count[away] += 1
            
            print(f"✓ Loaded {len(schedule)} games")
            print(f"Games per team: min={min(games_count.values())}, max={max(games_count.values())}, avg={np.mean(list(games_count.values())):.1f}")
            
            return schedule
            
        except Exception as e:
            print(f"ERROR loading schedule: {e}")
            raise

    def run_season(self, schedule_file, verbose=True):
        """Run all games in the schedule."""
        
        # Load schedule
        self.load_schedule_from_file(schedule_file)

        print("\n" + "="*70)
        print("STARTING 2025 SEASON SIMULATION")
        print("="*70)
        
        for idx, (home_abbr, away_abbr) in enumerate(tqdm(self.schedule, desc="Simulating Games")):
            if verbose and (idx + 1) % 500 == 0:
                print(f"\n{'='*70}")
                print(f"Progress: {idx+1}/{len(self.schedule)} games completed")
                print(f"{'='*70}")
                self._print_standings()

            home_team = self.teams_dict[home_abbr]
            away_team = self.teams_dict[away_abbr]
            
            # Reset for new game
            self._reset_team_for_game(home_team)
            self._reset_team_for_game(away_team)

            # Play game
            game = Game(home_team, away_team)
            winner_name = game.run_game()

            # Update records and stats
            if winner_name == home_team.name:
                self.season_stats.team_stats[home_abbr]["W"] += 1
                self.season_stats.team_stats[away_abbr]["L"] += 1
            else:
                self.season_stats.team_stats[away_abbr]["W"] += 1
                self.season_stats.team_stats[home_abbr]["L"] += 1
            
            # Update team runs
            self.season_stats.team_stats[home_abbr]["R"] += home_team.score
            self.season_stats.team_stats[home_abbr]["RA"] += away_team.score
            self.season_stats.team_stats[away_abbr]["R"] += away_team.score
            self.season_stats.team_stats[away_abbr]["RA"] += home_team.score

            # Update player stats
            self.update_season_stats(game, home_abbr, away_abbr, home_team, away_team)

        print("\n" + "="*70)
        print("FINAL STANDINGS - 2025 SEASON")
        print("="*70)
        self._print_standings()
        
        return self.season_stats

    def _reset_team_for_game(self, team):
        """Reset team state for a new game."""
        team.score = 0
        team.batter_index = 0
        team.pitcher_index = 0
        
        for pitcher in team.pitchers:
            pitcher.num_pitch = 0
            pitcher.num_batters = 0
            pitcher.runs_allowed = 0
            pitcher.hits_allowed = 0
            pitcher.outs = 0

    def _print_standings(self):
        """Print current standings."""
        standings = []
        for abbr, stats in self.season_stats.team_stats.items():
            w, l = stats["W"], stats["L"]
            pct = w / (w + l) if (w + l) > 0 else 0
            standings.append((abbr, w, l, pct, stats["R"], stats["RA"]))
        
        standings.sort(key=lambda x: x[3], reverse=True)
        
        print(f"\n{'Team':<6} {'W':>3} {'L':>3} {'PCT':>6} {'R':>5} {'RA':>5}")
        print("-" * 35)
        
        for abbr, w, l, pct, r, ra in standings[:15]:
            print(f"{abbr:<6} {w:>3} {l:>3} {pct:>6.3f} {r:>5} {ra:>5}")
        
        if len(standings) > 15:
            print(f"... (showing top 15 of {len(standings)})")

    def update_season_stats(self, game, home_abbr, away_abbr, home_team, away_team):
        """Update season stats from a game by using the game's box score."""
        
        # Get the box score that the game already calculated
        box_score = game.get_box_score()
        
        # Update batting stats
        for player_name, game_stats in box_score["batting"].items():
            # Determine which team this batter is on
            # Check home team first
            batter_team = None
            for batter in home_team.batters:
                if batter.name == player_name:
                    batter_team = home_abbr
                    break
            if not batter_team:
                for batter in away_team.batters:
                    if batter.name == player_name:
                        batter_team = away_abbr
                        break
            
            if not batter_team:
                continue
            
            # Register and update
            self.season_stats.register_batter(player_name, batter_team)
            season_stats = self.season_stats.batter_stats[player_name]
            
            # Add all the stats from this game
            for stat_key in ["PA", "AB", "H", "1B", "2B", "3B", "HR", "BB", "SO", "R", "RBI"]:
                if stat_key in game_stats:
                    season_stats[stat_key] += game_stats[stat_key]
        
        # Update pitching stats
        for player_name, game_stats in box_score["pitching"].items():
            # Determine which team this pitcher is on
            pitcher_team = None
            for pitcher in home_team.pitchers:
                if pitcher.name == player_name:
                    pitcher_team = home_abbr
                    break
            if not pitcher_team:
                for pitcher in away_team.pitchers:
                    if pitcher.name == player_name:
                        pitcher_team = away_abbr
                        break
            
            if not pitcher_team:
                continue
            
            # Register and update
            self.season_stats.register_pitcher(player_name, pitcher_team)
            season_stats = self.season_stats.pitcher_stats[player_name]
            
            # Add stats, converting IP to Outs
            for stat_key in ["BF", "H", "BB", "SO"]:
                if stat_key in game_stats:
                    season_stats[stat_key] += game_stats[stat_key]
            
            # Handle IP -> Outs conversion
            if "IP" in game_stats:
                # Convert IP back to outs (IP * 3)
                outs = int(round(game_stats["IP"] * 3))
                season_stats["Outs"] += outs
            
            # Add runs and ER
            if "ER" in game_stats:
                season_stats["R"] += game_stats["ER"]  # We use R since we don't track errors
        
        # Now handle stolen bases from event log
        event_df = pd.DataFrame(game.event_log)
        
        for _, row in event_df.iterrows():
            # Determine batting team
            if row["Inning Half"] == "top":
                batting_team = away_team
                batting_team_abbr = away_abbr
            else:
                batting_team = home_team
                batting_team_abbr = home_abbr
            
            # Check for stolen base attempts
            baserunning_event = str(row.get("Baserunning Event", "None"))
            baserunning_result = str(row.get("Baserunning Result", "NA"))
            
            if "steal" in baserunning_event.lower():
                # Get the baserunners to find who attempted the steal
                baserunners = row.get("Baserunners", [0, 0, 0])
                
                # Parse which base the steal was from
                runner_name = None
                if "first to second" in baserunning_event.lower():
                    runner_name = baserunners[0] if isinstance(baserunners, list) and len(baserunners) > 0 else None
                elif "second to third" in baserunning_event.lower():
                    runner_name = baserunners[1] if isinstance(baserunners, list) and len(baserunners) > 1 else None
                elif "third to home" in baserunning_event.lower():
                    runner_name = baserunners[2] if isinstance(baserunners, list) and len(baserunners) > 2 else None
                
                if runner_name and runner_name != 0:
                    # Register runner if needed
                    if runner_name not in self.season_stats.batter_stats:
                        self.season_stats.register_batter(runner_name, batting_team_abbr)
                    
                    runner_stats = self.season_stats.batter_stats[runner_name]
                    if "safe" in baserunning_result.lower():
                        runner_stats["SB"] += 1
                    elif "thrown out" in baserunning_result.lower():
                        runner_stats["CS"] += 1

    def export_stats(self, prefix="season_2025"):
        """Export season statistics to CSV files."""
        print("\n" + "="*70)
        print("EXPORTING STATISTICS")
        print("="*70)
        
        # Batting stats with advanced metrics
        bat_df = pd.DataFrame(self.season_stats.batter_stats).T
        if len(bat_df) > 0:
            # Filter out players with no plate appearances
            bat_df = bat_df[bat_df["PA"] > 0].copy()
            
            bat_df["AVG"] = (bat_df["H"] / bat_df["AB"].replace(0, pd.NA)).fillna(0).round(3)
            bat_df["OBP"] = ((bat_df["H"] + bat_df["BB"]) / bat_df["PA"]).fillna(0).round(3)
            
            # Calculate Total Bases: 1B + 2×2B + 3×3B + 4×HR
            bat_df["TB"] = bat_df["1B"] + (2 * bat_df["2B"]) + (3 * bat_df["3B"]) + (4 * bat_df["HR"])
            bat_df["SLG"] = (bat_df["TB"] / bat_df["AB"].replace(0, pd.NA)).fillna(0).round(3)
            bat_df["OPS"] = (bat_df["OBP"] + bat_df["SLG"]).round(3)
            
            # Sort by OPS
            bat_df = bat_df.sort_values("OPS", ascending=False)
            
            # Select columns to export
            bat_df = bat_df[["Team", "PA", "AB", "H", "1B", "2B", "3B", "HR", "BB", "SO", "SB", "CS", "R", "RBI", "AVG", "OBP", "SLG", "OPS"]]
        
        # Pitching stats
        pit_df = pd.DataFrame(self.season_stats.pitcher_stats).T
        if len(pit_df) > 0:
            # Filter out pitchers with no outs recorded
            pit_df = pit_df[pit_df["Outs"] > 0].copy()
            
            # Convert outs to innings pitched (IP = Outs / 3)
            pit_df["IP"] = (pit_df["Outs"] / 3).round(1)
            
            # RA (Runs Allowed per 9)
            pit_df["RA9"] = ((pit_df["R"] * 9) / pit_df["IP"]).fillna(0).round(2)
            pit_df["WHIP"] = ((pit_df["H"] + pit_df["BB"]) / pit_df["IP"]).fillna(0).round(2)
            pit_df["K/9"] = ((pit_df["SO"] * 9) / pit_df["IP"]).fillna(0).round(2)
            pit_df = pit_df.sort_values("IP", ascending=False)
            
            # Select columns (drop Outs since we have IP now)
            pit_df = pit_df[["Team", "W", "L", "IP", "H", "R", "BB", "SO", "BF", "RA9", "WHIP", "K/9"]]
        
        # Team standings
        standings_data = []
        for abbr, stats in self.season_stats.team_stats.items():
            w, l = stats["W"], stats["L"]
            pct = w / (w + l) if (w + l) > 0 else 0
            standings_data.append({
                "Team": abbr,
                "W": w,
                "L": l,
                "PCT": round(pct, 3),
                "R": stats["R"],
                "RA": stats["RA"],
                "RD": stats["R"] - stats["RA"]  # Run differential
            })
        standings_df = pd.DataFrame(standings_data).sort_values("PCT", ascending=False)

        # Export
        bat_df.to_csv(prefix + "_batting.csv")
        pit_df.to_csv(prefix + "_pitching.csv")
        standings_df.to_csv(prefix + "_standings.csv", index=False)

        print(f"✓ {prefix}_batting.csv")
        print(f"✓ {prefix}_pitching.csv")
        print(f"✓ {prefix}_standings.csv")
        
        # Show top performers
        print("\n" + "="*70)
        print("TOP PERFORMERS")
        print("="*70)
        
        if len(bat_df) > 0:
            print("\nTop 10 Hitters by OPS (min 100 PA):")
            top_batters = bat_df[bat_df["PA"] >= 100].head(10)
            print(top_batters[["Team", "PA", "AVG", "OBP", "SLG", "OPS"]].to_string())
            
            print("\nTop 10 Base Stealers:")
            top_stealers = bat_df.sort_values("SB", ascending=False).head(10)
            print(top_stealers[["Team", "SB", "CS", "AVG"]].to_string())
        
        if len(pit_df) > 0:
            print("\nTop 10 Pitchers by IP:")
            print(pit_df[["Team", "W", "L", "IP", "RA9", "WHIP"]].head(10).to_string())


def load_teams_from_data():
    """Load all teams from the baseball data files."""
    print("="*70)
    print("LOADING BASEBALL DATA")
    print("="*70)
    
    batter_df, fb_disc_df, ch_disc_df, cu_disc_df, sl_disc_df, pitcher_df, hit_traj_df, _ = read_data()
    
    # Forward-fill team names
    batter_df['Team'] = batter_df['Team'].replace('- - -', pd.NA).ffill()
    pitcher_df['Team'] = pitcher_df['Team'].replace('- - -', pd.NA).ffill()
    batter_df = batter_df.dropna(subset=['Team'])
    pitcher_df = pitcher_df.dropna(subset=['Team'])
    
    # Consolidate variants
    team_name_map = {'ATH': 'OAK'}
    batter_df['Team'] = batter_df['Team'].replace(team_name_map)
    pitcher_df['Team'] = pitcher_df['Team'].replace(team_name_map)
    
    # Get reference sets
    fb_names = set(fb_disc_df.index)
    cu_names = set(cu_disc_df.index)
    sl_names = set(sl_disc_df.index)
    ch_names = set(ch_disc_df.index)
    
    # Organize by team
    teams = {}
    base_velocity_dist = {95: 0.5, 100: 0.5}
    base_movement_prob = {'straight': 0.25, 'down': 0.25, 'side': 0.25, 'fade': 0.25}
    
    print("\nCreating team rosters...")
    
    # Process batters
    for idx, row in batter_df.iterrows():
        team_abbr = row["Team"]
        if team_abbr not in teams:
            teams[team_abbr] = {"batters": [], "pitchers": [], "name": team_abbr}
        
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
        teams[team_abbr]["batters"].append(Batter(batter_data))
    
    # Process pitchers
    for idx, row in pitcher_df.iterrows():
        team_abbr = row["Team"]
        if team_abbr not in teams:
            teams[team_abbr] = {"batters": [], "pitchers": [], "name": team_abbr}
        
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
        teams[team_abbr]["pitchers"].append(Pitcher(pitcher_data))
    
    # Mark first pitcher as starter
    for team_abbr in teams:
        if teams[team_abbr]["pitchers"]:
            teams[team_abbr]["pitchers"][0].starter = True
    
    # Create Team objects
    teams_dict = {}
    for abbr, roster in teams.items():
        if len(roster["batters"]) >= 9 and len(roster["pitchers"]) >= 5:
            team_obj = Team(roster["batters"], roster["pitchers"], abbr)
            teams_dict[abbr] = team_obj
    
    print(f"\n✓ Loaded {len(teams_dict)} teams")
    
    return teams_dict


if __name__ == '__main__':
    import sys
    import time
    
    # Get number of simulations from command line, default to 50
    num_simulations = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    
    print("="*70)
    print(f"RUNNING {num_simulations} SEASON SIMULATIONS")
    print("="*70)
    
    # Load teams once (they're the same for all simulations)
    teams_dict = load_teams_from_data()
    schedule_file_path = "../Data/2025_mlb_schedule.csv"
    
    # Storage for aggregated stats across all simulations
    all_batter_stats = {}  # {player_name: {stat: [values from each sim]}}
    all_pitcher_stats = {}
    all_team_stats = {}
    
    # Run multiple simulations
    for sim_num in range(num_simulations):
        seed = (int(time.time()) + sim_num) % (2**32)  # Keep seed within valid range
        np.random.seed(seed)
        
        print("\n" + "="*70)
        print(f"SIMULATION {sim_num + 1} of {num_simulations} (seed: {seed})")
        print("="*70)
        
        # Create new simulator for this run
        sim = SeasonSimulator(teams_dict)
        
        # Run season
        sim.run_season(schedule_file=schedule_file_path, verbose=False)
        
        # Collect stats from this simulation
        # Batting stats
        for player_name, stats in sim.season_stats.batter_stats.items():
            if player_name not in all_batter_stats:
                all_batter_stats[player_name] = {
                    "Team": stats["Team"],
                    "PA": [], "AB": [], "H": [], "1B": [], "2B": [],
                    "3B": [], "HR": [], "BB": [], "SO": [], "SB": [], "CS": [],
                    "R": [], "RBI": []
                }
            for stat_key in ["PA", "AB", "H", "1B", "2B", "3B", "HR", "BB", "SO", "SB", "CS", "R", "RBI"]:
                all_batter_stats[player_name][stat_key].append(stats[stat_key])
        
        # Pitching stats
        for player_name, stats in sim.season_stats.pitcher_stats.items():
            if player_name not in all_pitcher_stats:
                all_pitcher_stats[player_name] = {
                    "Team": stats["Team"],
                    "BF": [], "Outs": [], "H": [], "R": [], "BB": [], "SO": [], "W": [], "L": []
                }
            for stat_key in ["BF", "Outs", "H", "R", "BB", "SO", "W", "L"]:
                all_pitcher_stats[player_name][stat_key].append(stats[stat_key])
        
        # Team stats
        for team_abbr, stats in sim.season_stats.team_stats.items():
            if team_abbr not in all_team_stats:
                all_team_stats[team_abbr] = {"W": [], "L": [], "R": [], "RA": []}
            for stat_key in ["W", "L", "R", "RA"]:
                all_team_stats[team_abbr][stat_key].append(stats[stat_key])
        
        print(f"✓ Simulation {sim_num + 1} complete")
    
    # Now calculate averages
    print("\n" + "="*70)
    print("CALCULATING AVERAGES ACROSS ALL SIMULATIONS")
    print("="*70)
    
    # Average batting stats
    avg_batter_stats = {}
    for player_name, stats in all_batter_stats.items():
        avg_batter_stats[player_name] = {"Team": stats["Team"]}
        for stat_key in ["PA", "AB", "H", "1B", "2B", "3B", "HR", "BB", "SO", "SB", "CS", "R", "RBI"]:
            avg_batter_stats[player_name][stat_key] = np.mean(stats[stat_key])
    
    # Average pitching stats
    avg_pitcher_stats = {}
    for player_name, stats in all_pitcher_stats.items():
        avg_pitcher_stats[player_name] = {"Team": stats["Team"]}
        for stat_key in ["BF", "Outs", "H", "R", "BB", "SO", "W", "L"]:
            avg_pitcher_stats[player_name][stat_key] = np.mean(stats[stat_key])
    
    # Average team stats
    avg_team_stats = {}
    for team_abbr, stats in all_team_stats.items():
        avg_team_stats[team_abbr] = {}
        for stat_key in ["W", "L", "R", "RA"]:
            avg_team_stats[team_abbr][stat_key] = np.mean(stats[stat_key])
    
    # Export averaged stats
    print("\nExporting averaged statistics...")
    
    # Batting
    bat_df = pd.DataFrame(avg_batter_stats).T
    if len(bat_df) > 0:
        bat_df = bat_df[bat_df["PA"] > 0].copy()
        bat_df["AVG"] = (bat_df["H"] / bat_df["AB"].replace(0, pd.NA)).fillna(0).round(3)
        bat_df["OBP"] = ((bat_df["H"] + bat_df["BB"]) / bat_df["PA"]).fillna(0).round(3)
        bat_df["TB"] = bat_df["1B"] + (2 * bat_df["2B"]) + (3 * bat_df["3B"]) + (4 * bat_df["HR"])
        bat_df["SLG"] = (bat_df["TB"] / bat_df["AB"].replace(0, pd.NA)).fillna(0).round(3)
        bat_df["OPS"] = (bat_df["OBP"] + bat_df["SLG"]).round(3)
        bat_df = bat_df.sort_values("OPS", ascending=False)
        
        # Round counting stats to 1 decimal
        for col in ["PA", "AB", "H", "1B", "2B", "3B", "HR", "BB", "SO", "SB", "CS", "R", "RBI"]:
            bat_df[col] = bat_df[col].round(1)
        
        bat_df = bat_df[["Team", "PA", "AB", "H", "1B", "2B", "3B", "HR", "BB", "SO", "SB", "CS", "R", "RBI", "AVG", "OBP", "SLG", "OPS"]]
        bat_df.to_csv(f"avg_{num_simulations}sims_batting.csv")
    
    # Pitching
    pit_df = pd.DataFrame(avg_pitcher_stats).T
    if len(pit_df) > 0:
        pit_df = pit_df[pit_df["Outs"] > 0].copy()
        pit_df["IP"] = (pit_df["Outs"] / 3).round(1)
        pit_df["RA9"] = ((pit_df["R"] * 9) / pit_df["IP"]).fillna(0).round(2)
        pit_df["WHIP"] = ((pit_df["H"] + pit_df["BB"]) / pit_df["IP"]).fillna(0).round(2)
        pit_df["K/9"] = ((pit_df["SO"] * 9) / pit_df["IP"]).fillna(0).round(2)
        pit_df = pit_df.sort_values("IP", ascending=False)
        
        # Round counting stats
        for col in ["BF", "H", "R", "BB", "SO"]:
            pit_df[col] = pit_df[col].round(1)
        for col in ["W", "L"]:
            pit_df[col] = pit_df[col].round(1)
        
        pit_df = pit_df[["Team", "W", "L", "IP", "H", "R", "BB", "SO", "BF", "RA9", "WHIP", "K/9"]]
        pit_df.to_csv(f"avg_{num_simulations}sims_pitching.csv")
    
    # Team standings
    standings_data = []
    for abbr, stats in avg_team_stats.items():
        w = stats["W"]
        l = stats["L"]
        pct = w / (w + l) if (w + l) > 0 else 0
        standings_data.append({
            "Team": abbr,
            "W": round(w, 1),
            "L": round(l, 1),
            "PCT": round(pct, 3),
            "R": round(stats["R"], 1),
            "RA": round(stats["RA"], 1),
            "RD": round(stats["R"] - stats["RA"], 1)
        })
    standings_df = pd.DataFrame(standings_data).sort_values("PCT", ascending=False)
    standings_df.to_csv(f"avg_{num_simulations}sims_standings.csv", index=False)
    
    print(f"✓ avg_{num_simulations}sims_batting.csv")
    print(f"✓ avg_{num_simulations}sims_pitching.csv")
    print(f"✓ avg_{num_simulations}sims_standings.csv")
    
    # Show top performers
    print("\n" + "="*70)
    print(f"TOP PERFORMERS (AVERAGED OVER {num_simulations} SIMULATIONS)")
    print("="*70)
    
    if len(bat_df) > 0:
        print("\nTop 10 Hitters by OPS (min 100 PA avg):")
        top_batters = bat_df[bat_df["PA"] >= 100].head(10)
        print(top_batters[["Team", "PA", "AVG", "OBP", "SLG", "OPS"]].to_string())
    
    if len(pit_df) > 0:
        print("\nTop 10 Pitchers by IP (avg):")
        print(pit_df[["Team", "W", "L", "IP", "RA9", "WHIP"]].head(10).to_string())
    
    print("\n" + "="*70)
    print(f"✓ COMPLETED {num_simulations} SEASON SIMULATIONS!")
    print("="*70)