import pandas as pd

class SeasonStats:
    """
    Stores cumulative season-long batting & pitching statistics.
    """
    def __init__(self):
        self.batter_stats = {}   # {player_id: stat_dict}
        self.pitcher_stats = {}  # {player_id: stat_dict}

    def register_batter(self, batter_id):
        if batter_id not in self.batter_stats:
            self.batter_stats[batter_id] = {
                "PA": 0,
                "AB": 0,
                "H": 0,
                "1B": 0,
                "2B": 0,
                "3B": 0,
                "HR": 0,
                "BB": 0,
                "SO": 0,
                "R": 0,
                "RBI": 0
            }

    def register_pitcher(self, pitcher_id):
        if pitcher_id not in self.pitcher_stats:
            self.pitcher_stats[pitcher_id] = {
                "BF": 0,
                "IP": 0.0,
                "H": 0,
                "ER": 0,
                "BB": 0,
                "SO": 0
            }


class SeasonSimulator:
    """
    Runs a full season using the existing Game class.
    """
    def __init__(self, teams, game_class, season_stats=None):
        self.teams = teams
        self.game_class = game_class  # Inject your existing Game class
        self.season_stats = season_stats if season_stats else SeasonStats()
        self.schedule = []

    def build_schedule(self, games_per_matchup=3):
        """
        Generates a simple round-robin schedule.
        Each pair of teams plays a home and away series.
        """
        schedule = []
        num_teams = len(self.teams)

        for i in range(num_teams):
            for j in range(i + 1, num_teams):
                for _ in range(games_per_matchup):
                    schedule.append((self.teams[i], self.teams[j]))  # i home
                    schedule.append((self.teams[j], self.teams[i]))  # j home

        self.schedule = schedule
        return schedule

    def run_season(self, games_per_matchup=3, verbose=True):
        """
        Runs all games in the schedule.
        """
        if not self.schedule:
            self.build_schedule(games_per_matchup)

        for idx, (home, away) in enumerate(self.schedule):
            if verbose:
                print(f"Simulating Game {idx+1}/{len(self.schedule)}: {home.team_name} vs {away.team_name}")

            game = self.game_class(home, away)
            game.run_game()

            self.update_season_stats(game)

        return self.season_stats

    def update_season_stats(self, game):
        """
        Incorporates one game's stats into the season totals.
        Assumes the Game class has get_box_score().
        """
        box = game.get_box_score()

        # Batters
        for player_id, stats in box["batting"].items():
            self.season_stats.register_batter(player_id)
            for key in stats:
                self.season_stats.batter_stats[player_id][key] += stats[key]

        # Pitchers
        for player_id, stats in box["pitching"].items():
            self.season_stats.register_pitcher(player_id)
            for key in stats:
                self.season_stats.pitcher_stats[player_id][key] += stats[key]

    def export_stats(self, prefix="season"):
        """
        Exports season statistics to CSV files.
        """
        bat_df = pd.DataFrame(self.season_stats.batter_stats).T
        pit_df = pd.DataFrame(self.season_stats.pitcher_stats).T

        bat_df.to_csv(prefix + "_batting.csv")
        pit_df.to_csv(prefix + "_pitching.csv")

        print("Season statistics exported:")
        print(prefix + "_batting.csv")
        print(prefix + "_pitching.csv")
