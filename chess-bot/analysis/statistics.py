import time
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict

class Statistics:
    """
    Tracks and analyzes chess bot performance statistics.
    Monitors game results, playing strength, and improvement over time.
    """
    
    def __init__(self, stats_file="data/statistics.json"):
        self.stats_file = stats_file
        self.stats = self._load_stats()
        
        # Create directory if needed
        os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    
    def _load_stats(self):
        """Load statistics from file."""
        try:
            with open(self.stats_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._create_empty_stats()
    
    def _create_empty_stats(self):
        """Create empty statistics structure."""
        return {
            'games': {
                'total_played': 0,
                'wins': 0,
                'draws': 0,
                'losses': 0,
                'results_by_color': {
                    'white': {'wins': 0, 'draws': 0, 'losses': 0},
                    'black': {'wins': 0, 'draws': 0, 'losses': 0}
                }
            },
            'performance': {
                'current_elo': 1500,
                'peak_elo': 1500,
                'elo_history': [],
                'win_rate': 0.5,
                'average_game_length': 0,
                'accuracy_average': 0.0
            },
            'time_management': {
                'average_move_time': 0.0,
                'time_trouble_games': 0,
                'fastest_game': float('inf'),
                'longest_game': 0
            },
            'tactical': {
                'tactics_solved': 0,
                'tactics_attempted': 0,
                'blunders_per_game': 0.0,
                'missed_wins': 0,
                'brilliant_moves': 0
            },
            'openings': {
                'most_played': {},
                'best_performing': {},
                'win_rates_by_opening': {}
            },
            'opponents': {
                'unique_opponents': 0,
                'strongest_defeated': 0,
                'head_to_head': {}
            },
            'training': {
                'positions_analyzed': 0,
                'games_studied': 0,
                'training_time': 0.0,
                'improvement_rate': 0.0
            },
            'milestones': {
                'first_win': None,
                'rating_milestones': {},
                'achievement_dates': {}
            },
            'last_updated': datetime.now().isoformat()
        }
    
    def record_game_result(self, result, color, opponent_elo=None, game_data=None):
        """
        Record the result of a game.
        
        Args:
            result: 'win', 'draw', or 'loss'
            color: 'white' or 'black'
            opponent_elo: Opponent's ELO rating
            game_data: Additional game data
        """
        # Update basic game stats
        self.stats['games']['total_played'] += 1
        self.stats['games'][result + 's'] += 1
        self.stats['games']['results_by_color'][color][result + 's'] += 1
        
        # Update ELO if opponent rating provided
        if opponent_elo:
            self._update_elo(result, opponent_elo)
        
        # Update win rate
        total_games = self.stats['games']['total_played']
        wins = self.stats['games']['wins']
        draws = self.stats['games']['draws']
        self.stats['performance']['win_rate'] = (wins + draws * 0.5) / total_games
        
        # Record first win milestone
        if result == 'win' and not self.stats['milestones']['first_win']:
            self.stats['milestones']['first_win'] = datetime.now().isoformat()
        
        # Update game-specific data
        if game_data:
            self._update_game_data(game_data)
        
        self._save_stats()
    
    def _update_elo(self, result, opponent_elo):
        """Update ELO rating based on game result."""
        current_elo = self.stats['performance']['current_elo']
        
        # Calculate expected score
        expected_score = 1 / (1 + 10**((opponent_elo - current_elo) / 400))
        
        # Convert result to score
        if result == 'win':
            actual_score = 1.0
        elif result == 'draw':
            actual_score = 0.5
        else:
            actual_score = 0.0
        
        # K-factor (higher for new players)
        total_games = self.stats['games']['total_played']
        if total_games < 30:
            k_factor = 40
        elif current_elo < 2100:
            k_factor = 20
        else:
            k_factor = 10
        
        # Update ELO
        new_elo = current_elo + k_factor * (actual_score - expected_score)
        self.stats['performance']['current_elo'] = round(new_elo)
        
        # Update peak ELO
        if new_elo > self.stats['performance']['peak_elo']:
            self.stats['performance']['peak_elo'] = round(new_elo)
        
        # Record ELO history
        self.stats['performance']['elo_history'].append({
            'date': datetime.now().isoformat(),
            'elo': round(new_elo),
            'opponent_elo': opponent_elo,
            'result': result
        })
        
        # Check for rating milestones
        self._check_rating_milestones(new_elo)
    
    def _check_rating_milestones(self, new_elo):
        """Check and record rating milestones."""
        milestones = [1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500]
        
        for milestone in milestones:
            milestone_key = f"elo_{milestone}"
            
            if (new_elo >= milestone and 
                milestone_key not in self.stats['milestones']['rating_milestones']):
                
                self.stats['milestones']['rating_milestones'][milestone_key] = {
                    'date': datetime.now().isoformat(),
                    'elo': round(new_elo)
                }
    
    def _update_game_data(self, game_data):
        """Update statistics with additional game data."""
        # Game length
        if 'move_count' in game_data:
            move_count = game_data['move_count']
            total_games = self.stats['games']['total_played']
            current_avg = self.stats['performance']['average_game_length']
            
            self.stats['performance']['average_game_length'] = (
                (current_avg * (total_games - 1) + move_count) / total_games
            )
        
        # Game duration
        if 'duration' in game_data:
            duration = game_data['duration']
            self.stats['time_management']['fastest_game'] = min(
                self.stats['time_management']['fastest_game'], duration
            )
            self.stats['time_management']['longest_game'] = max(
                self.stats['time_management']['longest_game'], duration
            )
        
        # Time trouble
        if game_data.get('time_trouble', False):
            self.stats['time_management']['time_trouble_games'] += 1
        
        # Accuracy
        if 'accuracy' in game_data:
            accuracy = game_data['accuracy']
            total_games = self.stats['games']['total_played']
            current_avg = self.stats['performance']['accuracy_average']
            
            self.stats['performance']['accuracy_average'] = (
                (current_avg * (total_games - 1) + accuracy) / total_games
            )
        
        # Tactical statistics
        if 'blunders' in game_data:
            blunders = game_data['blunders']
            total_games = self.stats['games']['total_played']
            current_avg = self.stats['tactical']['blunders_per_game']
            
            self.stats['tactical']['blunders_per_game'] = (
                (current_avg * (total_games - 1) + blunders) / total_games
            )
        
        # Opening statistics
        if 'opening' in game_data:
            opening = game_data['opening']
            
            # Track most played openings
            if opening not in self.stats['openings']['most_played']:
                self.stats['openings']['most_played'][opening] = 0
            self.stats['openings']['most_played'][opening] += 1
            
            # Track opening performance
            result = game_data.get('result', 'unknown')
            if opening not in self.stats['openings']['win_rates_by_opening']:
                self.stats['openings']['win_rates_by_opening'][opening] = {
                    'games': 0, 'wins': 0, 'draws': 0, 'losses': 0
                }
            
            opening_stats = self.stats['openings']['win_rates_by_opening'][opening]
            opening_stats['games'] += 1
            if result in ['win', 'draw', 'loss']:
                opening_stats[result + 's'] += 1
    
    def record_training_session(self, session_data):
        """
        Record a training session.
        
        Args:
            session_data: Dictionary with training session information
        """
        training = self.stats['training']
        
        if 'positions_analyzed' in session_data:
            training['positions_analyzed'] += session_data['positions_analyzed']
        
        if 'games_studied' in session_data:
            training['games_studied'] += session_data['games_studied']
        
        if 'duration' in session_data:
            training['training_time'] += session_data['duration']
        
        self._save_stats()
    
    def record_tactical_puzzle(self, solved, difficulty=None):
        """
        Record tactical puzzle attempt.
        
        Args:
            solved: Whether the puzzle was solved correctly
            difficulty: Puzzle difficulty rating
        """
        self.stats['tactical']['tactics_attempted'] += 1
        
        if solved:
            self.stats['tactical']['tactics_solved'] += 1
        
        self._save_stats()
    
    def get_performance_summary(self):
        """Get a summary of current performance."""
        games = self.stats['games']
        performance = self.stats['performance']
        
        total_games = games['total_played']
        if total_games == 0:
            return "No games played yet."
        
        win_percentage = (games['wins'] / total_games) * 100
        draw_percentage = (games['draws'] / total_games) * 100
        loss_percentage = (games['losses'] / total_games) * 100
        
        summary = f"""
Performance Summary:
===================
Games Played: {total_games}
Record: {games['wins']}W-{games['draws']}D-{games['losses']}L
Win Rate: {win_percentage:.1f}% | Draw Rate: {draw_percentage:.1f}% | Loss Rate: {loss_percentage:.1f}%

Rating Information:
Current ELO: {performance['current_elo']}
Peak ELO: {performance['peak_elo']}
Overall Win Rate: {performance['win_rate']:.1%}

Game Averages:
Average Game Length: {performance['average_game_length']:.1f} moves
Average Accuracy: {performance['accuracy_average']:.1f}%
Blunders per Game: {self.stats['tactical']['blunders_per_game']:.1f}

Time Management:
Time Trouble Games: {self.stats['time_management']['time_trouble_games']} ({(self.stats['time_management']['time_trouble_games']/total_games)*100:.1f}%)
"""
        
        return summary
    
    def get_improvement_trends(self, days=30):
        """Analyze improvement trends over time period."""
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_iso = cutoff_date.isoformat()
        
        # Filter recent ELO history
        recent_elos = [
            entry for entry in self.stats['performance']['elo_history']
            if entry['date'] >= cutoff_iso
        ]
        
        if len(recent_elos) < 2:
            return "Not enough recent games for trend analysis."
        
        start_elo = recent_elos[0]['elo']
        end_elo = recent_elos[-1]['elo']
        elo_change = end_elo - start_elo
        
        # Calculate win rate for recent games
        recent_results = [entry['result'] for entry in recent_elos]
        recent_wins = recent_results.count('win')
        recent_draws = recent_results.count('draw')
        recent_total = len(recent_results)
        recent_win_rate = (recent_wins + recent_draws * 0.5) / recent_total
        
        trend_direction = "improving" if elo_change > 0 else "declining" if elo_change < 0 else "stable"
        
        return {
            'period_days': days,
            'games_played': recent_total,
            'elo_change': elo_change,
            'trend_direction': trend_direction,
            'recent_win_rate': recent_win_rate,
            'start_elo': start_elo,
            'end_elo': end_elo
        }
    
    def get_opening_statistics(self):
        """Get opening performance statistics."""
        openings = self.stats['openings']['win_rates_by_opening']
        
        if not openings:
            return "No opening data available."
        
        # Calculate win rates
        opening_performance = []
        for opening, stats in openings.items():
            if stats['games'] >= 3:  # Only include openings with enough games
                win_rate = (stats['wins'] + stats['draws'] * 0.5) / stats['games']
                opening_performance.append({
                    'opening': opening,
                    'games': stats['games'],
                    'win_rate': win_rate,
                    'record': f"{stats['wins']}-{stats['draws']}-{stats['losses']}"
                })
        
        # Sort by win rate
        opening_performance.sort(key=lambda x: x['win_rate'], reverse=True)
        
        return opening_performance[:10]  # Top 10 openings
    
    def get_milestones(self):
        """Get achieved milestones."""
        milestones = []
        
        # Rating milestones
        for milestone, data in self.stats['milestones']['rating_milestones'].items():
            rating = milestone.split('_')[1]
            milestones.append({
                'type': 'Rating',
                'description': f"Reached {rating} ELO",
                'date': data['date'],
                'value': data['elo']
            })
        
        # First win
        if self.stats['milestones']['first_win']:
            milestones.append({
                'type': 'Achievement',
                'description': "First victory",
                'date': self.stats['milestones']['first_win'],
                'value': 1
            })
        
        # Sort by date
        milestones.sort(key=lambda x: x['date'], reverse=True)
        
        return milestones
    
    def export_statistics(self, output_file):
        """Export statistics to file."""
        with open(output_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"Statistics exported to {output_file}")
    
    def reset_statistics(self):
        """Reset all statistics (use with caution)."""
        self.stats = self._create_empty_stats()
        self._save_stats()
        print("All statistics have been reset.")
    
    def _save_stats(self):
        """Save statistics to file."""
        self.stats['last_updated'] = datetime.now().isoformat()
        
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def get_daily_activity(self, days=7):
        """Get daily activity for the past N days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_iso = cutoff_date.isoformat()
        
        # Count games per day
        daily_games = defaultdict(int)
        
        for entry in self.stats['performance']['elo_history']:
            if entry['date'] >= cutoff_iso:
                date_only = entry['date'][:10]  # Extract YYYY-MM-DD
                daily_games[date_only] += 1
        
        return dict(daily_games)
    
    def compare_with_target(self, target_elo=2000):
        """Compare current performance with target rating."""
        current_elo = self.stats['performance']['current_elo']
        elo_diff = target_elo - current_elo
        
        # Estimate games needed (rough calculation)
        avg_elo_gain_per_win = 20  # Approximate
        games_needed = max(0, elo_diff / avg_elo_gain_per_win)
        
        return {
            'current_elo': current_elo,
            'target_elo': target_elo,
            'elo_difference': elo_diff,
            'estimated_games_needed': round(games_needed),
            'status': 'achieved' if elo_diff <= 0 else 'in_progress'
        }