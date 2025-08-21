import chess
import json
import os
from collections import defaultdict

class OpponentModel:
    """
    Models opponent behavior and adapts strategy accordingly.
    Tracks opening preferences, tactical weaknesses, and time management patterns.
    """
    
    def __init__(self, model_dir="data/opponent_models"):
        self.model_dir = model_dir
        self.opponent_profiles = {}
        self.current_opponent = None
        
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Load existing models
        self._load_all_models()
    
    def create_opponent_profile(self, opponent_id, name=None):
        """
        Create a new opponent profile.
        
        Args:
            opponent_id: Unique identifier for opponent
            name: Display name for opponent
        """
        profile = {
            'id': opponent_id,
            'name': name or opponent_id,
            'games_played': 0,
            'results': {'wins': 0, 'draws': 0, 'losses': 0},
            'opening_preferences': {},
            'tactical_patterns': {
                'blunder_rate': 0.0,
                'missed_tactics': 0,
                'calculation_depth': 3,
                'weak_endgames': []
            },
            'time_management': {
                'avg_move_time': 5.0,
                'time_trouble_frequency': 0.0,
                'blitz_rating_estimate': 1500
            },
            'playing_style': {
                'aggression': 0.5,  # 0.0 = passive, 1.0 = aggressive
                'positional_vs_tactical': 0.5,  # 0.0 = positional, 1.0 = tactical
                'risk_tolerance': 0.5  # 0.0 = safe, 1.0 = risky
            },
            'common_mistakes': [],
            'adaptation_notes': []
        }
        
        self.opponent_profiles[opponent_id] = profile
        self._save_model(opponent_id)
        
        return profile
    
    def update_from_game(self, opponent_id, game_data):
        """
        Update opponent model based on game data.
        
        Args:
            opponent_id: Opponent identifier
            game_data: Dictionary with game information
        """
        if opponent_id not in self.opponent_profiles:
            self.create_opponent_profile(opponent_id)
        
        profile = self.opponent_profiles[opponent_id]
        
        # Update game statistics
        profile['games_played'] += 1
        result = game_data.get('result', 'unknown')
        
        if result in profile['results']:
            profile['results'][result] += 1
        
        # Update opening preferences
        self._update_opening_preferences(profile, game_data)
        
        # Update tactical patterns
        self._update_tactical_patterns(profile, game_data)
        
        # Update time management
        self._update_time_management(profile, game_data)
        
        # Update playing style
        self._update_playing_style(profile, game_data)
        
        # Save updated model
        self._save_model(opponent_id)
    
    def _update_opening_preferences(self, profile, game_data):
        """Update opening preferences based on game."""
        moves = game_data.get('moves', [])
        
        if len(moves) >= 6:  # At least 3 moves per side
            opening_sequence = ' '.join(str(move) for move in moves[:6])
            
            if opening_sequence not in profile['opening_preferences']:
                profile['opening_preferences'][opening_sequence] = 0
            
            profile['opening_preferences'][opening_sequence] += 1
    
    def _update_tactical_patterns(self, profile, game_data):
        """Update tactical pattern analysis."""
        mistakes = game_data.get('mistakes', [])
        missed_tactics = game_data.get('missed_tactics', [])
        
        # Update blunder rate
        total_moves = len(game_data.get('moves', []))
        if total_moves > 0:
            blunder_count = len([m for m in mistakes if m.get('severity') == 'blunder'])
            current_blunder_rate = blunder_count / total_moves
            
            # Moving average
            games = profile['games_played']
            if games > 1:
                profile['tactical_patterns']['blunder_rate'] = (
                    (profile['tactical_patterns']['blunder_rate'] * (games - 1) + current_blunder_rate) / games
                )
            else:
                profile['tactical_patterns']['blunder_rate'] = current_blunder_rate
        
        # Update missed tactics
        profile['tactical_patterns']['missed_tactics'] += len(missed_tactics)
        
        # Estimate calculation depth
        if game_data.get('average_depth'):
            depth = game_data['average_depth']
            games = profile['games_played']
            if games > 1:
                profile['tactical_patterns']['calculation_depth'] = (
                    (profile['tactical_patterns']['calculation_depth'] * (games - 1) + depth) / games
                )
            else:
                profile['tactical_patterns']['calculation_depth'] = depth
    
    def _update_time_management(self, profile, game_data):
        """Update time management patterns."""
        move_times = game_data.get('move_times', [])
        time_trouble = game_data.get('time_trouble', False)
        
        if move_times:
            avg_time = sum(move_times) / len(move_times)
            games = profile['games_played']
            
            if games > 1:
                profile['time_management']['avg_move_time'] = (
                    (profile['time_management']['avg_move_time'] * (games - 1) + avg_time) / games
                )
            else:
                profile['time_management']['avg_move_time'] = avg_time
        
        # Update time trouble frequency
        if time_trouble:
            games = profile['games_played']
            current_freq = profile['time_management']['time_trouble_frequency']
            profile['time_management']['time_trouble_frequency'] = (
                (current_freq * (games - 1) + 1) / games
            )
    
    def _update_playing_style(self, profile, game_data):
        """Update playing style assessment."""
        # Analyze aggression level
        aggressive_moves = game_data.get('aggressive_moves', 0)
        total_moves = len(game_data.get('moves', []))
        
        if total_moves > 0:
            aggression_ratio = aggressive_moves / total_moves
            games = profile['games_played']
            
            if games > 1:
                profile['playing_style']['aggression'] = (
                    (profile['playing_style']['aggression'] * (games - 1) + aggression_ratio) / games
                )
            else:
                profile['playing_style']['aggression'] = aggression_ratio
        
        # Update positional vs tactical preference
        tactical_moves = game_data.get('tactical_moves', 0)
        positional_moves = game_data.get('positional_moves', 0)
        
        if tactical_moves + positional_moves > 0:
            tactical_ratio = tactical_moves / (tactical_moves + positional_moves)
            games = profile['games_played']
            
            if games > 1:
                profile['playing_style']['positional_vs_tactical'] = (
                    (profile['playing_style']['positional_vs_tactical'] * (games - 1) + tactical_ratio) / games
                )
            else:
                profile['playing_style']['positional_vs_tactical'] = tactical_ratio
    
    def get_strategy_recommendations(self, opponent_id, current_position=None):
        """
        Get strategic recommendations for playing against this opponent.
        
        Args:
            opponent_id: Opponent identifier
            current_position: Current board position (optional)
            
        Returns:
            dict: Strategy recommendations
        """
        if opponent_id not in self.opponent_profiles:
            return self._default_strategy()
        
        profile = self.opponent_profiles[opponent_id]
        recommendations = {
            'opening_strategy': self._get_opening_strategy(profile),
            'tactical_focus': self._get_tactical_strategy(profile),
            'time_strategy': self._get_time_strategy(profile),
            'playing_style': self._get_style_strategy(profile),
            'specific_weaknesses': self._identify_weaknesses(profile)
        }
        
        return recommendations
    
    def _get_opening_strategy(self, profile):
        """Get opening strategy recommendations."""
        preferences = profile['opening_preferences']
        
        if not preferences:
            return "No opening data available"
        
        # Find most common openings
        sorted_openings = sorted(preferences.items(), key=lambda x: x[1], reverse=True)
        most_common = sorted_openings[0][0] if sorted_openings else None
        
        strategy = f"Opponent commonly plays: {most_common}\n"
        strategy += "Consider preparing lines against their favorite openings.\n"
        
        # Suggest alternatives
        if len(sorted_openings) > 1:
            strategy += f"Alternative preparations needed for: {sorted_openings[1][0]}"
        
        return strategy
    
    def _get_tactical_strategy(self, profile):
        """Get tactical strategy recommendations."""
        patterns = profile['tactical_patterns']
        
        strategy = []
        
        if patterns['blunder_rate'] > 0.05:  # 5% blunder rate
            strategy.append("Opponent makes frequent blunders - maintain pressure")
        
        if patterns['calculation_depth'] < 4:
            strategy.append("Opponent has shallow calculation - create complex positions")
        
        if patterns['missed_tactics'] > 5:
            strategy.append("Opponent misses tactics - look for tactical opportunities")
        
        return '; '.join(strategy) if strategy else "Standard tactical approach"
    
    def _get_time_strategy(self, profile):
        """Get time management strategy."""
        time_mgmt = profile['time_management']
        
        if time_mgmt['time_trouble_frequency'] > 0.3:  # Gets in time trouble 30% of games
            return "Opponent often in time trouble - create complex positions late in game"
        
        if time_mgmt['avg_move_time'] > 10:
            return "Opponent thinks slowly - maintain time pressure"
        
        if time_mgmt['avg_move_time'] < 3:
            return "Opponent plays quickly - may make hasty decisions under pressure"
        
        return "Standard time management approach"
    
    def _get_style_strategy(self, profile):
        """Get playing style strategy."""
        style = profile['playing_style']
        
        strategies = []
        
        if style['aggression'] > 0.7:
            strategies.append("Opponent is aggressive - play solid, wait for overextension")
        elif style['aggression'] < 0.3:
            strategies.append("Opponent is passive - take initiative, create imbalances")
        
        if style['positional_vs_tactical'] > 0.7:
            strategies.append("Opponent prefers tactics - avoid sharp positions")
        elif style['positional_vs_tactical'] < 0.3:
            strategies.append("Opponent is positional - create tactical complications")
        
        if style['risk_tolerance'] > 0.7:
            strategies.append("Opponent takes risks - play soundly to exploit mistakes")
        elif style['risk_tolerance'] < 0.3:
            strategies.append("Opponent avoids risks - create forcing positions")
        
        return '; '.join(strategies) if strategies else "Balanced approach"
    
    def _identify_weaknesses(self, profile):
        """Identify specific weaknesses to exploit."""
        weaknesses = []
        
        # Tactical weaknesses
        if profile['tactical_patterns']['blunder_rate'] > 0.03:
            weaknesses.append("High blunder rate under pressure")
        
        # Time management weaknesses
        if profile['time_management']['time_trouble_frequency'] > 0.25:
            weaknesses.append("Poor time management")
        
        # Style weaknesses
        style = profile['playing_style']
        if style['aggression'] > 0.8:
            weaknesses.append("Overaggressive - can be provoked into mistakes")
        
        return weaknesses
    
    def _default_strategy(self):
        """Default strategy for unknown opponents."""
        return {
            'opening_strategy': "Play solid, mainline openings",
            'tactical_focus': "Standard tactical vigilance",
            'time_strategy': "Balanced time management",
            'playing_style': "Flexible, adapt to opponent's moves",
            'specific_weaknesses': []
        }
    
    def get_opponent_summary(self, opponent_id):
        """Get summary of opponent profile."""
        if opponent_id not in self.opponent_profiles:
            return "No data available for this opponent"
        
        profile = self.opponent_profiles[opponent_id]
        
        summary = f"""
Opponent: {profile['name']}
Games Played: {profile['games_played']}
Results: {profile['results']['wins']}W {profile['results']['draws']}D {profile['results']['losses']}L

Playing Style:
- Aggression: {profile['playing_style']['aggression']:.2f} (0=passive, 1=aggressive)
- Style: {profile['playing_style']['positional_vs_tactical']:.2f} (0=positional, 1=tactical)
- Risk: {profile['playing_style']['risk_tolerance']:.2f} (0=safe, 1=risky)

Tactical Patterns:
- Blunder Rate: {profile['tactical_patterns']['blunder_rate']:.1%}
- Calculation Depth: ~{profile['tactical_patterns']['calculation_depth']:.1f} moves
- Missed Tactics: {profile['tactical_patterns']['missed_tactics']}

Time Management:
- Average Move Time: {profile['time_management']['avg_move_time']:.1f}s
- Time Trouble Rate: {profile['time_management']['time_trouble_frequency']:.1%}
"""
        return summary
    
    def _save_model(self, opponent_id):
        """Save opponent model to disk."""
        if opponent_id in self.opponent_profiles:
            filepath = os.path.join(self.model_dir, f"{opponent_id}.json")
            with open(filepath, 'w') as f:
                json.dump(self.opponent_profiles[opponent_id], f, indent=2)
    
    def _load_all_models(self):
        """Load all opponent models from disk."""
        if not os.path.exists(self.model_dir):
            return
        
        for filename in os.listdir(self.model_dir):
            if filename.endswith('.json'):
                opponent_id = filename[:-5]  # Remove .json extension
                filepath = os.path.join(self.model_dir, filename)
                
                try:
                    with open(filepath, 'r') as f:
                        self.opponent_profiles[opponent_id] = json.load(f)
                except Exception as e:
                    print(f"Error loading opponent model {filename}: {e}")
    
    def set_current_opponent(self, opponent_id):
        """Set the current opponent for the game."""
        self.current_opponent = opponent_id
        
        if opponent_id not in self.opponent_profiles:
            self.create_opponent_profile(opponent_id)
    
    def get_current_strategy(self):
        """Get strategy for current opponent."""
        if self.current_opponent:
            return self.get_strategy_recommendations(self.current_opponent)
        else:
            return self._default_strategy()
    
    def list_known_opponents(self):
        """List all known opponents."""
        opponents = []
        for opponent_id, profile in self.opponent_profiles.items():
            opponents.append({
                'id': opponent_id,
                'name': profile['name'],
                'games': profile['games_played'],
                'score': f"{profile['results']['wins']}-{profile['results']['draws']}-{profile['results']['losses']}"
            })
        
        return sorted(opponents, key=lambda x: x['games'], reverse=True)