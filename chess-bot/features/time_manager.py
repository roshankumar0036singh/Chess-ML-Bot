import time

class TimeManager:
    """
    Manages time allocation for chess moves.
    Implements various time control schemes and adaptive time management.
    """
    
    def __init__(self, total_time=600, increment=0, time_control_type="fischer"):
        """
        Initialize time manager.
        
        Args:
            total_time: Total time in seconds
            increment: Time increment per move in seconds
            time_control_type: Type of time control ("fischer", "classical", "bullet", "blitz")
        """
        self.total_time = total_time
        self.increment = increment
        self.time_control_type = time_control_type
        self.time_used = 0
        self.moves_played = 0
        self.game_phase = "opening"  # opening, middlegame, endgame
        
        # Time allocation parameters
        self.base_time_factor = 0.04  # Use 4% of remaining time by default
        self.complexity_multiplier = 1.0
        self.pressure_multiplier = 1.0
        
    def allocate_time(self, board, depth=None):
        """
        Allocate thinking time for current position.
        
        Args:
            board: Current chess position
            depth: Search depth (optional)
            
        Returns:
            float: Time to spend on this move in seconds
        """
        remaining_time = max(0.1, self.total_time - self.time_used)
        
        # Base time allocation
        if self.time_control_type == "fischer":
            base_time = self._fischer_allocation(remaining_time)
        elif self.time_control_type == "classical":
            base_time = self._classical_allocation(remaining_time)
        elif self.time_control_type == "bullet":
            base_time = self._bullet_allocation(remaining_time)
        elif self.time_control_type == "blitz":
            base_time = self._blitz_allocation(remaining_time)
        else:
            base_time = remaining_time * self.base_time_factor
        
        # Adjust for position complexity
        complexity_factor = self._calculate_complexity(board)
        adjusted_time = base_time * complexity_factor
        
        # Adjust for time pressure
        pressure_factor = self._calculate_time_pressure(remaining_time)
        final_time = adjusted_time * pressure_factor
        
        # Apply game phase adjustments
        phase_factor = self._get_phase_factor(board)
        final_time *= phase_factor
        
        # Enforce minimum and maximum time limits
        min_time = 0.1
        max_time = min(remaining_time * 0.5, 30.0)  # Don't use more than 50% of remaining time
        
        return max(min_time, min(final_time, max_time))
    
    def _fischer_allocation(self, remaining_time):
        """Fischer time control allocation (time + increment)."""
        moves_to_go = max(1, 40 - self.moves_played)
        base_allocation = remaining_time / moves_to_go
        
        # Add increment to base allocation
        return base_allocation + self.increment * 0.8  # Use 80% of increment
    
    def _classical_allocation(self, remaining_time):
        """Classical time control allocation."""
        # Estimate moves to time control (typically 40 moves)
        moves_to_control = max(1, 40 - (self.moves_played % 40))
        return remaining_time / moves_to_control
    
    def _bullet_allocation(self, remaining_time):
        """Bullet time control allocation (very fast)."""
        # Use smaller percentage of time in bullet
        return remaining_time * 0.02  # 2% of remaining time
    
    def _blitz_allocation(self, remaining_time):
        """Blitz time control allocation."""
        # Conservative allocation for blitz
        return remaining_time * 0.03  # 3% of remaining time
    
    def _calculate_complexity(self, board):
        """
        Calculate position complexity factor.
        
        Args:
            board: Chess position
            
        Returns:
            float: Complexity multiplier (0.5 to 3.0)
        """
        complexity_score = 1.0
        
        # Number of legal moves (more moves = more complex)
        num_moves = len(list(board.legal_moves))
        if num_moves > 35:
            complexity_score += 0.5
        elif num_moves < 10:
            complexity_score -= 0.3
        
        # Check if in check (requires more calculation)
        if board.is_check():
            complexity_score += 0.4
        
        # Tactical indicators
        if self._has_tactical_themes(board):
            complexity_score += 0.6
        
        # Piece count (fewer pieces = endgame = more precise calculation needed)
        piece_count = len(board.piece_map())
        if piece_count <= 10:  # Endgame
            complexity_score += 0.3
        elif piece_count >= 28:  # Opening
            complexity_score -= 0.2
        
        return max(0.5, min(3.0, complexity_score))
    
    def _has_tactical_themes(self, board):
        """Check for tactical themes that require deeper calculation."""
        # Simplified tactical detection
        
        # Count attacked pieces
        attacked_pieces = 0
        for square in board.piece_map():
            piece = board.piece_at(square)
            if piece and board.is_attacked_by(not piece.color, square):
                attacked_pieces += 1
        
        # High number of attacked pieces suggests tactics
        return attacked_pieces > 4
    
    def _calculate_time_pressure(self, remaining_time):
        """
        Calculate time pressure factor.
        
        Args:
            remaining_time: Remaining time in seconds
            
        Returns:
            float: Time pressure multiplier (0.3 to 1.5)
        """
        initial_time = self.total_time
        time_ratio = remaining_time / initial_time
        
        if time_ratio > 0.5:
            # Plenty of time, can think longer
            return 1.2
        elif time_ratio > 0.2:
            # Moderate time pressure
            return 1.0
        elif time_ratio > 0.1:
            # High time pressure, think faster
            return 0.7
        else:
            # Critical time pressure, move quickly
            return 0.3
    
    def _get_phase_factor(self, board):
        """Get time allocation factor based on game phase."""
        piece_count = len(board.piece_map())
        
        if piece_count >= 28:  # Opening
            self.game_phase = "opening"
            return 0.8  # Spend less time in opening
        elif piece_count >= 14:  # Middlegame
            self.game_phase = "middlegame"
            return 1.2  # Spend more time in middlegame
        else:  # Endgame
            self.game_phase = "endgame"
            return 1.1  # Moderate time in endgame
    
    def record_move_time(self, time_spent):
        """
        Record time spent on last move.
        
        Args:
            time_spent: Time spent on the move in seconds
        """
        self.time_used += time_spent
        self.moves_played += 1
        
        # Add increment after move (Fischer rules)
        if self.time_control_type == "fischer":
            self.total_time += self.increment
    
    def get_time_status(self):
        """Get current time status."""
        remaining_time = max(0, self.total_time - self.time_used)
        time_per_move = remaining_time / max(1, 50 - self.moves_played)  # Estimate moves left
        
        status = {
            'remaining_time': remaining_time,
            'time_used': self.time_used,
            'moves_played': self.moves_played,
            'game_phase': self.game_phase,
            'avg_time_per_move': time_per_move,
            'time_pressure': self._get_time_pressure_level(remaining_time)
        }
        
        return status
    
    def _get_time_pressure_level(self, remaining_time):
        """Get time pressure level as string."""
        time_ratio = remaining_time / self.total_time
        
        if time_ratio > 0.5:
            return "low"
        elif time_ratio > 0.2:
            return "moderate"
        elif time_ratio > 0.1:
            return "high"
        else:
            return "critical"
    
    def is_in_time_trouble(self):
        """Check if in time trouble."""
        remaining_time = max(0, self.total_time - self.time_used)
        return remaining_time < self.total_time * 0.1  # Less than 10% of original time
    
    def emergency_time_allocation(self):
        """Emergency time allocation when in severe time trouble."""
        remaining_time = max(0, self.total_time - self.time_used)
        
        # In emergency, use very little time per move
        if remaining_time < 10:  # Less than 10 seconds
            return 0.5  # Half second per move
        elif remaining_time < 30:  # Less than 30 seconds
            return 1.0  # One second per move
        else:
            return 2.0  # Two seconds per move
    
    def suggest_time_control(self, target_game_length_minutes=30):
        """
        Suggest time control settings for target game length.
        
        Args:
            target_game_length_minutes: Target game length in minutes
            
        Returns:
            dict: Suggested time control settings
        """
        target_seconds = target_game_length_minutes * 60
        
        # Estimate moves per game (typically 40-80 moves)
        estimated_moves = 60
        
        if target_seconds < 180:  # Less than 3 minutes
            return {
                'type': 'bullet',
                'time': target_seconds,
                'increment': 0,
                'description': f'{target_seconds//60}+0 bullet'
            }
        elif target_seconds < 600:  # Less than 10 minutes
            increment = max(0, (target_seconds - 300) // 60)
            base_time = target_seconds - increment * estimated_moves
            return {
                'type': 'blitz',
                'time': base_time,
                'increment': increment,
                'description': f'{base_time//60}+{increment} blitz'
            }
        else:
            # Classical or rapid
            increment = min(30, target_seconds // 100)
            base_time = target_seconds - increment * estimated_moves
            return {
                'type': 'classical',
                'time': base_time,
                'increment': increment,
                'description': f'{base_time//60}+{increment} classical'
            }
    
    def reset(self, total_time=None, increment=None):
        """Reset time manager for new game."""
        if total_time is not None:
            self.total_time = total_time
        if increment is not None:
            self.increment = increment
            
        self.time_used = 0
        self.moves_played = 0
        self.game_phase = "opening"