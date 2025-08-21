import chess
import os

try:
    import chess.syzygy
    SYZYGY_AVAILABLE = True
except ImportError:
    SYZYGY_AVAILABLE = False
    print("Warning: chess.syzygy not available. Tablebase support disabled.")

class TablebaseBot:
    """
    Endgame tablebase integration for perfect endgame play.
    Uses Syzygy tablebases for positions with 7 or fewer pieces.
    """
    
    def __init__(self, tablebase_path="data/syzygy"):
        self.tablebase_path = tablebase_path
        self.tablebase = None
        self.enabled = False
        
        if SYZYGY_AVAILABLE:
            self._load_tablebase()
    
    def _load_tablebase(self):
        """Load Syzygy tablebase from specified path."""
        if not os.path.exists(self.tablebase_path):
            print(f"Tablebase path {self.tablebase_path} not found.")
            print("Download Syzygy tablebases and place them in the data/syzygy directory.")
            return
        
        try:
            self.tablebase = chess.syzygy.open_tablebase(self.tablebase_path)
            self.enabled = True
            print(f"Tablebase loaded from {self.tablebase_path}")
            
            # Check which tablebases are available
            available = self._check_available_tablebases()
            if available:
                print(f"Available tablebases: {', '.join(available)}")
            else:
                print("No tablebase files found in directory.")
                self.enabled = False
                
        except Exception as e:
            print(f"Error loading tablebase: {e}")
            self.enabled = False
    
    def _check_available_tablebases(self):
        """Check which tablebase files are available."""
        if not os.path.exists(self.tablebase_path):
            return []
        
        tablebase_files = []
        for filename in os.listdir(self.tablebase_path):
            if filename.endswith('.rtbw') or filename.endswith('.rtbz'):
                # Extract piece configuration from filename
                base_name = filename.split('.')[0]
                if base_name not in tablebase_files:
                    tablebase_files.append(base_name)
        
        return sorted(tablebase_files)
    
    def probe_tablebase(self, board):
        """
        Probe tablebase for current position.
        
        Returns:
            None: Position not in tablebase or tablebase not available
            int: WDL (Win/Draw/Loss) value
                 2: Win in optimal play
                 1: Cursed win (win but 50-move rule)
                 0: Draw
                 -1: Blessed loss (loss but 50-move rule)
                 -2: Loss in optimal play
        """
        if not self.enabled or not self.tablebase:
            return None
        
        # Only probe if 7 or fewer pieces
        if len(board.piece_map()) > 7:
            return None
        
        try:
            wdl = self.tablebase.probe_wdl(board)
            return wdl
        except chess.syzygy.MissingTableError:
            # Tablebase for this material configuration not available
            return None
        except Exception:
            # Other errors (invalid position, etc.)
            return None
    
    def probe_dtz(self, board):
        """
        Probe distance-to-zero (DTZ) tablebase.
        
        Returns:
            None: Not available
            int: Distance to zeroing move (capture/pawn move)
        """
        if not self.enabled or not self.tablebase:
            return None
        
        if len(board.piece_map()) > 7:
            return None
        
        try:
            dtz = self.tablebase.probe_dtz(board)
            return dtz
        except chess.syzygy.MissingTableError:
            return None
        except Exception:
            return None
    
    def get_best_move(self, board):
        """
        Get the best move according to tablebase.
        
        Returns:
            None: No tablebase move available
            chess.Move: Best move for current position
        """
        if not self.enabled or not self.tablebase:
            return None
        
        current_wdl = self.probe_tablebase(board)
        if current_wdl is None:
            return None
        
        best_move = None
        best_wdl = None
        
        # Try all legal moves and find the best outcome
        for move in board.legal_moves:
            board.push(move)
            
            # Get WDL from opponent's perspective (so flip sign)
            move_wdl = self.probe_tablebase(board)
            if move_wdl is not None:
                move_wdl = -move_wdl  # Flip for current player
                
                if best_wdl is None or move_wdl > best_wdl:
                    best_wdl = move_wdl
                    best_move = move
            
            board.pop()
        
        return best_move
    
    def get_tablebase_info(self, board):
        """
        Get detailed tablebase information for position.
        
        Returns:
            dict: Tablebase information or None if not available
        """
        if not self.enabled:
            return None
        
        wdl = self.probe_tablebase(board)
        if wdl is None:
            return None
        
        dtz = self.probe_dtz(board)
        
        info = {
            'wdl': wdl,
            'dtz': dtz,
            'evaluation': self._wdl_to_evaluation(wdl),
            'description': self._wdl_to_description(wdl)
        }
        
        return info
    
    def _wdl_to_evaluation(self, wdl):
        """Convert WDL to evaluation score."""
        if wdl == 2:
            return 10000  # Winning
        elif wdl == 1:
            return 5000   # Cursed win
        elif wdl == 0:
            return 0      # Draw
        elif wdl == -1:
            return -5000  # Blessed loss
        elif wdl == -2:
            return -10000 # Losing
        else:
            return 0
    
    def _wdl_to_description(self, wdl):
        """Convert WDL to human-readable description."""
        descriptions = {
            2: "Win with perfect play",
            1: "Win, but may be drawn by 50-move rule",
            0: "Draw with perfect play",
            -1: "Loss, but may be drawn by 50-move rule",
            -2: "Loss with perfect play"
        }
        return descriptions.get(wdl, "Unknown")
    
    def analyze_endgame(self, board):
        """
        Provide detailed endgame analysis using tablebase.
        
        Returns:
            dict: Analysis results
        """
        if not self.enabled:
            return {"error": "Tablebase not available"}
        
        if len(board.piece_map()) > 7:
            return {"error": "Too many pieces for tablebase"}
        
        info = self.get_tablebase_info(board)
        if not info:
            return {"error": "Position not in tablebase"}
        
        analysis = {
            "position_eval": info['evaluation'],
            "result": info['description'],
            "best_move": None,
            "move_analysis": []
        }
        
        # Find best move
        best_move = self.get_best_move(board)
        if best_move:
            analysis['best_move'] = str(best_move)
        
        # Analyze all legal moves
        for move in board.legal_moves:
            board.push(move)
            move_info = self.get_tablebase_info(board)
            board.pop()
            
            if move_info:
                analysis['move_analysis'].append({
                    'move': str(move),
                    'wdl': -move_info['wdl'],  # Flip for current player
                    'eval': -move_info['evaluation'],
                    'description': move_info['description']
                })
        
        # Sort moves by evaluation
        analysis['move_analysis'].sort(key=lambda x: x['eval'], reverse=True)
        
        return analysis
    
    def is_tablebase_position(self, board):
        """Check if position can be handled by tablebase."""
        return (self.enabled and 
                len(board.piece_map()) <= 7 and
                self.probe_tablebase(board) is not None)
    
    def get_statistics(self):
        """Get tablebase statistics and information."""
        stats = {
            'enabled': self.enabled,
            'path': self.tablebase_path,
            'available': SYZYGY_AVAILABLE
        }
        
        if self.enabled:
            stats['tablebase_files'] = self._check_available_tablebases()
            stats['max_pieces'] = 7
        
        return stats

class EndgameKnowledge:
    """
    Basic endgame knowledge for positions not covered by tablebase.
    Implements fundamental endgame principles and patterns.
    """
    
    def __init__(self):
        pass
    
    def evaluate_endgame(self, board):
        """
        Evaluate endgame position using basic principles.
        
        Returns:
            float: Evaluation score
        """
        score = 0
        
        # King activity bonus in endgame
        score += self._king_activity_bonus(board)
        
        # Passed pawn evaluation
        score += self._passed_pawn_evaluation(board)
        
        # Opposition in king and pawn endgames
        if self._is_king_pawn_endgame(board):
            score += self._opposition_evaluation(board)
        
        return score
    
    def _king_activity_bonus(self, board):
        """Evaluate king activity in endgame."""
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        
        if white_king is None or black_king is None:
            return 0
        
        # Centralization bonus
        white_centralization = self._centralization_score(white_king)
        black_centralization = self._centralization_score(black_king)
        
        return (white_centralization - black_centralization) * 10
    
    def _centralization_score(self, square):
        """Calculate centralization score for a square."""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        # Distance from center
        center_distance = max(abs(file - 3.5), abs(rank - 3.5))
        
        # Lower distance = higher centralization
        return 4 - center_distance
    
    def _passed_pawn_evaluation(self, board):
        """Evaluate passed pawns in endgame."""
        score = 0
        
        for color in [chess.WHITE, chess.BLACK]:
            pawns = board.pieces(chess.PAWN, color)
            
            for pawn_square in pawns:
                if self._is_passed_pawn(board, pawn_square, color):
                    rank = chess.square_rank(pawn_square)
                    
                    # Passed pawn value increases with advancement
                    if color == chess.WHITE:
                        advancement = rank - 1  # 0-6
                        score += 20 + advancement * 10
                    else:
                        advancement = 6 - rank  # 0-6
                        score -= 20 + advancement * 10
        
        return score
    
    def _is_passed_pawn(self, board, pawn_square, color):
        """Check if pawn is passed."""
        file = chess.square_file(pawn_square)
        rank = chess.square_rank(pawn_square)
        
        # Check files and adjacent files for opposing pawns
        for check_file in [file - 1, file, file + 1]:
            if 0 <= check_file <= 7:
                file_mask = chess.BB_FILES[check_file]
                opposing_pawns = board.pieces(chess.PAWN, not color) & file_mask
                
                for opp_pawn in opposing_pawns:
                    opp_rank = chess.square_rank(opp_pawn)
                    
                    # Check if opposing pawn can stop this pawn
                    if color == chess.WHITE and opp_rank > rank:
                        return False
                    elif color == chess.BLACK and opp_rank < rank:
                        return False
        
        return True
    
    def _is_king_pawn_endgame(self, board):
        """Check if this is a king and pawn endgame."""
        piece_types = set()
        for piece in board.piece_map().values():
            piece_types.add(piece.piece_type)
        
        # Only kings and pawns
        return piece_types <= {chess.KING, chess.PAWN}
    
    def _opposition_evaluation(self, board):
        """Evaluate opposition in king and pawn endgames."""
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        
        if white_king is None or black_king is None:
            return 0
        
        # Simple opposition check
        file_diff = abs(chess.square_file(white_king) - chess.square_file(black_king))
        rank_diff = abs(chess.square_rank(white_king) - chess.square_rank(black_king))
        
        # Distant opposition (same file/rank, odd distance)
        if (file_diff == 0 and rank_diff % 2 == 1) or \
           (rank_diff == 0 and file_diff % 2 == 1):
            # Player to move has opposition
            if board.turn == chess.WHITE:
                return 30
            else:
                return -30
        
        return 0