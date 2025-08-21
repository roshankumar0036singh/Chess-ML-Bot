import chess
import numpy as np

class PositionEvaluator:
    """
    Advanced position evaluation for chess positions.
    Combines material, positional, and strategic factors.
    """
    
    def __init__(self):
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Piece-square tables for positional evaluation
        self.pst = self._initialize_piece_square_tables()
        
    def evaluate(self, board: chess.Board):
        """
        Main evaluation function.
        Returns score in centipawns (positive = white advantage).
        """
        if board.is_checkmate():
            return -20000 if board.turn == chess.WHITE else 20000
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        score = 0
        
        # Material and positional evaluation
        score += self._material_evaluation(board)
        score += self._positional_evaluation(board)
        score += self._pawn_structure_evaluation(board)
        score += self._king_safety_evaluation(board)
        score += self._piece_mobility_evaluation(board)
        score += self._center_control_evaluation(board)
        
        # Endgame adjustments
        if self._is_endgame(board):
            score += self._endgame_evaluation(board)
        
        return score
    
    def _material_evaluation(self, board):
        """Basic material counting."""
        score = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    score += value
                else:
                    score -= value
        
        return score
    
    def _positional_evaluation(self, board):
        """Piece-square table evaluation."""
        score = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                pst_value = self.pst[piece.piece_type][square]
                if piece.color == chess.WHITE:
                    score += pst_value
                else:
                    score -= pst_value
        
        return score
    
    def _pawn_structure_evaluation(self, board):
        """Evaluate pawn structure."""
        score = 0
        
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        
        # Doubled pawns penalty
        for file in range(8):
            file_mask = chess.BB_FILES[file]
            
            white_pawns_on_file = len(white_pawns & file_mask)
            black_pawns_on_file = len(black_pawns & file_mask)
            
            if white_pawns_on_file > 1:
                score -= 20 * (white_pawns_on_file - 1)
            if black_pawns_on_file > 1:
                score += 20 * (black_pawns_on_file - 1)
        
        # Isolated pawns penalty
        score -= 15 * len(self._get_isolated_pawns(board, chess.WHITE))
        score += 15 * len(self._get_isolated_pawns(board, chess.BLACK))
        
        # Passed pawns bonus
        score += 30 * len(self._get_passed_pawns(board, chess.WHITE))
        score -= 30 * len(self._get_passed_pawns(board, chess.BLACK))
        
        return score
    
    def _king_safety_evaluation(self, board):
        """Evaluate king safety."""
        score = 0
        
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        
        if white_king is not None:
            score += self._king_safety_score(board, white_king, chess.WHITE)
        
        if black_king is not None:
            score -= self._king_safety_score(board, black_king, chess.BLACK)
        
        return score
    
    def _piece_mobility_evaluation(self, board):
        """Evaluate piece mobility."""
        score = 0
        
        # Count legal moves for each side
        white_mobility = len(list(board.legal_moves)) if board.turn == chess.WHITE else 0
        
        board.push(chess.Move.null())  # Switch turns
        black_mobility = len(list(board.legal_moves)) if board.turn == chess.BLACK else 0
        board.pop()
        
        score += (white_mobility - black_mobility) * 2
        
        return score
    
    def _center_control_evaluation(self, board):
        """Evaluate control of center squares."""
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        extended_center = [chess.C3, chess.C4, chess.C5, chess.C6,
                          chess.D3, chess.D6, chess.E3, chess.E6,
                          chess.F3, chess.F4, chess.F5, chess.F6]
        
        score = 0
        
        for square in center_squares:
            if board.is_attacked_by(chess.WHITE, square):
                score += 10
            if board.is_attacked_by(chess.BLACK, square):
                score -= 10
        
        for square in extended_center:
            if board.is_attacked_by(chess.WHITE, square):
                score += 5
            if board.is_attacked_by(chess.BLACK, square):
                score -= 5
        
        return score
    
    def _endgame_evaluation(self, board):
        """Special endgame evaluation."""
        score = 0
        
        # King activity in endgame
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        
        if white_king is not None and black_king is not None:
            # Centralize kings in endgame
            white_king_center_distance = self._distance_to_center(white_king)
            black_king_center_distance = self._distance_to_center(black_king)
            
            score += (black_king_center_distance - white_king_center_distance) * 5
        
        return score
    
    def _king_safety_score(self, board, king_square, color):
        """Calculate king safety score."""
        score = 0
        
        # Pawn shield bonus
        if color == chess.WHITE:
            shield_squares = [king_square + 8, king_square + 7, king_square + 9]
        else:
            shield_squares = [king_square - 8, king_square - 7, king_square - 9]
        
        for square in shield_squares:
            if 0 <= square <= 63:
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    score += 10
        
        # Penalty for exposed king
        attackers = 0
        for attacker_square in chess.SQUARES:
            piece = board.piece_at(attacker_square)
            if piece and piece.color != color:
                if board.is_attacked_by(not color, king_square):
                    attackers += 1
        
        score -= attackers * 15
        
        return score
    
    def _get_isolated_pawns(self, board, color):
        """Get list of isolated pawns."""
        pawns = board.pieces(chess.PAWN, color)
        isolated = []
        
        for pawn in pawns:
            file = chess.square_file(pawn)
            has_support = False
            
            # Check adjacent files for friendly pawns
            for adj_file in [file - 1, file + 1]:
                if 0 <= adj_file <= 7:
                    adj_file_mask = chess.BB_FILES[adj_file]
                    if pawns & adj_file_mask:
                        has_support = True
                        break
            
            if not has_support:
                isolated.append(pawn)
        
        return isolated
    
    def _get_passed_pawns(self, board, color):
        """Get list of passed pawns."""
        pawns = board.pieces(chess.PAWN, color)
        opponent_pawns = board.pieces(chess.PAWN, not color)
        passed = []
        
        for pawn in pawns:
            file = chess.square_file(pawn)
            rank = chess.square_rank(pawn)
            
            is_passed = True
            
            # Check if opponent pawns can stop this pawn
            for adj_file in [file - 1, file, file + 1]:
                if 0 <= adj_file <= 7:
                    adj_file_mask = chess.BB_FILES[adj_file]
                    opponent_pawns_on_file = opponent_pawns & adj_file_mask
                    
                    for opp_pawn in opponent_pawns_on_file:
                        opp_rank = chess.square_rank(opp_pawn)
                        
                        # Check if opponent pawn can stop advancement
                        if color == chess.WHITE:
                            if opp_rank > rank:
                                is_passed = False
                                break
                        else:
                            if opp_rank < rank:
                                is_passed = False
                                break
                    
                    if not is_passed:
                        break
            
            if is_passed:
                passed.append(pawn)
        
        return passed
    
    def _distance_to_center(self, square):
        """Calculate distance from square to center of board."""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        file_distance = min(file, 7 - file)
        rank_distance = min(rank, 7 - rank)
        
        return min(file_distance, rank_distance)
    
    def _is_endgame(self, board):
        """Determine if position is in endgame phase."""
        # Simple endgame detection: few pieces remaining
        piece_count = len(board.piece_map())
        return piece_count <= 12
    
    def _initialize_piece_square_tables(self):
        """Initialize piece-square tables for positional evaluation."""
        
        # Pawn table
        pawn_table = [
            0,   0,   0,   0,   0,   0,   0,   0,
            50,  50,  50,  50,  50,  50,  50,  50,
            10,  10,  20,  30,  30,  20,  10,  10,
            5,   5,  10,  25,  25,  10,   5,   5,
            0,   0,   0,  20,  20,   0,   0,   0,
            5,  -5, -10,   0,   0, -10,  -5,   5,
            5,  10,  10, -20, -20,  10,  10,   5,
            0,   0,   0,   0,   0,   0,   0,   0
        ]
        
        # Knight table
        knight_table = [
            -50, -40, -30, -30, -30, -30, -40, -50,
            -40, -20,   0,   0,   0,   0, -20, -40,
            -30,   0,  10,  15,  15,  10,   0, -30,
            -30,   5,  15,  20,  20,  15,   5, -30,
            -30,   0,  15,  20,  20,  15,   0, -30,
            -30,   5,  10,  15,  15,  10,   5, -30,
            -40, -20,   0,   5,   5,   0, -20, -40,
            -50, -40, -30, -30, -30, -30, -40, -50
        ]
        
        # Bishop table
        bishop_table = [
            -20, -10, -10, -10, -10, -10, -10, -20,
            -10,   0,   0,   0,   0,   0,   0, -10,
            -10,   0,   5,  10,  10,   5,   0, -10,
            -10,   5,   5,  10,  10,   5,   5, -10,
            -10,   0,  10,  10,  10,  10,   0, -10,
            -10,  10,  10,  10,  10,  10,  10, -10,
            -10,   5,   0,   0,   0,   0,   5, -10,
            -20, -10, -10, -10, -10, -10, -10, -20
        ]
        
        # Rook table
        rook_table = [
            0,   0,   0,   0,   0,   0,   0,   0,
            5,  10,  10,  10,  10,  10,  10,   5,
            -5,   0,   0,   0,   0,   0,   0,  -5,
            -5,   0,   0,   0,   0,   0,   0,  -5,
            -5,   0,   0,   0,   0,   0,   0,  -5,
            -5,   0,   0,   0,   0,   0,   0,  -5,
            -5,   0,   0,   0,   0,   0,   0,  -5,
            0,   0,   0,   5,   5,   0,   0,   0
        ]
        
        # Queen table
        queen_table = [
            -20, -10, -10,  -5,  -5, -10, -10, -20,
            -10,   0,   0,   0,   0,   0,   0, -10,
            -10,   0,   5,   5,   5,   5,   0, -10,
            -5,   0,   5,   5,   5,   5,   0,  -5,
            0,   0,   5,   5,   5,   5,   0,  -5,
            -10,   5,   5,   5,   5,   5,   0, -10,
            -10,   0,   5,   0,   0,   0,   0, -10,
            -20, -10, -10,  -5,  -5, -10, -10, -20
        ]
        
        # King middle game table
        king_table = [
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -20, -30, -30, -40, -40, -30, -30, -20,
            -10, -20, -20, -20, -20, -20, -20, -10,
            20,  20,   0,   0,   0,   0,  20,  20,
            20,  30,  10,   0,   0,  10,  30,  20
        ]
        
        return {
            chess.PAWN: pawn_table,
            chess.KNIGHT: knight_table,
            chess.BISHOP: bishop_table,
            chess.ROOK: rook_table,
            chess.QUEEN: queen_table,
            chess.KING: king_table
        }