import chess
import random
from .search import MCTS
from .neural_net import ChessNet
from .evaluation import PositionEvaluator
from features.opening_book import OpeningBook
from features.tablebase import TablebaseBot
from features.time_manager import TimeManager

class ChessEngine:
    """
    Main chess engine that coordinates all components:
    - Neural network for position evaluation
    - MCTS for move selection
    - Opening book for early game
    - Tablebase for endgame perfection
    - Time management
    """
    
    def __init__(self):
        self.board = chess.Board()
        self.net = ChessNet()
        self.mcts = MCTS(self.net)
        self.evaluator = PositionEvaluator()
        self.opening_book = OpeningBook("data/opening_book.pgn")
        self.tablebase = TablebaseBot("data/syzygy/")
        self.time_manager = TimeManager()
        self.move_history = []
        
    def make_move(self, move):
        """Execute a move on the board and update history."""
        if move in self.board.legal_moves:
            self.board.push(move)
            self.move_history.append(move)
            return True
        return False
        
    def undo_move(self):
        """Undo the last move."""
        if self.move_history:
            self.board.pop()
            self.move_history.pop()
            
    def get_best_move(self):
        """
        Get the best move using the following priority:
        1. Opening book (if available)
        2. Tablebase (if in endgame)
        3. Neural network + MCTS search
        """
        
        # Check opening book first
        book_move = self.opening_book.get_move(self.board)
        if book_move:
            print(f"Using opening book move: {book_move}")
            return book_move
            
        # Check tablebase for endgame
        tb_result = self.tablebase.probe_tablebase(self.board)
        if tb_result is not None:
            # Get best tablebase move
            best_moves = []
            for move in self.board.legal_moves:
                self.board.push(move)
                result = self.tablebase.probe_tablebase(self.board)
                if result is not None and result > 0:  # Winning move
                    best_moves.append(move)
                self.board.pop()
            if best_moves:
                print(f"Using tablebase move: {best_moves[0]}")
                return best_moves[0]
                
        # Use neural network + MCTS
        time_allocation = self.time_manager.allocate_time(self.board)
        move = self.mcts.search(self.board, time_allocation)
        print(f"Using MCTS move: {move}")
        return move
        
    def evaluate_position(self):
        """Get position evaluation from neural network."""
        return self.evaluator.evaluate(self.board)
        
    def reset_game(self):
        """Reset the board to starting position."""
        self.board.reset()
        self.move_history.clear()
        
    def is_game_over(self):
        """Check if the game is over."""
        return self.board.is_game_over()
        
    def get_game_result(self):
        """Get the result of the game."""
        return self.board.result()
        
    def get_legal_moves(self):
        """Get all legal moves in current position."""
        return list(self.board.legal_moves)
        
    def get_board_fen(self):
        """Get FEN representation of current position."""
        return self.board.fen()
        
    def set_position_from_fen(self, fen):
        """Set board position from FEN string."""
        try:
            self.board.set_fen(fen)
            return True
        except ValueError:
            return False