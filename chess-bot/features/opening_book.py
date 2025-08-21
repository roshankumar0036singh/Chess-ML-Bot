import chess
import chess.pgn
import random
import json
from collections import defaultdict

class OpeningBook:
    """
    Opening book system for chess bot.
    Supports both static books from PGN files and dynamic learning.
    """
    
    def __init__(self, book_file=None):
        self.book = defaultdict(list)  # FEN -> list of (move, weight, games_count)
        self.transpositions = {}  # Track transpositions
        self.total_games = 0
        
        if book_file:
            self.load_from_pgn(book_file)
    
    def load_from_pgn(self, pgn_file):
        """Load opening book from PGN file of master games."""
        print(f"Loading opening book from {pgn_file}...")
        
        try:
            with open(pgn_file, 'r') as f:
                games_processed = 0
                
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    
                    # Only use games from strong players
                    white_elo = game.headers.get("WhiteElo", "0")
                    black_elo = game.headers.get("BlackElo", "0")
                    
                    try:
                        if int(white_elo) >= 2000 and int(black_elo) >= 2000:
                            self._process_game(game)
                            games_processed += 1
                            
                            if games_processed % 1000 == 0:
                                print(f"Processed {games_processed} games...")
                                
                    except ValueError:
                        continue  # Skip games without valid ELO
                
                self.total_games = games_processed
                print(f"Opening book loaded: {len(self.book)} positions from {games_processed} games")
                
        except FileNotFoundError:
            print(f"Opening book file {pgn_file} not found. Creating empty book.")
            
    def _process_game(self, game):
        """Process a single game and add moves to opening book."""
        board = game.board()
        move_count = 0
        
        # Only process first 15 moves (opening phase)
        for move in game.mainline_moves():
            if move_count >= 15:
                break
                
            fen = board.fen().split(' ')[0]  # Position without move counters
            
            # Add move to book
            found = False
            for i, (book_move, weight, count) in enumerate(self.book[fen]):
                if book_move == move:
                    # Update existing move
                    self.book[fen][i] = (book_move, weight + 1, count + 1)
                    found = True
                    break
            
            if not found:
                # Add new move
                self.book[fen].append((move, 1, 1))
            
            board.push(move)
            move_count += 1
    
    def get_move(self, board):
        """
        Get a book move for the current position.
        Returns None if position is not in book.
        """
        fen = board.fen().split(' ')[0]  # Position without move counters
        
        if fen not in self.book:
            return None
        
        moves = self.book[fen]
        if not moves:
            return None
        
        # Filter out illegal moves (book might be corrupted)
        legal_moves = []
        for move, weight, count in moves:
            if move in board.legal_moves:
                legal_moves.append((move, weight, count))
        
        if not legal_moves:
            return None
        
        # Select move based on weights (popularity)
        return self._select_weighted_move(legal_moves)
    
    def _select_weighted_move(self, moves):
        """Select move based on weights/popularity."""
        total_weight = sum(weight for _, weight, _ in moves)
        
        if total_weight == 0:
            return random.choice(moves)[0]
        
        # Weighted random selection
        r = random.random() * total_weight
        current_weight = 0
        
        for move, weight, count in moves:
            current_weight += weight
            if r <= current_weight:
                return move
        
        # Fallback
        return moves[0][0]
    
    def add_game_result(self, moves, result):
        """
        Add a game result to update move weights.
        
        Args:
            moves: List of moves played
            result: Game result (1.0 = white win, 0.5 = draw, 0.0 = black win)
        """
        board = chess.Board()
        
        for i, move in enumerate(moves):
            if i >= 15:  # Only update opening moves
                break
                
            fen = board.fen().split(' ')[0]
            
            if fen in self.book:
                # Update weight based on result
                for j, (book_move, weight, count) in enumerate(self.book[fen]):
                    if book_move == move:
                        # Adjust weight based on result
                        if board.turn == chess.WHITE:
                            new_weight = weight + result
                        else:
                            new_weight = weight + (1.0 - result)
                        
                        self.book[fen][j] = (book_move, new_weight, count + 1)
                        break
            
            board.push(move)
    
    def get_book_statistics(self):
        """Get statistics about the opening book."""
        total_positions = len(self.book)
        total_moves = sum(len(moves) for moves in self.book.values())
        
        if total_positions == 0:
            return "Opening book is empty"
        
        avg_moves_per_position = total_moves / total_positions
        
        stats = f"Opening Book Statistics:\n"
        stats += f"  Total positions: {total_positions}\n"
        stats += f"  Total moves: {total_moves}\n"
        stats += f"  Average moves per position: {avg_moves_per_position:.1f}\n"
        stats += f"  Games processed: {self.total_games}"
        
        return stats
    
    def get_book_moves(self, board):
        """Get all book moves for current position with statistics."""
        fen = board.fen().split(' ')[0]
        
        if fen not in self.book:
            return []
        
        moves = []
        for move, weight, count in self.book[fen]:
            if move in board.legal_moves:
                percentage = (count / max(1, sum(c for _, _, c in self.book[fen]))) * 100
                moves.append({
                    'move': move,
                    'weight': weight,
                    'count': count,
                    'percentage': percentage
                })
        
        return sorted(moves, key=lambda x: x['count'], reverse=True)
    
    def save_to_file(self, filename):
        """Save opening book to JSON file."""
        book_data = {}
        
        for fen, moves in self.book.items():
            book_data[fen] = [(str(move), weight, count) for move, weight, count in moves]
        
        try:
            with open(filename, 'w') as f:
                json.dump({
                    'book': book_data,
                    'total_games': self.total_games
                }, f, indent=2)
            
            print(f"Opening book saved to {filename}")
            
        except Exception as e:
            print(f"Error saving opening book: {e}")
    
    def load_from_file(self, filename):
        """Load opening book from JSON file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.book.clear()
            book_data = data.get('book', {})
            
            for fen, moves in book_data.items():
                self.book[fen] = [(chess.Move.from_uci(move_str), weight, count) 
                                 for move_str, weight, count in moves]
            
            self.total_games = data.get('total_games', 0)
            print(f"Opening book loaded from {filename}")
            
        except Exception as e:
            print(f"Error loading opening book: {e}")

class OpeningAnalyzer:
    """Analyzes opening trends and provides recommendations."""
    
    def __init__(self, opening_book):
        self.book = opening_book
    
    def analyze_opening(self, moves):
        """Analyze an opening sequence."""
        board = chess.Board()
        analysis = {
            'name': 'Unknown Opening',
            'moves_in_book': 0,
            'popularity': 0,
            'recommendations': []
        }
        
        for i, move in enumerate(moves):
            if i >= 15:
                break
                
            book_moves = self.book.get_book_moves(board)
            
            if book_moves:
                analysis['moves_in_book'] = i + 1
                
                # Check if played move is in book
                for book_move in book_moves:
                    if book_move['move'] == move:
                        analysis['popularity'] += book_move['percentage']
                        break
            
            board.push(move)
        
        # Get recommendations for next moves
        next_moves = self.book.get_book_moves(board)
        analysis['recommendations'] = next_moves[:3]  # Top 3 moves
        
        return analysis
    
    def identify_opening(self, moves):
        """Attempt to identify the opening name."""
        # This would be more sophisticated in a real implementation
        # You would need a database of opening names and their move sequences
        
        if not moves:
            return "Starting Position"
        
        first_move = moves[0]
        
        # Basic opening identification
        if first_move == chess.Move.from_uci("e2e4"):
            if len(moves) > 1 and moves[1] == chess.Move.from_uci("e7e5"):
                return "King's Pawn Game"
            elif len(moves) > 1 and moves[1] == chess.Move.from_uci("c7c5"):
                return "Sicilian Defense"
            else:
                return "King's Pawn Opening"
        elif first_move == chess.Move.from_uci("d2d4"):
            if len(moves) > 1 and moves[1] == chess.Move.from_uci("d7d5"):
                return "Queen's Pawn Game"
            elif len(moves) > 1 and moves[1] == chess.Move.from_uci("g8f6"):
                return "Indian Defense"
            else:
                return "Queen's Pawn Opening"
        elif first_move == chess.Move.from_uci("g1f3"):
            return "Reti Opening"
        elif first_move == chess.Move.from_uci("c2c4"):
            return "English Opening"
        
        return "Irregular Opening"

class OpeningTrainer:
    """Training mode for opening practice."""
    
    def __init__(self, opening_book):
        self.book = opening_book
        self.current_line = []
        self.mistakes = []
    
    def start_training(self, opening_name=None):
        """Start opening training session."""
        self.current_line = []
        self.mistakes = []
        
        print("Opening Training Mode")
        print("Play the moves you want to practice.")
        print("The book will show you popular continuations.")
        print("Type 'hint' for suggestions, 'quit' to exit training.")
    
    def process_move(self, board, move):
        """Process a training move and provide feedback."""
        book_moves = self.book.get_book_moves(board)
        
        if not book_moves:
            return "Position not in opening book. Training complete."
        
        # Check if move is in book
        move_found = False
        move_quality = "Uncommon"
        
        for book_move in book_moves:
            if book_move['move'] == move:
                move_found = True
                if book_move['percentage'] > 50:
                    move_quality = "Main line"
                elif book_move['percentage'] > 20:
                    move_quality = "Popular"
                elif book_move['percentage'] > 5:
                    move_quality = "Playable"
                else:
                    move_quality = "Rare"
                break
        
        if not move_found:
            self.mistakes.append((len(self.current_line), move))
            return f"Move {move} is not in the opening book. Consider: {book_moves[0]['move']}"
        
        self.current_line.append(move)
        return f"Good! {move} is {move_quality} ({book_move['percentage']:.1f}% of games)"
    
    def get_hints(self, board):
        """Get opening hints for current position."""
        book_moves = self.book.get_book_moves(board)
        
        if not book_moves:
            return "No book moves available for this position."
        
        hints = "Popular continuations:\n"
        for i, move_data in enumerate(book_moves[:3]):
            hints += f"  {i+1}. {move_data['move']} ({move_data['percentage']:.1f}%)\n"
        
        return hints
    
    def get_training_summary(self):
        """Get summary of training session."""
        if not self.current_line:
            return "No moves played in training."
        
        summary = f"Training Summary:\n"
        summary += f"Moves played: {len(self.current_line)}\n"
        summary += f"Mistakes: {len(self.mistakes)}\n"
        
        if self.mistakes:
            summary += "Positions to review:\n"
            for move_num, move in self.mistakes:
                summary += f"  Move {move_num + 1}: {move}\n"
        
        return summary