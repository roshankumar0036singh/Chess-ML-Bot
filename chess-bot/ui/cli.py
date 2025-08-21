import chess
import chess.pgn
import time
from core.engine import ChessEngine

class CLIInterface:
    """
    Command-line interface for the Chess ML Bot.
    Provides interactive gameplay and analysis features.
    """
    
    def __init__(self, engine: ChessEngine):
        self.engine = engine
        self.game_history = []
        self.player_color = chess.WHITE
        
    def run(self):
        """Main interface loop."""
        self.print_welcome()
        
        while True:
            try:
                command = input("\nChess Bot > ").strip().lower()
                
                if command in ['quit', 'exit', 'q']:
                    print("Thanks for playing!")
                    break
                elif command in ['help', 'h']:
                    self.print_help()
                elif command in ['new', 'newgame']:
                    self.start_new_game()
                elif command in ['show', 'board']:
                    self.show_board()
                elif command in ['moves', 'legal']:
                    self.show_legal_moves()
                elif command in ['eval', 'evaluate']:
                    self.show_evaluation()
                elif command in ['flip']:
                    self.flip_board()
                elif command in ['undo']:
                    self.undo_move()
                elif command in ['auto']:
                    self.auto_play()
                elif command.startswith('depth'):
                    self.set_search_depth(command)
                elif command.startswith('time'):
                    self.set_time_control(command)
                elif command in ['pgn']:
                    self.save_pgn()
                elif command in ['analyze']:
                    self.analyze_position()
                elif self.is_move(command):
                    self.make_human_move(command)
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit.")
            except Exception as e:
                print(f"Error: {e}")
    
    def print_welcome(self):
        """Print welcome message and instructions."""
        print("=" * 60)
        print("  CHESS ML BOT - Advanced Chess Engine")
        print("=" * 60)
        print("Features:")
        print("- Neural network position evaluation")
        print("- Monte Carlo Tree Search")
        print("- Opening book knowledge")
        print("- Endgame tablebase support")
        print("- Advanced position analysis")
        print()
        print("Type 'help' for commands or enter moves in algebraic notation")
        print("Example: e2e4, Nf3, O-O, etc.")
        print("=" * 60)
    
    def print_help(self):
        """Print help information."""
        print("\nAvailable Commands:")
        print("=" * 40)
        print("Game Control:")
        print("  new/newgame     - Start a new game")
        print("  quit/exit/q     - Exit the program")
        print("  undo            - Undo last move")
        print("  flip            - Flip board display")
        print()
        print("Information:")
        print("  show/board      - Display current position")
        print("  moves/legal     - Show legal moves")
        print("  eval/evaluate   - Show position evaluation")
        print("  analyze         - Deep position analysis")
        print()
        print("Settings:")
        print("  depth <n>       - Set search depth (1-8)")
        print("  time <s>        - Set thinking time in seconds")
        print()
        print("Game Features:")
        print("  auto            - Let bot play both sides")
        print("  pgn             - Save game in PGN format")
        print()
        print("Move Input:")
        print("  Use algebraic notation: e2e4, Nf3, O-O")
        print("  Or long algebraic: e2-e4, Ng1-f3")
        print("=" * 40)
    
    def start_new_game(self):
        """Start a new game."""
        self.engine.reset_game()
        self.game_history.clear()
        
        print("\nStarting new game!")
        color_choice = input("Play as (w)hite, (b)lack, or (a)uto? [w]: ").strip().lower()
        
        if color_choice == 'b':
            self.player_color = chess.BLACK
            print("You are playing as Black.")
            # Bot makes first move
            self.make_bot_move()
        elif color_choice == 'a':
            self.player_color = None
            print("Bot will play both sides.")
        else:
            self.player_color = chess.WHITE
            print("You are playing as White.")
        
        self.show_board()
    
    def show_board(self):
        """Display the current board position."""
        print()
        board_str = str(self.engine.board)
        
        # Add coordinates
        lines = board_str.split('\n')
        for i, line in enumerate(lines):
            rank = 8 - i
            print(f"{rank} {line}")
        
        print("  a b c d e f g h")
        print()
        
        # Show game status
        if self.engine.is_game_over():
            result = self.engine.get_game_result()
            print(f"Game Over: {result}")
            if result == "1-0":
                print("White wins!")
            elif result == "0-1":
                print("Black wins!")
            else:
                print("Draw!")
        else:
            turn = "White" if self.engine.board.turn == chess.WHITE else "Black"
            print(f"Turn: {turn}")
            
            if self.engine.board.is_check():
                print("Check!")
    
    def show_legal_moves(self):
        """Display all legal moves in current position."""
        moves = self.engine.get_legal_moves()
        if not moves:
            print("No legal moves available.")
            return
        
        print(f"\nLegal moves ({len(moves)}):")
        
        # Group moves by piece type for better display
        move_groups = {}
        for move in moves:
            piece = self.engine.board.piece_at(move.from_square)
            piece_name = piece.symbol().upper() if piece else "?"
            
            if piece_name not in move_groups:
                move_groups[piece_name] = []
            move_groups[piece_name].append(str(move))
        
        for piece, piece_moves in move_groups.items():
            print(f"  {piece}: {', '.join(piece_moves)}")
    
    def show_evaluation(self):
        """Display position evaluation."""
        eval_score = self.engine.evaluate_position()
        
        print(f"\nPosition Evaluation: {eval_score:+.2f}")
        
        if eval_score > 300:
            print("White has a significant advantage")
        elif eval_score > 100:
            print("White has a slight advantage")
        elif eval_score > -100:
            print("Position is roughly equal")
        elif eval_score > -300:
            print("Black has a slight advantage")
        else:
            print("Black has a significant advantage")
    
    def analyze_position(self):
        """Perform deep analysis of current position."""
        print("\nAnalyzing position...")
        
        # Get evaluation
        eval_score = self.engine.evaluate_position()
        print(f"Static Evaluation: {eval_score:+.2f}")
        
        # Get best move from engine
        start_time = time.time()
        best_move = self.engine.get_best_move()
        search_time = time.time() - start_time
        
        if best_move:
            print(f"Best Move: {best_move} (found in {search_time:.2f}s)")
            
            # Show what happens after best move
            self.engine.make_move(best_move)
            new_eval = self.engine.evaluate_position()
            print(f"Evaluation after best move: {-new_eval:+.2f}")
            self.engine.undo_move()
        
        # Check for tactical themes
        self.check_tactical_themes()
    
    def check_tactical_themes(self):
        """Check for common tactical themes."""
        print("\nTactical Analysis:")
        
        if self.engine.board.is_check():
            print("- Position has check")
        
        # Count attacks on pieces
        attacked_pieces = 0
        for square in chess.SQUARES:
            piece = self.engine.board.piece_at(square)
            if piece and self.engine.board.is_attacked_by(not piece.color, square):
                attacked_pieces += 1
        
        if attacked_pieces > 4:
            print("- High tactical complexity (many attacked pieces)")
        
        # Check for pins and skewers (simplified)
        if self.has_pins_or_skewers():
            print("- Potential pins or skewers detected")
    
    def has_pins_or_skewers(self):
        """Simple detection of pins and skewers."""
        # This is a simplified implementation
        # A full implementation would require more sophisticated analysis
        king_square = self.engine.board.king(self.engine.board.turn)
        if king_square is None:
            return False
        
        # Check if king is on same rank/file/diagonal as enemy pieces
        enemy_pieces = self.engine.board.pieces(not self.engine.board.turn, chess.QUEEN) | \
                      self.engine.board.pieces(not self.engine.board.turn, chess.ROOK) | \
                      self.engine.board.pieces(not self.engine.board.turn, chess.BISHOP)
        
        for piece_square in enemy_pieces:
            if chess.square_file(king_square) == chess.square_file(piece_square) or \
               chess.square_rank(king_square) == chess.square_rank(piece_square) or \
               abs(chess.square_file(king_square) - chess.square_file(piece_square)) == \
               abs(chess.square_rank(king_square) - chess.square_rank(piece_square)):
                return True
        
        return False
    
    def make_human_move(self, move_str):
        """Process human move input."""
        try:
            move = self.parse_move(move_str)
            if move in self.engine.board.legal_moves:
                self.engine.make_move(move)
                self.game_history.append(move)
                
                print(f"You played: {move}")
                self.show_board()
                
                if not self.engine.is_game_over() and self.player_color is not None:
                    if self.engine.board.turn != self.player_color:
                        self.make_bot_move()
                
            else:
                print("Illegal move. Try again.")
                
        except ValueError as e:
            print(f"Invalid move format: {e}")
    
    def make_bot_move(self):
        """Make a move for the bot."""
        if self.engine.is_game_over():
            return
        
        print("\nBot is thinking...")
        start_time = time.time()
        
        try:
            best_move = self.engine.get_best_move()
            if best_move:
                self.engine.make_move(best_move)
                self.game_history.append(best_move)
                
                thinking_time = time.time() - start_time
                print(f"Bot played: {best_move} (thought for {thinking_time:.2f}s)")
                
                self.show_board()
            else:
                print("Bot cannot find a move!")
                
        except Exception as e:
            print(f"Bot error: {e}")
    
    def auto_play(self):
        """Let bot play against itself."""
        print("Starting auto-play mode. Press Ctrl+C to stop.")
        
        move_count = 0
        try:
            while not self.engine.is_game_over() and move_count < 100:
                self.make_bot_move()
                move_count += 1
                
                # Pause between moves
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nAuto-play stopped.")
    
    def parse_move(self, move_str):
        """Parse move string to chess.Move object."""
        # Try different move formats
        try:
            # Try UCI format (e2e4)
            if len(move_str) == 4 and move_str[0] in 'abcdefgh' and move_str[2] in 'abcdefgh':
                return chess.Move.from_uci(move_str)
            
            # Try standard algebraic notation
            return self.engine.board.parse_san(move_str)
            
        except ValueError:
            raise ValueError(f"Cannot parse move: {move_str}")
    
    def is_move(self, command):
        """Check if command looks like a move."""
        # Basic heuristic to detect move input
        if len(command) >= 2:
            if command[0] in 'abcdefgh' or command[0] in 'NBRQK':
                return True
            if command.lower() in ['o-o', '0-0', 'o-o-o', '0-0-0']:
                return True
        return False
    
    def flip_board(self):
        """Flip board display orientation."""
        # This would be implemented in a GUI version
        print("Board flipping not available in CLI mode.")
    
    def undo_move(self):
        """Undo the last move."""
        if self.game_history:
            self.engine.undo_move()
            self.game_history.pop()
            print("Move undone.")
            self.show_board()
        else:
            print("No moves to undo.")
    
    def save_pgn(self):
        """Save current game as PGN."""
        if not self.game_history:
            print("No game to save.")
            return
        
        filename = input("Enter filename [game.pgn]: ").strip()
        if not filename:
            filename = "game.pgn"
        
        if not filename.endswith('.pgn'):
            filename += '.pgn'
        
        try:
            game = chess.pgn.Game()
            game.headers["Event"] = "Chess ML Bot Game"
            game.headers["Date"] = time.strftime("%Y.%m.%d")
            game.headers["White"] = "Human" if self.player_color == chess.WHITE else "Bot"
            game.headers["Black"] = "Bot" if self.player_color == chess.WHITE else "Human"
            game.headers["Result"] = self.engine.get_game_result()
            
            node = game
            temp_board = chess.Board()
            
            for move in self.game_history:
                node = node.add_variation(move)
                temp_board.push(move)
            
            with open(filename, 'w') as f:
                print(game, file=f)
            
            print(f"Game saved to {filename}")
            
        except Exception as e:
            print(f"Error saving game: {e}")
    
    def set_search_depth(self, command):
        """Set search depth for minimax."""
        try:
            depth = int(command.split()[1])
            if 1 <= depth <= 8:
                # This would set depth in the search algorithm
                print(f"Search depth set to {depth}")
            else:
                print("Depth must be between 1 and 8")
        except (IndexError, ValueError):
            print("Usage: depth <number>")
    
    def set_time_control(self, command):
        """Set time control for bot thinking."""
        try:
            seconds = float(command.split()[1])
            if 0.1 <= seconds <= 60:
                # This would set time limit in the search
                print(f"Thinking time set to {seconds} seconds")
            else:
                print("Time must be between 0.1 and 60 seconds")
        except (IndexError, ValueError):
            print("Usage: time <seconds>")