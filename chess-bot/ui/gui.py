import pygame
import chess
import sys
import os
import threading
import time

class ChessGUI:
    def __init__(self, engine):
        """Initialize the chess GUI"""
        self.engine = engine
        self.board = chess.Board()
        
        # Initialize pygame
        pygame.init()
        
        # Display settings
        self.board_size = 512
        self.square_size = self.board_size // 8
        self.info_width = 220
        self.screen_width = self.board_size + self.info_width
        self.screen_height = self.board_size
        
        # Create screen
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Chess ML Bot")
        
        # Game state
        self.selected_square = None
        self.legal_moves = []
        self.running = True
        self.thinking = False
        self.last_move = None
        self.bot_move_ready = False
        self.pending_bot_move = None
        
        # Threading for bot moves
        self.bot_thread = None
        self.move_lock = threading.Lock()
        
        # Colors
        self.colors = {
            'light_square': (240, 217, 181),
            'dark_square': (181, 136, 99),
            'selected': (255, 255, 0, 150),
            'legal_move': (0, 255, 0, 100),
            'last_move': (255, 255, 0, 80),
            'bot_move': (255, 100, 100, 120),
            'background': (50, 50, 50),
            'text': (255, 255, 255),
            'move_history': (200, 200, 200),
            'thinking': (255, 200, 0)
        }
        
        # Fonts
        self.font_large = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        self.font_move = pygame.font.Font(None, 16)
        
        # Clock and timing
        self.clock = pygame.time.Clock()
        
        # Move history
        self.move_history = []
        self.thinking_dots = 0
        self.thinking_timer = 0
        
        # Load piece images
        self._load_piece_images()
        
        print("Chess GUI initialized successfully")

    def _load_piece_images(self):
        """Load and resize piece images"""
        self.piece_images = {}
        piece_size = int(self.square_size * 0.8)
        
        # Unicode chess symbols
        pieces = {
            'wp': '♙', 'wr': '♖', 'wn': '♘', 'wb': '♗', 'wq': '♕', 'wk': '♔',
            'bp': '♟', 'br': '♜', 'bn': '♞', 'bb': '♝', 'bq': '♛', 'bk': '♚'
        }
        
        for piece_code, symbol in pieces.items():
            try:
                image_path = f"assets/pieces/{piece_code}.png"
                if os.path.exists(image_path):
                    img = pygame.image.load(image_path)
                    self.piece_images[piece_code] = pygame.transform.scale(img, (piece_size, piece_size))
                else:
                    self._create_text_piece(piece_code, symbol, piece_size)
            except:
                self._create_text_piece(piece_code, symbol, piece_size)

    def _create_text_piece(self, piece_code, symbol, size):
        """Create a text-based piece image"""
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        font = pygame.font.Font(None, int(size * 0.9))
        
        text_color = (50, 50, 50) if piece_code[0] == 'w' else (200, 200, 200)
        bg_color = (240, 240, 240) if piece_code[0] == 'w' else (60, 60, 60)
        
        pygame.draw.circle(surface, bg_color, (size//2, size//2), size//2 - 2)
        pygame.draw.circle(surface, (100, 100, 100), (size//2, size//2), size//2 - 2, 2)
        
        text = font.render(symbol, True, text_color)
        text_rect = text.get_rect(center=(size//2, size//2))
        surface.blit(text, text_rect)
        
        self.piece_images[piece_code] = surface

    def run(self):
        """Main game loop"""
        print("Starting Chess GUI main loop")
        
        while self.running:
            try:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if not self.thinking:  # Only allow moves when not thinking
                            self.handle_click(event.pos)
                    elif event.type == pygame.KEYDOWN:
                        self.handle_keypress(event.key)
                
                # Check for completed bot moves
                if self.bot_move_ready:
                    with self.move_lock:
                        if self.pending_bot_move:
                            self.apply_bot_move(self.pending_bot_move)
                            self.pending_bot_move = None
                        self.bot_move_ready = False
                        self.thinking = False
                
                # Update thinking animation
                if self.thinking:
                    self.thinking_timer += self.clock.get_time()
                    if self.thinking_timer > 500:  # Update every 500ms
                        self.thinking_dots = (self.thinking_dots + 1) % 4
                        self.thinking_timer = 0
                
                # Draw everything
                self.draw()
                pygame.display.flip()
                self.clock.tick(60)
                
            except Exception as e:
                print(f"GUI Error: {e}")
                continue
        
        # Clean up threads
        if self.bot_thread and self.bot_thread.is_alive():
            self.bot_thread.join(timeout=1.0)
        
        pygame.quit()
        print("Chess GUI closed")

    def handle_click(self, pos):
        """Handle mouse clicks on the board"""
        try:
            if pos[0] >= self.board_size or self.thinking:
                return
            
            file = pos[0] // self.square_size
            rank = 7 - (pos[1] // self.square_size)
            square = chess.square(file, rank)
            
            if self.selected_square is None:
                piece = self.board.piece_at(square)
                if piece and piece.color == self.board.turn:
                    self.selected_square = square
                    self.legal_moves = [move for move in self.board.legal_moves 
                                       if move.from_square == square]
                    print(f"Selected {piece} at {chess.square_name(square)}")
            else:
                move = chess.Move(self.selected_square, square)
                
                # Handle promotion
                piece = self.board.piece_at(self.selected_square)
                if (piece and piece.piece_type == chess.PAWN and 
                    ((piece.color == chess.WHITE and rank == 7) or 
                     (piece.color == chess.BLACK and rank == 0))):
                    move = chess.Move(self.selected_square, square, promotion=chess.QUEEN)
                
                if move in self.board.legal_moves:
                    self.make_player_move(move)
                
                self.selected_square = None
                self.legal_moves = []
                
        except Exception as e:
            print(f"Click handling error: {e}")
            self.selected_square = None
            self.legal_moves = []

    def make_player_move(self, move):
        """Make a player move and trigger bot response"""
        try:
            # Record the move
            move_san = self.board.san(move)
            self.board.push(move)
            self.last_move = move
            
            # Add to move history
            move_num = len(self.board.move_stack)
            if move_num % 2 == 1:
                self.move_history.append(f"{(move_num + 1) // 2}. {move_san}")
            else:
                if len(self.move_history) > 0:
                    self.move_history[-1] += f" {move_san}"
            
            print(f"Player move: {move_san}")
            
            # Update engine board state
            if self.engine:
                self.engine.board = self.board.copy()
            
            # Check if game is over
            if self.board.is_game_over():
                self.handle_game_over()
                return
            
            # Start bot thinking in separate thread
            if not self.thinking:
                self.thinking = True
                self.thinking_dots = 0
                self.thinking_timer = 0
                self.bot_thread = threading.Thread(target=self.bot_think_thread, daemon=True)
                self.bot_thread.start()
                
        except Exception as e:
            print(f"Error making player move: {e}")

    def bot_think_thread(self):
        """Bot thinking thread - runs in background"""
        try:
            print("Bot is thinking...")
            
            # Get bot move from engine
            bot_move = None
            if self.engine:
                try:
                    # Add a small delay to show thinking state
                    time.sleep(0.5)
                    bot_move = self.engine.get_best_move()
                except Exception as e:
                    print(f"Engine error: {e}")
            
            # Fallback to random legal move
            if bot_move is None:
                legal_moves = list(self.board.legal_moves)
                if legal_moves:
                    import random
                    bot_move = random.choice(legal_moves)
                    time.sleep(1.0)  # Simulate thinking time for random move
            
            # Signal that move is ready
            if bot_move:
                with self.move_lock:
                    self.pending_bot_move = bot_move
                    self.bot_move_ready = True
            else:
                self.thinking = False
                
        except Exception as e:
            print(f"Error in bot thinking thread: {e}")
            self.thinking = False

    def apply_bot_move(self, bot_move):
        """Apply the bot move to the board"""
        try:
            # Record the move
            move_san = self.board.san(bot_move)
            self.board.push(bot_move)
            self.last_move = bot_move
            
            # Add to move history
            move_num = len(self.board.move_stack)
            if move_num % 2 == 1:
                self.move_history.append(f"{(move_num + 1) // 2}. {move_san}")
            else:
                if len(self.move_history) > 0:
                    self.move_history[-1] += f" {move_san}"
            
            print(f"Bot move: {move_san} ({bot_move.uci()})")
            
            # Update engine board state
            if self.engine:
                self.engine.board = self.board.copy()
            
            # Check if game is over
            if self.board.is_game_over():
                self.handle_game_over()
                
        except Exception as e:
            print(f"Error applying bot move: {e}")

    def handle_keypress(self, key):
        """Handle keyboard input"""
        try:
            if key == pygame.K_n:  # New game
                # Stop any ongoing bot thinking
                if self.bot_thread and self.bot_thread.is_alive():
                    self.thinking = False
                    # Don't wait for thread, just reset
                
                self.board.reset()
                self.selected_square = None
                self.legal_moves = []
                self.thinking = False
                self.last_move = None
                self.move_history = []
                self.bot_move_ready = False
                self.pending_bot_move = None
                print("New game started")
                
            elif key == pygame.K_u and not self.thinking:  # Undo (only when not thinking)
                if len(self.board.move_stack) > 0:
                    move = self.board.pop()
                    self.last_move = self.board.move_stack[-1] if self.board.move_stack else None
                    # Update move history
                    if self.move_history:
                        if " " in self.move_history[-1]:
                            self.move_history[-1] = self.move_history[-1].split()[0] + "."
                        else:
                            self.move_history.pop()
                    print(f"Undid move: {move.uci()}")
                    
        except Exception as e:
            print(f"Keypress handling error: {e}")

    def handle_game_over(self):
        """Handle game over scenarios"""
        if self.board.is_checkmate():
            winner = "White" if self.board.turn == chess.BLACK else "Black"
            print(f"Checkmate! {winner} wins!")
        elif self.board.is_stalemate():
            print("Stalemate! Draw!")
        else:
            print("Game over!")

    def draw(self):
        """Draw the entire game state"""
        self.screen.fill(self.colors['background'])
        self.draw_board()
        self.draw_highlights()
        self.draw_pieces()
        self.draw_info_panel()

    def draw_board(self):
        """Draw the chessboard squares"""
        for rank in range(8):
            for file in range(8):
                is_light = (rank + file) % 2 == 0
                color = self.colors['light_square'] if is_light else self.colors['dark_square']
                
                rect = pygame.Rect(
                    file * self.square_size,
                    (7 - rank) * self.square_size,
                    self.square_size,
                    self.square_size
                )
                pygame.draw.rect(self.screen, color, rect)

    def draw_highlights(self):
        """Draw square highlights"""
        # Highlight last move
        if self.last_move:
            for square in [self.last_move.from_square, self.last_move.to_square]:
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                
                rect = pygame.Rect(
                    file * self.square_size,
                    (7 - rank) * self.square_size,
                    self.square_size,
                    self.square_size
                )
                
                highlight = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
                color = self.colors['bot_move'] if len(self.board.move_stack) % 2 == 0 else self.colors['last_move']
                highlight.fill(color)
                self.screen.blit(highlight, rect.topleft)
        
        # Highlight selected square
        if self.selected_square is not None and not self.thinking:
            file = chess.square_file(self.selected_square)
            rank = chess.square_rank(self.selected_square)
            
            rect = pygame.Rect(
                file * self.square_size,
                (7 - rank) * self.square_size,
                self.square_size,
                self.square_size
            )
            
            highlight = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
            highlight.fill(self.colors['selected'])
            self.screen.blit(highlight, rect.topleft)
        
        # Highlight legal moves
        if not self.thinking:
            for move in self.legal_moves:
                file = chess.square_file(move.to_square)
                rank = chess.square_rank(move.to_square)
                
                center_x = file * self.square_size + self.square_size // 2
                center_y = (7 - rank) * self.square_size + self.square_size // 2
                
                pygame.draw.circle(self.screen, self.colors['legal_move'][:3], 
                                 (center_x, center_y), self.square_size // 6)

    def draw_pieces(self):
        """Draw pieces on the board"""
        piece_size = int(self.square_size * 0.8)
        offset = (self.square_size - piece_size) // 2
        
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, rank)
                piece = self.board.piece_at(square)
                
                if piece:
                    color_char = 'w' if piece.color == chess.WHITE else 'b'
                    piece_char = piece.symbol().lower()
                    piece_key = color_char + piece_char
                    
                    if piece_key in self.piece_images:
                        x = file * self.square_size + offset
                        y = (7 - rank) * self.square_size + offset
                        self.screen.blit(self.piece_images[piece_key], (x, y))

    def draw_info_panel(self):
        """Draw the information panel"""
        panel_x = self.board_size + 10
        y = 20
        
        # Title
        title = self.font_large.render("Chess ML Bot", True, self.colors['text'])
        self.screen.blit(title, (panel_x, y))
        y += 40
        
        # Current turn with thinking indicator
        if self.thinking:
            dots = "." * (self.thinking_dots + 1)
            turn_text = f"Bot thinking{dots}"
            color = self.colors['thinking']
        else:
            turn_text = f"Turn: {'White' if self.board.turn == chess.WHITE else 'Black'}"
            color = self.colors['text']
        
        turn_surface = self.font_large.render(turn_text, True, color)
        self.screen.blit(turn_surface, (panel_x, y))
        y += 30
        
        # Move count
        moves_text = f"Moves: {len(self.board.move_stack)}"
        moves_surface = self.font_large.render(moves_text, True, self.colors['text'])
        self.screen.blit(moves_surface, (panel_x, y))
        y += 30
        
        # Last move display
        if self.last_move:
            last_move_text = f"Last: {self.last_move.uci()}"
            last_move_surface = self.font_large.render(last_move_text, True, (255, 255, 0))
            self.screen.blit(last_move_surface, (panel_x, y))
            y += 30
        
        # Game status
        if self.board.is_check():
            check_surface = self.font_large.render("CHECK!", True, (255, 0, 0))
            self.screen.blit(check_surface, (panel_x, y))
            y += 30
        
        # Status message
        if self.thinking:
            status_surface = self.font_small.render("Please wait...", True, self.colors['thinking'])
            self.screen.blit(status_surface, (panel_x, y))
            y += 25
        
        # Controls
        y += 10
        controls = [
            "Controls:",
            "N - New game",
            "U - Undo move" if not self.thinking else "U - Undo (disabled)",
            "",
            "Click squares to move" if not self.thinking else "Bot is thinking...",
            "",
            "Move History:"
        ]
        
        for control in controls:
            if control:
                font = self.font_large if control.endswith(":") else self.font_small
                text_color = self.colors['text']
                if "disabled" in control or "Bot is thinking" in control:
                    text_color = (150, 150, 150)
                surface = font.render(control, True, text_color)
                self.screen.blit(surface, (panel_x, y))
            y += 20
        
        # Move history
        for i, move_text in enumerate(self.move_history[-8:]):
            move_surface = self.font_move.render(move_text, True, self.colors['move_history'])
            self.screen.blit(move_surface, (panel_x, y))
            y += 18


def create_gui(engine=None):
    """Factory function to create GUI instance"""
    try:
        return ChessGUI(engine)
    except Exception as e:
        print(f"Failed to create GUI: {e}")
        return None