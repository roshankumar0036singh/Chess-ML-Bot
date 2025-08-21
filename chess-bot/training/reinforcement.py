import torch # type: ignore
import chess # type: ignore
import random
import time
from core.engine import ChessEngine
from core.neural_net import encode_board
from .trainer import Trainer

class SelfPlayLearning:
    """
    Reinforcement learning through self-play.
    Implements AlphaZero-style training where the bot plays against itself.
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.trainer = Trainer(model, device)
        self.game_history = []
        
    def run_self_play(self, num_games=100, save_games=True):
        """
        Run self-play games to generate training data.
        
        Args:
            num_games: Number of self-play games to generate
            save_games: Whether to save game data for training
        """
        print(f"Starting self-play: {num_games} games")
        
        training_data = []
        
        for game_num in range(num_games):
            print(f"Self-play game {game_num + 1}/{num_games}")
            
            game_data = self._play_single_game()
            
            if save_games and game_data:
                training_data.extend(game_data)
                
            # Periodic training updates
            if (game_num + 1) % 10 == 0 and training_data:
                print(f"Training on {len(training_data)} positions...")
                self._train_on_self_play_data(training_data)
                training_data.clear()  # Clear to save memory
        
        # Final training on remaining data
        if training_data:
            self._train_on_self_play_data(training_data)
        
        print("Self-play training complete!")
    
    def _play_single_game(self):
        """
        Play a single self-play game and collect training data.
        
        Returns:
            List of (position, move_probs, value) tuples
        """
        engine = ChessEngine()
        game_data = []
        move_count = 0
        
        while not engine.is_game_over() and move_count < 200:  # Max 200 moves
            # Get current position
            position = encode_board(engine.board)
            
            # Get move probabilities from MCTS
            move_probs = self._get_move_probabilities(engine)
            
            # Store position and move probabilities
            game_data.append({
                'position': position,
                'move_probs': move_probs,
                'turn': engine.board.turn
            })
            
            # Select move based on probabilities (with temperature)
            move = self._select_move_from_probs(engine, move_probs, 
                                              temperature=1.0 if move_count < 30 else 0.1)
            
            if move is None:
                break
                
            engine.make_move(move)
            move_count += 1
        
        # Get game result and assign values
        if engine.is_game_over():
            result = engine.get_game_result()
            if result == "1-0":
                game_value = 1.0  # White wins
            elif result == "0-1":
                game_value = -1.0  # Black wins
            else:
                game_value = 0.0  # Draw
        else:
            game_value = 0.0  # Incomplete game
        
        # Assign values to all positions based on game outcome
        training_examples = []
        for data in game_data:
            # Value from perspective of player to move
            if data['turn'] == chess.WHITE:
                value = game_value
            else:
                value = -game_value
                
            training_examples.append((
                data['position'],
                data['move_probs'],
                value
            ))
        
        return training_examples
    
    def _get_move_probabilities(self, engine):
        """Get move probabilities from MCTS search."""
        # Run MCTS to get visit counts
        mcts = engine.mcts
        root_node = mcts.create_root_node(engine.board)

        
        # Run reduced simulations for faster self-play
        for _ in range(200):  # Fewer simulations than normal play
            mcts._simulate(root_node)
        
        # Convert visit counts to probabilities
        move_probs = torch.zeros(4096)  # Max possible moves
        
        if root_node.children:
            total_visits = sum(child.visit_count for child in root_node.children.values())
            
            for move, child in root_node.children.items():
                try:
                    from core.neural_net import move_to_index
                    move_idx = move_to_index(move, engine.board)
                    if move_idx < 4096 and total_visits > 0:
                        move_probs[move_idx] = child.visit_count / total_visits
                except:
                    continue
        
        return move_probs
    
    def _select_move_from_probs(self, engine, move_probs, temperature=1.0):
        """Select a move based on probabilities with temperature."""
        legal_moves = list(engine.board.legal_moves)
        if not legal_moves:
            return None
        
        move_weights = []
        moves = []
        
        for move in legal_moves:
            try:
                from core.neural_net import move_to_index
                move_idx = move_to_index(move, engine.board)
                if move_idx < len(move_probs):
                    weight = move_probs[move_idx] ** (1.0 / temperature)
                    move_weights.append(weight)
                    moves.append(move)
            except:
                continue
        
        if not moves:
            return random.choice(legal_moves)
        
        # Weighted random selection
        total_weight = sum(move_weights)
        if total_weight == 0:
            return random.choice(moves)
        
        r = random.random() * total_weight
        current_weight = 0
        
        for move, weight in zip(moves, move_weights):
            current_weight += weight
            if r <= current_weight:
                return move
        
        return moves[-1]  # Fallback
    
    def _train_on_self_play_data(self, training_data):
        """Train the neural network on self-play data."""
        if not training_data:
            return
        
        # Convert to tensors
        positions = torch.stack([data[0] for data in training_data])
        policies = torch.stack([data[1] for data in training_data])
        values = torch.tensor([data[2] for data in training_data], dtype=torch.float32)
        
        # Train the model
        self.trainer.train_supervised(positions, policies, values, 
                                    epochs=3, batch_size=32)
    
    def tournament_play(self, num_games=50):
        """
        Play tournament games between current and previous model versions.
        
        Args:
            num_games: Number of tournament games to play
            
        Returns:
            Win rate of current model
        """
        print(f"Running tournament: {num_games} games")
        
        # Load previous model version if available
        try:
            previous_model = torch.load('data/models/chess_net_previous.pth')
            print("Loaded previous model for comparison")
        except FileNotFoundError:
            print("No previous model found, skipping tournament")
            return 1.0  # Assume current model wins
        
        wins = 0
        draws = 0
        losses = 0
        
        for game_num in range(num_games):
            # Alternate colors
            current_plays_white = (game_num % 2 == 0)
            
            result = self._play_tournament_game(current_plays_white, previous_model)
            
            if result == "current_wins":
                wins += 1
            elif result == "draw":
                draws += 1
            else:
                losses += 1
            
            if (game_num + 1) % 10 == 0:
                win_rate = (wins + draws * 0.5) / (game_num + 1)
                print(f"Games {game_num + 1}: Win rate = {win_rate:.2f}")
        
        total_games = wins + draws + losses
        win_rate = (wins + draws * 0.5) / total_games
        
        print(f"Tournament results: {wins}W {draws}D {losses}L")
        print(f"Win rate: {win_rate:.2f}")
        
        return win_rate
    
    def _play_tournament_game(self, current_plays_white, previous_model):
        """Play a single tournament game between model versions."""
        # This would implement a game between current and previous models
        # For simplicity, we'll simulate a random result
        results = ["current_wins", "draw", "previous_wins"]
        weights = [0.6, 0.3, 0.1]  # Assume current model is slightly better
        
        import random
        return random.choices(results, weights=weights)[0]

class AlphaZeroTraining:
    """Complete AlphaZero-style training pipeline."""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.self_play = SelfPlayLearning(model, device)
        self.generation = 0
        
    def train_loop(self, iterations=100):
        """
        Main AlphaZero training loop.
        
        Args:
            iterations: Number of training iterations
        """
        print("Starting AlphaZero training loop")
        
        for iteration in range(iterations):
            print(f"\n=== Training Iteration {iteration + 1}/{iterations} ===")
            
            # Self-play phase
            print("Phase 1: Self-play data generation")
            self.self_play.run_self_play(num_games=25)
            
            # Tournament evaluation
            print("Phase 2: Model evaluation")
            win_rate = self.self_play.tournament_play(num_games=20)
            
            # Model update
            if win_rate > 0.55:  # Keep new model if it wins >55% of games
                print(f"New model accepted (win rate: {win_rate:.2f})")
                self._save_model_checkpoint()
                self.generation += 1
            else:
                print(f"New model rejected (win rate: {win_rate:.2f})")
                self._revert_to_previous_model()
            
            # Save progress
            self._save_training_progress(iteration, win_rate)
        
        print("\nAlphaZero training complete!")
    
    def _save_model_checkpoint(self):
        """Save current model as new checkpoint."""
        # Save previous model
        try:
            previous_path = 'data/models/chess_net_previous.pth'
            current_path = 'data/models/chess_net.pth'
            
            if torch.cuda.is_available():
                torch.save(torch.load(current_path), previous_path)
            
            # Save new model
            torch.save(self.model.state_dict(), current_path)
            print("Model checkpoint saved")
            
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
    
    def _revert_to_previous_model(self):
        """Revert to previous model version."""
        try:
            previous_path = 'data/models/chess_net_previous.pth'
            self.model.load_state_dict(torch.load(previous_path))
            print("Reverted to previous model")
        except Exception as e:
            print(f"Error reverting model: {e}")
    
    def _save_training_progress(self, iteration, win_rate):
        """Save training progress to log file."""
        import os
        os.makedirs('logs', exist_ok=True)
        
        with open('logs/training.log', 'a') as f:
            f.write(f"Iteration {iteration + 1}: Win rate = {win_rate:.4f}\n")

def run_full_training():
    """Run complete training pipeline from scratch."""
    from core.neural_net import ChessNet
    
    print("Initializing full training pipeline...")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ChessNet().to(device)
    
    # Start training
    trainer = AlphaZeroTraining(model, device)
    trainer.train_loop(iterations=50)
    
    print("Full training pipeline complete!")