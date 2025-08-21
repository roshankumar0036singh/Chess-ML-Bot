import torch # type: ignore
import numpy as np
from utils.pgn_parser import parse_pgn_games
from .trainer import Trainer

class SupervisedLearning:
    """
    Supervised learning from master games.
    Trains neural network on positions and moves from high-level play.
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.trainer = Trainer(model, device)
        
    def train_from_pgn(self, pgn_file, epochs=20, batch_size=32):
        """
        Train model from PGN file of master games.
        
        Args:
            pgn_file: Path to PGN file
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        print(f"Loading games from {pgn_file}...")
        games = parse_pgn_games(pgn_file)
        
        # Filter games by rating (only strong players)
        filtered_games = []
        for game in games:
            white_elo = game.headers.get("WhiteElo", "0")
            black_elo = game.headers.get("BlackElo", "0")
            
            try:
                if int(white_elo) >= 2000 and int(black_elo) >= 2000:
                    filtered_games.append(game)
            except ValueError:
                continue
        
        print(f"Using {len(filtered_games)} games from strong players")
        
        # Extract training data
        positions, policies, values = self._extract_training_data(filtered_games)
        
        # Train the model
        self.trainer.train_supervised(positions, policies, values, epochs, batch_size)
        
    def _extract_training_data(self, games):
        """Extract (position, policy, value) tuples from games."""
        from core.neural_net import encode_board
        import chess # type: ignore
        
        positions = []
        policies = []
        values = []
        
        for game in games:
            board = game.board()
            moves = list(game.mainline_moves())
            
            # Get game result
            result = game.headers.get("Result", "*")
            if result == "1-0":
                game_value = 1.0
            elif result == "0-1":
                game_value = -1.0
            else:
                game_value = 0.0
            
            # Process first 30 moves (opening/middlegame)
            for i, move in enumerate(moves[:30]):
                if board.is_game_over():
                    break
                
                # Encode position
                position_tensor = encode_board(board)
                positions.append(position_tensor)
                
                # Create policy (one-hot for played move)
                policy = torch.zeros(4096)  # Max possible moves
                try:
                    from core.neural_net import move_to_index
                    move_idx = move_to_index(move, board)
                    if move_idx < 4096:
                        policy[move_idx] = 1.0
                except:
                    pass  # Skip if move encoding fails
                
                policies.append(policy)
                
                # Value based on game result and position
                if board.turn == chess.WHITE:
                    value = game_value
                else:
                    value = -game_value
                
                values.append(value)
                
                board.push(move)
        
        return (torch.stack(positions), 
                torch.stack(policies), 
                torch.tensor(values, dtype=torch.float32))
    
    def evaluate_on_test_set(self, test_pgn_file):
        """Evaluate model performance on test positions."""
        print(f"Evaluating on {test_pgn_file}...")
        
        games = parse_pgn_games(test_pgn_file)
        positions, policies, values = self._extract_training_data(games[:100])  # Sample
        
        self.model.eval()
        with torch.no_grad():
            pred_policies, pred_values = self.model(positions.to(self.device))
            
            # Calculate accuracy
            policy_loss = torch.nn.functional.kl_div(pred_policies, policies.to(self.device))
            value_loss = torch.nn.functional.mse_loss(pred_values.squeeze(), values.to(self.device))
            
            print(f"Test Policy Loss: {policy_loss:.4f}")
            print(f"Test Value Loss: {value_loss:.4f}")
            
        return policy_loss.item(), value_loss.item()

def train_from_lichess_database(model, database_path, num_games=10000):
    """
    Train model from Lichess database.
    
    Args:
        model: Neural network model
        database_path: Path to Lichess PGN database
        num_games: Number of games to use for training
    """
    supervisor = SupervisedLearning(model)
    
    print(f"Training from Lichess database: {database_path}")
    print(f"Using {num_games} games...")
    
    # This would process a large Lichess database file
    # In practice, you'd stream the file to handle large databases
    supervisor.train_from_pgn(database_path, epochs=20, batch_size=64)
    
    print("Supervised training complete!")

def create_training_dataset(pgn_files, output_file):
    """
    Create a preprocessed training dataset from multiple PGN files.
    
    Args:
        pgn_files: List of PGN file paths
        output_file: Output file for processed dataset
    """
    all_positions = []
    all_policies = []
    all_values = []
    
    supervisor = SupervisedLearning(None)  # No model needed for data extraction
    
    for pgn_file in pgn_files:
        print(f"Processing {pgn_file}...")
        games = parse_pgn_games(pgn_file)
        positions, policies, values = supervisor._extract_training_data(games)
        
        all_positions.append(positions)
        all_policies.append(policies)
        all_values.append(values)
    
    # Combine all data
    combined_positions = torch.cat(all_positions, dim=0)
    combined_policies = torch.cat(all_policies, dim=0)
    combined_values = torch.cat(all_values, dim=0)
    
    # Save dataset
    torch.save({
        'positions': combined_positions,
        'policies': combined_policies,
        'values': combined_values
    }, output_file)
    
    print(f"Dataset saved to {output_file}")
    print(f"Total positions: {len(combined_positions)}")

def load_training_dataset(dataset_file):
    """Load preprocessed training dataset."""
    data = torch.load(dataset_file)
    return data['positions'], data['policies'], data['values']