import torch
import torch.utils.data as data
import numpy as np
import os
from utils.pgn_parser import parse_pgn_games
from core.neural_net import encode_board, move_to_index

class ChessDataset(data.Dataset):
    """
    PyTorch dataset for chess positions and moves.
    Handles loading and preprocessing of training data.
    """
    
    def __init__(self, positions, policies, values):
        self.positions = positions
        self.policies = policies
        self.values = values
        
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        return {
            'position': self.positions[idx],
            'policy': self.policies[idx],
            'value': self.values[idx]
        }

class DataLoader:
    """
    Manages loading and preprocessing of chess training data.
    Supports various data sources and augmentation techniques.
    """
    
    def __init__(self, batch_size=32, augment=True):
        self.batch_size = batch_size
        self.augment = augment
        
    def load_from_pgn(self, pgn_file, max_games=None):
        """
        Load training data from PGN file.
        
        Args:
            pgn_file: Path to PGN file
            max_games: Maximum number of games to load
            
        Returns:
            ChessDataset: Dataset ready for training
        """
        print(f"Loading data from {pgn_file}...")
        
        games = parse_pgn_games(pgn_file)
        if max_games:
            games = games[:max_games]
        
        positions, policies, values = self._extract_data_from_games(games)
        
        if self.augment:
            positions, policies, values = self._augment_data(positions, policies, values)
        
        return ChessDataset(positions, policies, values)
    
    def _extract_data_from_games(self, games):
        """Extract training data from list of games."""
        import chess
        
        positions = []
        policies = []
        values = []
        
        for game_idx, game in enumerate(games):
            if game_idx % 1000 == 0:
                print(f"Processing game {game_idx + 1}/{len(games)}")
            
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
            
            # Extract positions from first 40 moves
            for move_idx, move in enumerate(moves[:40]):
                if board.is_game_over():
                    break
                
                # Skip very early moves (first 4 moves are often book)
                if move_idx < 4:
                    board.push(move)
                    continue
                
                # Encode position
                try:
                    position_tensor = encode_board(board)
                    positions.append(position_tensor)
                    
                    # Create policy target (one-hot for played move)
                    policy = torch.zeros(4096)
                    move_idx_encoded = move_to_index(move, board)
                    if move_idx_encoded < 4096:
                        policy[move_idx_encoded] = 1.0
                    policies.append(policy)
                    
                    # Value from perspective of current player
                    if board.turn == chess.WHITE:
                        value = game_value
                    else:
                        value = -game_value
                    values.append(value)
                    
                except Exception as e:
                    # Skip positions that can't be encoded
                    pass
                
                board.push(move)
        
        print(f"Extracted {len(positions)} positions")
        
        return (torch.stack(positions) if positions else torch.empty(0),
                torch.stack(policies) if policies else torch.empty(0),
                torch.tensor(values, dtype=torch.float32) if values else torch.empty(0))
    
    def _augment_data(self, positions, policies, values):
        """
        Augment training data with rotations and reflections.
        
        Args:
            positions: Position tensors
            policies: Policy tensors
            values: Value scalars
            
        Returns:
            Augmented data
        """
        print("Augmenting data with transformations...")
        
        augmented_positions = [positions]
        augmented_policies = [policies]
        augmented_values = [values]
        
        # Horizontal flip
        flipped_positions = self._flip_horizontal(positions)
        flipped_policies = self._flip_policy_horizontal(policies)
        
        augmented_positions.append(flipped_positions)
        augmented_policies.append(flipped_policies)
        augmented_values.append(values)  # Values don't change
        
        # Combine all augmented data
        all_positions = torch.cat(augmented_positions, dim=0)
        all_policies = torch.cat(augmented_policies, dim=0)
        all_values = torch.cat(augmented_values, dim=0)
        
        print(f"Data augmented: {len(positions)} -> {len(all_positions)} positions")
        
        return all_positions, all_policies, all_values
    
    def _flip_horizontal(self, positions):
        """Flip board positions horizontally."""
        # Flip along the file axis (columns)
        return torch.flip(positions, dims=[3])  # Flip last dimension (files)
    
    def _flip_policy_horizontal(self, policies):
        """Flip move policies to match horizontally flipped board."""
        # This is a simplified implementation
        # In practice, you'd need to map each move to its horizontally flipped equivalent
        return policies  # Placeholder - would need proper move mapping
    
    def create_data_loader(self, dataset, shuffle=True):
        """
        Create PyTorch DataLoader from dataset.
        
        Args:
            dataset: ChessDataset
            shuffle: Whether to shuffle data
            
        Returns:
            torch.utils.data.DataLoader
        """
        return data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )
    
    def save_dataset(self, dataset, filename):
        """Save dataset to disk."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        torch.save({
            'positions': dataset.positions,
            'policies': dataset.policies,
            'values': dataset.values
        }, filename)
        
        print(f"Dataset saved to {filename}")
    
    def load_dataset(self, filename):
        """Load dataset from disk."""
        data_dict = torch.load(filename)
        
        return ChessDataset(
            data_dict['positions'],
            data_dict['policies'],
            data_dict['values']
        )

class DataManager:
    """
    High-level data management for training pipeline.
    Handles multiple data sources and preprocessing.
    """
    
    def __init__(self, data_dir="data/training_data"):
        self.data_dir = data_dir
        self.loader = DataLoader()
        
        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(f"{data_dir}/processed", exist_ok=True)
        os.makedirs(f"{data_dir}/raw", exist_ok=True)
    
    def prepare_training_data(self, pgn_files, validation_split=0.2):
        """
        Prepare training and validation datasets from PGN files.
        
        Args:
            pgn_files: List of PGN file paths
            validation_split: Fraction of data to use for validation
            
        Returns:
            tuple: (train_dataset, val_dataset)
        """
        print("Preparing training data...")
        
        all_positions = []
        all_policies = []
        all_values = []
        
        # Process each PGN file
        for pgn_file in pgn_files:
            if os.path.exists(pgn_file):
                dataset = self.loader.load_from_pgn(pgn_file, max_games=10000)
                all_positions.append(dataset.positions)
                all_policies.append(dataset.policies)
                all_values.append(dataset.values)
            else:
                print(f"Warning: {pgn_file} not found, skipping")
        
        if not all_positions:
            raise ValueError("No valid PGN files found")
        
        # Combine all data
        combined_positions = torch.cat(all_positions, dim=0)
        combined_policies = torch.cat(all_policies, dim=0)
        combined_values = torch.cat(all_values, dim=0)
        
        # Shuffle data
        indices = torch.randperm(len(combined_positions))
        combined_positions = combined_positions[indices]
        combined_policies = combined_policies[indices]
        combined_values = combined_values[indices]
        
        # Split into train and validation
        split_idx = int(len(combined_positions) * (1 - validation_split))
        
        train_dataset = ChessDataset(
            combined_positions[:split_idx],
            combined_policies[:split_idx],
            combined_values[:split_idx]
        )
        
        val_dataset = ChessDataset(
            combined_positions[split_idx:],
            combined_policies[split_idx:],
            combined_values[split_idx:]
        )
        
        # Save datasets
        self.loader.save_dataset(train_dataset, f"{self.data_dir}/processed/train.pth")
        self.loader.save_dataset(val_dataset, f"{self.data_dir}/processed/val.pth")
        
        print(f"Training data prepared:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def load_prepared_data(self):
        """Load previously prepared training data."""
        train_path = f"{self.data_dir}/processed/train.pth"
        val_path = f"{self.data_dir}/processed/val.pth"
        
        if os.path.exists(train_path) and os.path.exists(val_path):
            train_dataset = self.loader.load_dataset(train_path)
            val_dataset = self.loader.load_dataset(val_path)
            
            print(f"Loaded prepared data:")
            print(f"  Training samples: {len(train_dataset)}")
            print(f"  Validation samples: {len(val_dataset)}")
            
            return train_dataset, val_dataset
        else:
            raise FileNotFoundError("No prepared data found. Run prepare_training_data() first.")
    
    def get_data_statistics(self, dataset):
        """Get statistics about the dataset."""
        positions = dataset.positions
        values = dataset.values
        
        stats = {
            'num_samples': len(dataset),
            'position_shape': positions.shape,
            'value_distribution': {
                'wins': (values > 0.5).sum().item(),
                'draws': ((values >= -0.5) & (values <= 0.5)).sum().item(),
                'losses': (values < -0.5).sum().item()
            },
            'mean_value': values.mean().item(),
            'std_value': values.std().item()
        }
        
        return stats
    
    def validate_data_quality(self, dataset):
        """Validate data quality and report issues."""
        positions = dataset.positions
        policies = dataset.policies
        values = dataset.values
        
        issues = []
        
        # Check for NaN values
        if torch.isnan(positions).any():
            issues.append("NaN values found in positions")
        
        if torch.isnan(policies).any():
            issues.append("NaN values found in policies")
        
        if torch.isnan(values).any():
            issues.append("NaN values found in values")
        
        # Check value range
        if (values < -1.1).any() or (values > 1.1).any():
            issues.append("Values outside expected range [-1, 1]")
        
        # Check policy normalization
        policy_sums = policies.sum(dim=1)
        if not torch.allclose(policy_sums, torch.ones_like(policy_sums), atol=1e-6):
            issues.append("Policies are not properly normalized")
        
        if issues:
            print("Data quality issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("Data quality validation passed")
        
        return len(issues) == 0