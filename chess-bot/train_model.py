#!/usr/bin/env python3
"""
Chess ML Bot Training Script

This script trains the neural network on PGN game databases.
Supports both supervised learning from master games and reinforcement learning.

Usage:
    python train_model.py --pgn <file> [options]

Examples:
    python train_model.py --pgn master_games.pgn
    python train_model.py --pgn lichess_db.pgn --epochs 20 --batch-size 64
    python train_model.py --self-play --games 100
"""

import argparse
import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.neural_net import ChessNet
from training.supervised import SupervisedLearning, train_from_lichess_database
from training.reinforcement import SelfPlayLearning, AlphaZeroTraining
from training.data_loader import DataManager
from utils.config import get_config

def train_supervised(args):
    """Train model using supervised learning from PGN files."""
    print(f"🧠 Starting Supervised Training")
    print(f"📁 PGN File: {args.pgn}")
    print(f"📊 Epochs: {args.epochs}")
    print(f"🔢 Batch Size: {args.batch_size}")
    print("=" * 50)
    
    # Check if PGN file exists
    if not os.path.exists(args.pgn):
        print(f"❌ PGN file not found: {args.pgn}")
        print("\n💡 To get training data:")
        print("   1. Download from Lichess: https://database.lichess.org/")
        print("   2. Download from FICS: http://www.ficsgames.org/")
        print("   3. Use your own games in PGN format")
        return
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Initialize model
    model = ChessNet().to(device)
    print(f"🏗️  Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Load existing model if available
    model_path = "data/models/chess_net.pth"
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✅ Loaded existing model from {model_path}")
        except Exception as e:
            print(f"⚠️  Could not load existing model: {e}")
    
    # Initialize trainer
    trainer = SupervisedLearning(model, device)
    
    try:
        # Train the model
        trainer.train_from_pgn(
            pgn_file=args.pgn,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Save trained model
        os.makedirs("data/models", exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"💾 Model saved to {model_path}")
        
        # Test model on validation data if available
        test_pgn = args.test_pgn or args.pgn
        if os.path.exists(test_pgn):
            print("\n🧪 Evaluating model...")
            trainer.evaluate_on_test_set(test_pgn)
        
        print("\n🎉 Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

def train_self_play(args):
    """Train model using self-play reinforcement learning."""
    print(f"🤖 Starting Self-Play Training")
    print(f"🎮 Number of games: {args.games}")
    print("=" * 50)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Initialize model
    model = ChessNet().to(device)
    
    # Load existing model
    model_path = "data/models/chess_net.pth"
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✅ Loaded existing model from {model_path}")
        except Exception as e:
            print(f"⚠️  Could not load existing model: {e}")
            print("🔄 Starting with random weights")
    
    # Initialize self-play trainer
    if args.alphazero:
        trainer = AlphaZeroTraining(model, device)
        trainer.train_loop(iterations=args.iterations)
    else:
        trainer = SelfPlayLearning(model, device)
        trainer.run_self_play(num_games=args.games)
    
    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved to {model_path}")

def prepare_training_data(args):
    """Prepare and preprocess training data."""
    print(f"📋 Preparing Training Data")
    print(f"📁 Input files: {args.pgn_files}")
    print("=" * 50)
    
    data_manager = DataManager()
    
    # Prepare training data
    train_dataset, val_dataset = data_manager.prepare_training_data(
        pgn_files=args.pgn_files,
        validation_split=0.2
    )
    
    # Show statistics
    train_stats = data_manager.get_data_statistics(train_dataset)
    val_stats = data_manager.get_data_statistics(val_dataset)
    
    print(f"\n📊 Training Data Statistics:")
    print(f"   Training samples: {train_stats['num_samples']}")
    print(f"   Validation samples: {val_stats['num_samples']}")
    print(f"   Win/Draw/Loss distribution: {train_stats['value_distribution']}")
    
    # Validate data quality
    print(f"\n🔍 Validating data quality...")
    is_valid = data_manager.validate_data_quality(train_dataset)
    
    if is_valid:
        print("✅ Data quality check passed")
    else:
        print("⚠️  Data quality issues detected")

def download_sample_data():
    """Download sample training data."""
    print("📥 Downloading Sample Training Data")
    print("=" * 50)
    
    urls = [
        "https://database.lichess.org/standard/lichess_db_standard_rated_2023-01.pgn.bz2",
        "https://database.lichess.org/standard/lichess_db_standard_rated_2023-02.pgn.bz2"
    ]
    
    print("💡 Available training data sources:")
    print("1. Lichess Database: https://database.lichess.org/")
    print("2. FICS Database: http://www.ficsgames.org/")
    print("3. Chess.com games (export from your account)")
    print("4. TWIC (This Week in Chess): https://www.theweekinchess.com/")
    
    print("\n📋 Manual download instructions:")
    print("1. Download PGN files from any source above")
    print("2. Place them in data/opening_books/ directory")
    print("3. Run training: python train_model.py --pgn your_file.pgn")

def main():
    parser = argparse.ArgumentParser(
        description="Train Chess ML Bot on PGN game databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Training modes
    parser.add_argument("--pgn", help="PGN file for supervised training")
    parser.add_argument("--self-play", action="store_true", help="Use self-play training")
    parser.add_argument("--alphazero", action="store_true", help="Full AlphaZero training")
    parser.add_argument("--prepare-data", action="store_true", help="Prepare training data")
    parser.add_argument("--download-data", action="store_true", help="Show data download info")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--games", type=int, default=100, help="Number of self-play games")
    parser.add_argument("--iterations", type=int, default=50, help="AlphaZero iterations")
    
    # Data parameters
    parser.add_argument("--test-pgn", help="Separate PGN file for testing")
    parser.add_argument("--pgn-files", nargs="+", help="Multiple PGN files for data preparation")
    parser.add_argument("--max-games", type=int, help="Maximum games to use from PGN")
    parser.add_argument("--min-rating", type=int, default=2000, help="Minimum player rating")
    
    args = parser.parse_args()
    
    # Show download info
    if args.download_data:
        download_sample_data()
        return
    
    # Prepare data
    if args.prepare_data:
        if not args.pgn_files:
            print("❌ --pgn-files required for data preparation")
            return
        prepare_training_data(args)
        return
    
    # Self-play training
    if args.self_play or args.alphazero:
        train_self_play(args)
        return
    
    # Supervised training
    if args.pgn:
        train_supervised(args)
        return
    
    # No arguments - show help
    print("🤖 Chess ML Bot Training")
    print("=" * 30)
    print("Choose a training mode:")
    print("")
    print("📚 Supervised Learning (from master games):")
    print("   python train_model.py --pgn master_games.pgn")
    print("")
    print("🤖 Self-Play Learning:")
    print("   python train_model.py --self-play --games 100")
    print("")
    print("🧠 Full AlphaZero Training:")
    print("   python train_model.py --alphazero --iterations 20")
    print("")
    print("📥 Download training data:")
    print("   python train_model.py --download-data")
    print("")
    print("For more options: python train_model.py --help")

if __name__ == "__main__":
    main()