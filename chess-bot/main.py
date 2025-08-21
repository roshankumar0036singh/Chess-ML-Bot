
import sys
import argparse
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def setup_environment():
    """Set up the environment and check dependencies."""
    try:
        # Check core dependencies
        import chess
        import torch
        import numpy as np
        
        print("✓ Core dependencies available")
        return True
        
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install required packages:")
        print("  pip install -r requirements.txt")
        return False

def run_setup():
    """Run initial setup wizard."""
    print("Chess ML Bot Setup Wizard")
    print("=" * 40)
    
    # Import and run setup
    try:
        from setup import main as setup_main
        setup_main()
    except ImportError:
        print("Setup script not found. Creating basic directory structure...")
        
        # Create basic directories
        directories = [
            "data/models",
            "data/opening_books", 
            "data/syzygy",
            "logs",
            "saved_games"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created directory: {directory}")
        
        print("\nSetup complete! You can now run the chess bot.")

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Chess ML Bot - Advanced AI Chess Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--interface", 
        choices=["cli", "gui"],
        default="cli",
        help="Choose user interface (default: cli)"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Run initial setup wizard"
    )
    
    parser.add_argument(
        "--train",
        action="store_true", 
        help="Start training mode"
    )
    
    parser.add_argument(
        "--analyze",
        help="Analyze a PGN file"
    )
    
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information"
    )
    
    args = parser.parse_args()
    
    # Show version
    if args.version:
        print("Chess ML Bot v1.0.0")
        print("Advanced AI Chess Engine with Neural Networks")
        return
    
    # Run setup if requested
    if args.setup:
        run_setup()
        return
    
    # Check environment
    if not setup_environment():
        print("\nPlease install dependencies and try again.")
        return
    
    # Initialize configuration
    try:
        from utils.config import get_config, setup_logging
        config = get_config(args.config or "config.json")
        logger = setup_logging()
        
        # Create necessary directories
        config.setup_directories()
        
    except Exception as e:
        print(f"Configuration error: {e}")
        return
    
    # Training mode
    if args.train:
        try:
            from training.supervised import SupervisedLearning
            from core.neural_net import ChessNet
            import torch
            
            print("Starting training mode...")
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = ChessNet()
            trainer = SupervisedLearning(model, device)
            
            # Look for training data
            training_data = "data/opening_books/master_games.pgn"
            if os.path.exists(training_data):
                trainer.train_from_pgn(training_data, epochs=10)
            else:
                print(f"Training data not found: {training_data}")
                print("Place PGN files in data/opening_books/ directory")
                
        except Exception as e:
            print(f"Training error: {e}")
        return
    
    # Analyze mode
    if args.analyze:
        try:
            from analysis.game_analyzer import GameAnalyzer
            from utils.pgn_parser import parse_pgn_games
            
            print(f"Analyzing games from: {args.analyze}")
            
            if not os.path.exists(args.analyze):
                print(f"File not found: {args.analyze}")
                return
            
            analyzer = GameAnalyzer()
            games = parse_pgn_games(args.analyze)
            
            for i, game in enumerate(games[:5]):  # Analyze first 5 games
                print(f"\nAnalyzing game {i+1}...")
                analysis = analyzer.analyze_game(game)
                report = analyzer.generate_analysis_report(analysis)
                print(report)
                
        except Exception as e:
            print(f"Analysis error: {e}")
        return
    
    # Main application
    try:
        # Initialize the chess engine
        from core.engine import ChessEngine
        
        print("Initializing Chess ML Bot...")
        engine = ChessEngine()
        
        # Choose interface
        if args.interface == "gui":
            try:
                from ui.gui import ChessGUI
                print("Starting GUI interface...")
                gui = ChessGUI(engine)
                gui.run()
                
            except ImportError:
                print("GUI dependencies not available. Install pygame:")
                print("  pip install pygame")
                print("Falling back to CLI interface...")
                args.interface = "cli"
            except Exception as e:
                print(f"GUI error: {e}")
                print("Falling back to CLI interface...")
                args.interface = "cli"
        
        if args.interface == "cli":
            from ui.cli import CLIInterface
            print("Starting CLI interface...")
            cli = CLIInterface(engine)
            cli.run()
    
    except KeyboardInterrupt:
        print("\nShutting down Chess ML Bot...")
    
    except Exception as e:
        print(f"Application error: {e}")
        print("\nTry running setup first:")
        print("  python main.py --setup")

def show_welcome():
    """Show welcome message."""
    print("""
    ♔ ♕ ♖ ♗ ♘ ♙    CHESS ML BOT    ♟ ♞ ♝ ♜ ♛ ♚
    
    Advanced AI Chess Engine with Neural Networks
    ============================================
    
    Features:
    • AlphaZero-style neural network evaluation
    • Monte Carlo Tree Search (MCTS)
    • Opening book with master games
    • Endgame tablebase support
    • Advanced position analysis
    • Self-play learning
    
    Commands:
    • python main.py            - Start CLI interface
    • python main.py --gui      - Start GUI interface  
    • python main.py --setup    - Run setup wizard
    • python main.py --train    - Start training
    • python main.py --help     - Show help
    
    """)

if __name__ == "__main__":
    show_welcome()
    main()