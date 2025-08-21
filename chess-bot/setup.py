

import os
import sys
import subprocess
import shutil
from pathlib import Path

def create_directory_structure():
    """Create the complete directory structure."""
    directories = [
        # Core package directories
        "core",
        "training", 
        "features",
        "analysis",
        "ui",
        "utils",
        
        # Data storage directories
        "data",
        "data/models",
        "data/opening_books",
        "data/syzygy", 
        "data/training_data",
        "data/training_data/processed",
        "data/training_data/raw",
        
        # Logging directories
        "logs",
        "logs/training",
        "logs/games",
        
        # Output directories
        "saved_games",
        "saved_games/tournament_games",
        
        # Optional directories
        "tests",
        "docs", 
        "scripts",
        "assets",
        "assets/pieces"
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")

def create_package_init_files():
    """Create __init__.py files for all Python packages."""
    
    package_inits = {
        "__init__.py": '"""Chess ML Bot - Advanced AI Chess Engine"""\n\n__version__ = "1.0.0"\n',
        
        "core/__init__.py": '''"""Core package for Chess ML Bot"""

try:
    from .engine import ChessEngine
    from .neural_net import ChessNet, encode_board, decode_moves, move_to_index
    from .search import MCTSSearch, MinimaxSearch, SearchNode
    from .evaluation import PositionEvaluator
except ImportError as e:
    print(f"Warning: Could not import core modules: {e}")

__all__ = [
    'ChessEngine',
    'ChessNet', 
    'encode_board',
    'decode_moves',
    'move_to_index',
    'MCTSSearch',
    'MinimaxSearch',
    'SearchNode',
    'PositionEvaluator'
]
''',
        
        "training/__init__.py": '''"""Training package for Chess ML Bot"""

try:
    from .supervised import SupervisedLearning, train_from_lichess_database
    from .reinforcement import SelfPlayLearning, AlphaZeroTraining
    from .data_loader import DataLoader, DataManager, ChessDataset
    from .trainer import Trainer
except ImportError as e:
    print(f"Warning: Could not import training modules: {e}")

__all__ = [
    'SupervisedLearning',
    'SelfPlayLearning', 
    'AlphaZeroTraining',
    'DataLoader',
    'DataManager',
    'ChessDataset',
    'Trainer',
    'train_from_lichess_database'
]
''',
        
        "features/__init__.py": '''"""Features package for Chess ML Bot"""

try:
    from .opening_book import OpeningBook, OpeningAnalyzer, OpeningTrainer
    from .tablebase import TablebaseBot, EndgameKnowledge
    from .time_manager import TimeManager
    from .opponent_model import OpponentModel
except ImportError as e:
    print(f"Warning: Could not import feature modules: {e}")

__all__ = [
    'OpeningBook',
    'OpeningAnalyzer', 
    'OpeningTrainer',
    'TablebaseBot',
    'EndgameKnowledge',
    'TimeManager',
    'OpponentModel'
]
''',
        
        "analysis/__init__.py": '''"""Analysis package for Chess ML Bot"""

try:
    from .game_analyzer import GameAnalyzer
    from .position_db import PositionDB
    from .statistics import Statistics
except ImportError as e:
    print(f"Warning: Could not import analysis modules: {e}")

__all__ = [
    'GameAnalyzer',
    'PositionDB',
    'Statistics'
]
''',
        
        "ui/__init__.py": '''"""UI package for Chess ML Bot"""

try:
    from .cli import CLIInterface
    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False
    CLIInterface = None

# Conditional imports for optional dependencies
try:
    from .gui import ChessGUI
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    ChessGUI = None

try:
    from .web_interface import create_web_interface
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False
    create_web_interface = None

__all__ = [
    'CLIInterface',
    'ChessGUI',
    'create_web_interface',
    'CLI_AVAILABLE',
    'GUI_AVAILABLE',
    'WEB_AVAILABLE'
]
''',
        
        "utils/__init__.py": '''"""Utils package for Chess ML Bot"""

try:
    from .pgn_parser import (
        parse_pgn_games, 
        parse_pgn_stream,
        filter_games_by_rating,
        extract_opening_moves,
        analyze_pgn_database
    )
    
    from .fen_utils import (
        fen_to_board,
        board_to_fen,
        normalize_fen,
        validate_fen,
        starting_position_fen
    )
    
    from .config import (
        Config,
        get_config,
        get_engine_depth,
        get_mcts_simulations,
        get_training_device
    )
except ImportError as e:
    print(f"Warning: Could not import utility modules: {e}")

__all__ = [
    # PGN utilities
    'parse_pgn_games',
    'parse_pgn_stream', 
    'filter_games_by_rating',
    'extract_opening_moves',
    'analyze_pgn_database',
    
    # FEN utilities
    'fen_to_board',
    'board_to_fen',
    'normalize_fen', 
    'validate_fen',
    'starting_position_fen',
    
    # Configuration
    'Config',
    'get_config',
    'get_engine_depth',
    'get_mcts_simulations', 
    'get_training_device'
]
'''
    }
    
    print("Creating package initialization files...")
    for file_path, content in package_inits.items():
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"  ✓ {file_path}")

def organize_downloaded_files():
    """Organize downloaded files into correct directory structure."""
    
    # File organization mapping
    file_mappings = {
        # Core files remain in root
        "main.py": "main.py",
        "setup.py": "setup.py",
        "requirements.txt": "requirements.txt", 
        "README.md": "README.md",
        "train_model.py": "train_model.py",
        
        # Data files
        "sample_games.pgn": "data/opening_books/sample_games.pgn",
        
        # Core package files - only move if they exist in root
        "engine.py": "core/engine.py",
        "neural_net.py": "core/neural_net.py",
        "search.py": "core/search.py", 
        "evaluation.py": "core/evaluation.py",
        
        # Training files
        "supervised.py": "training/supervised.py",
        "reinforcement.py": "training/reinforcement.py",
        "data_loader.py": "training/data_loader.py",
        "trainer.py": "training/trainer.py",
        
        # Feature files
        "opening_book.py": "features/opening_book.py",
        "tablebase.py": "features/tablebase.py",
        "time_manager.py": "features/time_manager.py",
        "opponent_model.py": "features/opponent_model.py",
        
        # Analysis files
        "game_analyzer.py": "analysis/game_analyzer.py", 
        "position_db.py": "analysis/position_db.py",
        "statistics.py": "analysis/statistics.py",
        
        # UI files
        "cli.py": "ui/cli.py",
        "gui.py": "ui/gui.py",
        "web_interface.py": "ui/web_interface.py",
        
        # Utility files
        "pgn_parser.py": "utils/pgn_parser.py",
        "fen_utils.py": "utils/fen_utils.py",
        "config.py": "utils/config.py"
    }
    
    print("Organizing files into directory structure...")
    organized_count = 0
    
    for source_file, target_file in file_mappings.items():
        if os.path.exists(source_file) and source_file != target_file:
            # Create target directory
            target_dir = os.path.dirname(target_file)
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)
            
            # Move file
            shutil.move(source_file, target_file)
            print(f"  ✓ Moved {source_file} → {target_file}")
            organized_count += 1
        elif os.path.exists(target_file):
            print(f"  ✓ {target_file} already in place")
    
    if organized_count > 0:
        print(f"Organized {organized_count} files into proper structure")
    else:
        print("All files already in correct locations")

def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    if sys.version_info < (3, 7):
        print("  ✗ Python 3.7+ required. Current version:", sys.version)
        return False
    else:
        print(f"  ✓ Python {sys.version.split()[0]} detected")
        return True

def install_dependencies():
    """Install required Python packages."""
    print("Installing dependencies...")
    
    if not os.path.exists('requirements.txt'):
        print("  ✗ requirements.txt not found!")
        return False
    
    try:
        # Install packages
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("  ✓ All dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error installing dependencies: {e}")
        print("  Please install manually: pip install -r requirements.txt")
        return False

def create_sample_data():
    """Create sample data files if they don't exist."""
    print("Setting up sample data...")
    
    # Create sample opening book if it doesn't exist
    sample_book_path = "data/opening_books/sample_games.pgn"
    if not os.path.exists(sample_book_path):
        sample_content = '''[Event "Sample Game 1"]
[Site "Setup"]
[Date "2025.01.01"]
[Round "1"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "2200"]
[BlackElo "2180"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0

[Event "Sample Game 2"] 
[Site "Setup"]
[Date "2025.01.01"]
[Round "2"]
[White "Player3"]
[Black "Player4"]
[Result "1/2-1/2"]
[WhiteElo "2300"]
[BlackElo "2250"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. cxd5 exd5 5. Bg5 c6 6. e3 Bf5 7. Qf3 Bg6 8. Bxf6 Qxf6 9. Qxf6 gxf6 1/2-1/2
'''
        with open(sample_book_path, 'w') as f:
            f.write(sample_content)
        print(f"  ✓ Created sample opening book: {sample_book_path}")

def create_default_config():
    """Create default configuration file."""
    config_path = "config.json"
    if not os.path.exists(config_path):
        config_content = '''{
  "engine": {
    "search_depth": 4,
    "mcts_simulations": 800,
    "time_per_move": 5.0,
    "use_opening_book": true,
    "use_tablebase": true,
    "neural_network_enabled": true
  },
  "training": {
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 10,
    "device": "auto",
    "data_augmentation": true,
    "validation_split": 0.2
  },
  "neural_network": {
    "input_channels": 14,
    "residual_blocks": 19,
    "filters": 256,
    "policy_head_filters": 32,
    "value_head_filters": 1
  },
  "paths": {
    "models_dir": "data/models",
    "opening_book_dir": "data/opening_books",
    "tablebase_dir": "data/syzygy",
    "training_data_dir": "data/training_data",
    "logs_dir": "logs",
    "saved_games_dir": "saved_games"
  },
  "ui": {
    "default_interface": "cli",
    "show_coordinates": true,
    "show_legal_moves": true
  }
}'''
        with open(config_path, 'w') as f:
            f.write(config_content)
        print(f"  ✓ Created configuration file: {config_path}")

def test_installation():
    """Test if the installation works correctly."""
    print("Testing installation...")
    
    try:
        # Test core imports
        import chess
        print("  ✓ python-chess import successful")
        
        import torch
        print("  ✓ PyTorch import successful")
        
        import numpy
        print("  ✓ NumPy import successful")
        
        # Test basic chess functionality
        board = chess.Board()
        moves = list(board.legal_moves)
        print(f"  ✓ Chess engine working ({len(moves)} legal moves in starting position)")
        
        return True
        
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Test error: {e}")
        return False

def print_next_steps():
    """Print instructions for next steps."""
    print("\n" + "="*60)
    print("🎉 CHESS ML BOT SETUP COMPLETE!")
    print("="*60)
    
    print("\n📁 Project Structure Created:")
    print("├── main.py                    # Start the bot")
    print("├── train_model.py             # Train neural network")
    print("├── core/                      # Core engine components")
    print("├── training/                  # ML training modules")
    print("├── features/                  # Advanced chess features")
    print("├── analysis/                  # Game analysis tools")
    print("├── ui/                        # User interfaces")
    print("├── utils/                     # Utility functions")
    print("└── data/                      # Data storage")
    
    print("\n🚀 Quick Start:")
    print("1. Start playing:    python main.py")
    print("2. Start training:   python train_model.py --pgn data/opening_books/sample_games.pgn")
    print("3. Use GUI:          python main.py --interface gui")
    print("4. Self-play:        python train_model.py --self-play --games 50")
    
    print("\n📚 For Better Performance:")
    print("• Download larger PGN databases from https://database.lichess.org/")
    print("• Place PGN files in data/opening_books/ directory")
    print("• Train with: python train_model.py --pgn your_large_database.pgn")
    print("• Download Syzygy tablebases for perfect endgame play")
    
    print("\n💡 Available Commands in CLI:")
    print("• help          - Show all commands")
    print("• new           - Start new game")  
    print("• e2e4          - Make moves")
    print("• analyze       - Analyze position")
    print("• auto          - Watch bot play itself")
    print("• training      - Enter training mode")
    
    print("\n📖 Documentation:")
    print("• README.md               - Complete user guide")
    print("• folder-structure.md     - Project organization")
    print("• config.json             - Configuration settings")
    
    print("\n" + "="*60)
    print("Ready to play chess! 🚀")
    print("="*60)

def main():
    """Main setup function."""
    print("♔ ♕ ♖ ♗ ♘ ♙ Chess ML Bot Setup ♟ ♞ ♝ ♜ ♛ ♚")
    print("=" * 50)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Create directory structure
    create_directory_structure()
    
    # Create package files
    create_package_init_files()
    
    # Organize downloaded files
    organize_downloaded_files()
    
    # Install dependencies
    if success and not install_dependencies():
        success = False
    
    # Create sample data
    create_sample_data()
    
    # Create configuration
    create_default_config()
    
    # Test installation
    if success and test_installation():
        print("\n✅ All tests passed!")
    else:
        print("\n⚠️  Some tests failed - check errors above")
        success = False
    
    # Print next steps
    print_next_steps()
    
    if not success:
        print("\n⚠️  Setup completed with warnings.")
        print("Check error messages above and install missing dependencies.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())