# Chess ML Bot

A sophisticated machine learning chess bot that combines traditional chess programming techniques with modern AI approaches. Features neural network position evaluation, Monte Carlo Tree Search, opening book knowledge, and endgame tablebase support.

## Features

### Core Engine
- **Neural Network Evaluation**: AlphaZero-style CNN for position assessment
- **Monte Carlo Tree Search (MCTS)**: Advanced search algorithm with neural network guidance
- **Minimax with Alpha-Beta**: Traditional search as fallback
- **Opening Book**: Master game database for strong opening play
- **Endgame Tablebases**: Perfect play in positions with 7 or fewer pieces
- **Advanced Evaluation**: Material, positional, tactical, and strategic factors

### Learning Capabilities
- **Supervised Learning**: Train from master game databases
- **Reinforcement Learning**: Self-play improvement
- **Opponent Modeling**: Adapt to different playing styles
- **Continuous Learning**: Update from new games

### User Interface
- **Command Line Interface**: Interactive text-based play
- **Analysis Tools**: Position evaluation and move analysis
- **Training Modes**: Opening practice and tactical training
- **Game Management**: PGN import/export, move history

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup
1. **Clone or download the project files**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create data directories:**
   ```bash
   mkdir -p data/models data/opening_books data/syzygy
   ```

4. **Download resources (optional):**
   - **Master games**: Download PGN files from Lichess or FICS
   - **Syzygy tablebases**: Download from official sources
   - **Pre-trained models**: If available

## Quick Start

### Basic Usage
```bash
python main.py
```

This starts the CLI interface where you can:
- Type `help` for available commands
- Enter moves in algebraic notation (e.g., `e2e4`, `Nf3`)
- Use commands like `analyze`, `eval`, `show`

### Example Session
```
Chess Bot > new
Starting new game!
Play as (w)hite, (b)lack, or (a)uto? [w]: w

8 r n b q k b n r
7 p p p p p p p p
6 . . . . . . . .
5 . . . . . . . .
4 . . . . . . . .
3 . . . . . . . .
2 P P P P P P P P
1 R N B Q K B N R
  a b c d e f g h

Turn: White

Chess Bot > e2e4
You played: e2e4

Bot is thinking...
Bot played: e7e5 (thought for 1.2s)

Chess Bot > analyze
Analyzing position...
Static Evaluation: +0.15
Best Move: Nf3 (found in 0.8s)
```

## Commands Reference

### Game Control
- `new` / `newgame` - Start a new game
- `quit` / `exit` / `q` - Exit the program
- `undo` - Undo last move
- `auto` - Let bot play both sides

### Information
- `show` / `board` - Display current position
- `moves` / `legal` - Show all legal moves
- `eval` / `evaluate` - Show position evaluation
- `analyze` - Deep position analysis

### Settings
- `depth <n>` - Set search depth (1-8)
- `time <s>` - Set thinking time in seconds

### Game Features
- `pgn` - Save game in PGN format
- `help` / `h` - Show help information

### Move Input
Enter moves in standard algebraic notation:
- Basic moves: `e2e4`, `g1f3`
- Captures: `exd5`, `Nxf7`
- Castling: `O-O`, `O-O-O`
- Promotions: `e8=Q`, `a1=N`

## Configuration

### Neural Network
The bot uses a CNN with the following architecture:
- Input: 8×8×14 tensor (board representation)
- 19 residual blocks with 256 filters
- Policy head: Move probability distribution
- Value head: Position evaluation (-1 to +1)

### Search Parameters
- **MCTS Simulations**: 800 (default)
- **Search Time**: Adaptive based on position complexity
- **Exploration Parameter**: 1.0 (C-PUCT)

### Training Data
For optimal performance, provide:
- Master games database (PGN format, 2000+ ELO)
- Minimum 100,000 games for supervised learning
- Self-play games for reinforcement learning

## File Structure

```
chess_ml_bot/
├── main.py                # Entry point
├── core/                  # Core engine components
│   ├── engine.py         # Main chess engine
│   ├── neural_net.py     # Neural network implementation
│   ├── search.py         # MCTS and minimax search
│   └── evaluation.py     # Position evaluation
├── training/              # Learning and training
│   ├── supervised.py     # Supervised learning
│   ├── reinforcement.py  # Self-play learning
│   ├── data_loader.py    # Data management
│   └── trainer.py        # Training coordination
├── features/              # Advanced features
│   ├── opening_book.py   # Opening book system
│   ├── tablebase.py      # Endgame tablebase support
│   ├── time_manager.py   # Time control management
│   └── opponent_model.py # Opponent adaptation
├── analysis/              # Analysis tools
│   ├── game_analyzer.py  # Game analysis
│   ├── position_db.py    # Position database
│   └── statistics.py     # Performance tracking
├── ui/                    # User interfaces
│   ├── gui.py            # Graphical interface
│   ├── cli.py            # Command-line interface
│   └── web_interface.py  # Web interface
└── utils/                 # Utilities
    ├── pgn_parser.py     # PGN file handling
    ├── fen_utils.py      # FEN utilities
    └── config.py         # Configuration
```

## Training the Bot

### Supervised Learning
1. **Prepare master games:**
   ```python
   from training.supervised import supervised_train
   supervised_train(model, "master_games.pgn", epochs=50)
   ```

2. **Monitor training progress:**
   - Loss curves for policy and value heads
   - Validation accuracy on held-out positions

### Reinforcement Learning
1. **Start self-play:**
   ```python
   from training.reinforcement import self_play
   games = self_play(num_games=1000)
   ```

2. **Update model:**
   - Train on self-play games
   - Compare with previous version
   - Keep best performing model

## Advanced Features

### Opening Book
- Automatically learns from master games
- Tracks move popularity and success rates
- Adapts to opponent preferences
- Supports transpositions

### Endgame Tablebases
- Perfect play with 7 or fewer pieces
- Instant mate detection
- Optimal conversion techniques
- Download Syzygy tablebases for full support

### Opponent Modeling
- Tracks opening preferences
- Identifies tactical weaknesses
- Adapts playing style
- Maintains opponent profiles

### Analysis Tools
- Centipawn loss calculation
- Mistake identification
- Critical position detection
- Opening preparation

## Performance Optimization

### Hardware Requirements
- **Minimum**: 4GB RAM, modern CPU
- **Recommended**: 8GB+ RAM, GPU with CUDA support
- **Training**: 16GB+ RAM, high-end GPU

### Speed Optimizations
- Use GPU acceleration for neural network
- Enable multi-threading for search
- Cache frequent positions
- Use opening book for fast starts

## Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'chess'"**
- Install dependencies: `pip install python-chess`

**"Neural network runs slowly"**
- Install PyTorch with CUDA support
- Reduce number of MCTS simulations
- Use smaller neural network

**"Opening book not loading"**
- Check file path in engine.py
- Ensure PGN file is valid format
- Create empty book if file missing

**"Bot makes weak moves"**
- Train neural network on more data
- Increase search time/simulations
- Check evaluation function

### Performance Tuning

**For faster play:**
- Reduce MCTS simulations (200-400)
- Decrease search depth for minimax
- Use lighter neural network

**For stronger play:**
- Increase MCTS simulations (1600+)
- Use larger neural network
- Add more training data
- Enable GPU acceleration

## Contributing

This is a learning implementation with room for improvements:

### Areas for Enhancement
- **Training**: Implement full AlphaZero training pipeline
- **Search**: Add advanced search techniques (LMR, null move)
- **Evaluation**: Expand evaluation features
- **UI**: Develop full GUI with analysis boards
- **Performance**: Optimize for tournament play

### Development Setup
```bash
# Install development dependencies
pip install pytest black flake8

# Run tests
python -m pytest

# Format code
black .

# Check style
flake8 .
```

## License

This project is for educational purposes. Please respect licensing for any datasets or external resources used.

## Resources

### Learning Materials
- [Chess Programming Wiki](https://www.chessprogramming.org/)
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)
- [python-chess Documentation](https://python-chess.readthedocs.io/)

### Data Sources
- [Lichess Database](https://database.lichess.org/)
- [FICS Games Database](http://www.ficsgames.org/)
- [Syzygy Endgame Tablebases](http://tablebase.sesse.net/)

### Engines for Testing
- [Stockfish](https://stockfishchess.org/)
- [Komodo](https://komodochess.com/)
- [Leela Chess Zero](https://lczero.org/)

## Support

For questions or issues:
1. Check this README and documentation
2. Review the code comments and examples
3. Test with simpler configurations first
4. Verify all dependencies are installed correctly

Remember: This is a learning implementation. Building a world-class chess engine requires significant computational resources and expertise!