# Chess ML Bot 🤖♟️

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/roshankumar0036singh/chess-ml-bot?style=social)](https://github.com/roshankumar0036singh/chess-ml-bot)

A sophisticated AI chess engine powered by deep learning and Monte Carlo Tree Search (MCTS). Features a responsive GUI, opening books, endgame tablebases, and AlphaZero-style self-play training.

![Chess Bot Demo](assets/demo.gif)

## 🌟 Features

### 🧠 **AI Engine**
- **Deep Neural Network**: Custom PyTorch CNN with residual blocks for position evaluation
- **Monte Carlo Tree Search**: Intelligent move selection with 800+ simulations per move
- **Opening Book**: Million+ position database from master games
- **Endgame Tablebases**: Perfect play using Syzygy tablebases
- **Self-Play Learning**: Continuous improvement through reinforcement learning

### 🎮 **Interactive GUI**
- **Smooth Gameplay**: Pygame-based responsive interface
- **Visual Feedback**: Move highlighting, legal moves, and thinking animations
- **Real-time Stats**: Live move history, evaluation scores, and game analysis
- **Non-blocking UI**: Threaded bot calculations keep interface responsive

### 🚀 **Training Pipeline**
- **Supervised Learning**: Train on PGN databases of master games
- **Reinforcement Learning**: Generate training data through self-play
- **Model Checkpointing**: Automatic saving and version management
- **Performance Monitoring**: TensorBoard integration with loss tracking

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Estimated Elo** | 1800-2200 |
| **Policy Loss** | 6.5 → 2.8 (after training) |
| **Value Loss** | 1.0 → 0.86 (after training) |
| **Search Speed** | 800 simulations in 2-5s |
| **Opening Positions** | 500,000+ |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/chess-ml-bot.git
cd chess-ml-bot

# Create virtual environment
python -m venv chess_env
source chess_env/bin/activate  # Windows: chess_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install CUDA PyTorch (for GPU training)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Play Against the Bot

```bash
python main.py --interface gui
```

### Train Your Own Model

```bash
# Supervised training
python train_model.py --mode supervised --epochs 30 --batch-size 64

# Self-play training
python train_model.py --mode self_play --games 100
```

## 🎮 Usage

### GUI Controls
- **Click**: Select and move pieces
- **N**: New game
- **U**: Undo move
- **F**: Flip board
- **A**: Analysis mode

### Command Line
```bash
# CLI gameplay
python main.py --interface cli

# Analysis mode
python main.py --analyze position.fen

# Tournament mode
python tournament.py --games 50
```

## 🏗️ Architecture

```
📦 chess-ml-bot/
├── 🧠 core/                 # AI Engine
│   ├── engine.py           # Main chess engine
│   ├── neural_net.py       # PyTorch neural network
│   ├── search.py           # MCTS implementation
│   └── evaluation.py       # Position evaluation
├── 🎮 ui/                   # User Interfaces
│   ├── gui.py              # Pygame GUI
│   └── cli.py              # Command line
├── 🚀 training/             # ML Training
│   ├── trainer.py          # Training pipeline
│   ├── reinforcement.py    # Self-play learning
│   └── data_loader.py      # Data processing
├── ⚡ features/             # Advanced Features
│   ├── opening_book.py     # Opening database
│   ├── tablebase.py        # Endgame tablebases
│   └── time_manager.py     # Time allocation
└── 📊 data/                 # Data Storage
    ├── models/             # Trained models
    ├── opening_books/      # PGN databases
    └── training_data/      # Training datasets
```

## 🧠 Neural Network

The model uses a ResNet-inspired architecture:

```python
Input: 14×8×8 board representation
├── Convolutional layers (3×3 kernels)
├── 12× Residual blocks (256 filters each)
├── Batch normalization + ReLU activation
└── Dual heads:
    ├── Policy head → 4096 possible moves
    └── Value head → Position evaluation (-1 to +1)
```

## 📈 Training Results

![Training Loss](assets/training_loss.png)

### Loss Progression (10 epochs)
- **Policy Loss**: 6.53 → 2.80 (-57%)
- **Value Loss**: 1.02 → 0.86 (-16%)
- **Total Loss**: 7.54 → 3.66 (-51%)

## 🎯 Getting Started with Training

### 1. Prepare Data
```bash
# Download master games (example: Lichess database)
wget https://database.lichess.org/standard/lichess_db_standard_rated_2023-01.pgn.bz2
bunzip2 lichess_db_standard_rated_2023-01.pgn.bz2
mv lichess_db_standard_rated_2023-01.pgn data/opening_books/master_games.pgn
```

### 2. Configure Training
Edit `config.json`:
```json
{
    "model": {
        "layers": 12,
        "channels": 256,
        "learning_rate": 0.001
    },
    "training": {
        "epochs": 30,
        "batch_size": 64,
        "device": "cuda"
    }
}
```

### 3. Start Training
```bash
python train_model.py --config config.json
```

## 🔧 Advanced Usage

### Custom Network Architecture
```python
from core.neural_net import ChessNet

# Create custom model
model = ChessNet(
    input_channels=14,
    residual_blocks=20,
    filters=512
)
```

### Engine Integration
```python
from core.engine import ChessEngine

# Initialize engine
engine = ChessEngine()
best_move = engine.get_best_move()
```

### Self-Play Training
```python
from training.reinforcement import SelfPlayLearning

# Start self-play
trainer = SelfPlayLearning(model)
trainer.run_self_play(num_games=1000)
```

## 🎪 Demo & Examples

### Example Game
```
1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7
Bot evaluation: +0.2 (slight advantage to White)
Best move: d3 (35% confidence)
```

### Analysis Mode
```bash
python main.py --analyze "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
# Outputs detailed position analysis and best moves
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature-name`
3. **Make** your changes and add tests
4. **Run** tests: `python -m pytest tests/`
5. **Submit** a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run code formatting
black .
flake8 .

# Run type checking
mypy core/ training/
```

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| GPU not detected | Install CUDA-enabled PyTorch |
| Missing opening book | Download PGN files to `data/opening_books/` |
| GUI freezing | Enable threading in config |
| Training slow | Use GPU and increase batch size |

### Performance Tips
- Use mixed precision training: `--mixed-precision`
- Increase batch size for GPU: `--batch-size 128`
- Monitor with TensorBoard: `tensorboard --logdir=logs/`

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DeepMind AlphaZero** for the self-play methodology
- **Stockfish** for benchmarking and inspiration  
- **python-chess** library for chess logic
- **PyTorch** team for the ML framework
- **Lichess** for open chess databases

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/your-username/chess-ml-bot/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/your-username/chess-ml-bot/discussions)
- 📧 **Email**: your-email@example.com
- 🐦 **Twitter**: [@your-twitter](https://twitter.com/your-twitter)

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=your-username/chess-ml-bot&type=Date)](https://star-history.com/#your-username/chess-ml-bot&Date)

---

<div align="center">

**🔥 Ready to play chess against AI?**

[**Play Now**](https://github.com/your-username/chess-ml-bot) • [**Documentation**](https://your-username.github.io/chess-ml-bot) • [**Demo**](https://your-demo-link.com)

</div>
