import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess

class ResidualBlock(nn.Module):
    """Residual block for deep feature extraction in chess positions."""
    
    def __init__(self, channels=256):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class ChessNet(nn.Module):
    """
    AlphaZero-style neural network for chess.
    Input: 8x8x14 tensor representing board position
    Output: Policy head (move probabilities) and Value head (position evaluation)
    """
    
    def __init__(self, input_channels=14, num_res_blocks=19):
        super(ChessNet, self).__init__()
        
        # Input convolution
        self.conv_input = nn.Conv2d(input_channels, 256, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(256)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([ResidualBlock(256) for _ in range(num_res_blocks)])
        
        # Policy head for move prediction
        self.policy_conv = nn.Conv2d(256, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4096)  # Max possible moves
        
        # Value head for position evaluation
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        # Input convolution
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
            
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = F.log_softmax(self.policy_fc(policy), dim=1)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def predict(self, board_tensor):
        """Make prediction for a single board position."""
        self.eval()
        with torch.no_grad():
            if board_tensor.dim() == 3:
                board_tensor = board_tensor.unsqueeze(0)
            policy, value = self.forward(board_tensor)
            return policy.squeeze(), value.squeeze()

def encode_board(board):
    """
    Encode chess board to 8x8x14 tensor.
    Channels:
    0-5: White pieces (P, N, B, R, Q, K)
    6-11: Black pieces (P, N, B, R, Q, K)
    12: Castling rights
    13: En passant
    """
    tensor = np.zeros((14, 8, 8), dtype=np.float32)
    
    # Piece placement
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row = 7 - (square // 8)  # Convert to array coordinates
        col = square % 8
        
        # White pieces (channels 0-5)
        if piece.color == chess.WHITE:
            if piece.piece_type == chess.PAWN:
                tensor[0, row, col] = 1
            elif piece.piece_type == chess.KNIGHT:
                tensor[1, row, col] = 1
            elif piece.piece_type == chess.BISHOP:
                tensor[2, row, col] = 1
            elif piece.piece_type == chess.ROOK:
                tensor[3, row, col] = 1
            elif piece.piece_type == chess.QUEEN:
                tensor[4, row, col] = 1
            elif piece.piece_type == chess.KING:
                tensor[5, row, col] = 1
        
        # Black pieces (channels 6-11)
        else:
            if piece.piece_type == chess.PAWN:
                tensor[6, row, col] = 1
            elif piece.piece_type == chess.KNIGHT:
                tensor[7, row, col] = 1
            elif piece.piece_type == chess.BISHOP:
                tensor[8, row, col] = 1
            elif piece.piece_type == chess.ROOK:
                tensor[9, row, col] = 1
            elif piece.piece_type == chess.QUEEN:
                tensor[10, row, col] = 1
            elif piece.piece_type == chess.KING:
                tensor[11, row, col] = 1
    
    # Castling rights (channel 12)
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[12, 7, 6:8] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[12, 7, 0:3] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[12, 0, 6:8] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[12, 0, 0:3] = 1
    
    # En passant (channel 13)
    if board.ep_square is not None:
        ep_row = 7 - (board.ep_square // 8)
        ep_col = board.ep_square % 8
        tensor[13, ep_row, ep_col] = 1
    
    return torch.FloatTensor(tensor)

def move_to_index(move, board):
    """Convert a chess move to policy index."""
    # Simplified move encoding - in practice, this needs to handle
    # all possible move types including promotions
    from_square = move.from_square
    to_square = move.to_square
    
    # Simple encoding: from_square * 64 + to_square
    return from_square * 64 + to_square

def index_to_move(index, board):
    """Convert policy index back to chess move."""
    from_square = index // 64
    to_square = index % 64
    
    try:
        move = chess.Move(from_square, to_square)
        if move in board.legal_moves:
            return move
    except:
        pass
    return None

class ModelManager:
    """Manages loading, saving, and updating the neural network model."""
    
    def __init__(self, model_path="models/chess_net.pth"):
        self.model_path = model_path
        self.model = ChessNet()
        self.load_model()
        
    def load_model(self):
        """Load model weights from file."""
        try:
            self.model.load_state_dict(torch.load(self.model_path))
            print(f"Loaded model from {self.model_path}")
        except FileNotFoundError:
            print(f"No model found at {self.model_path}, using random weights")
            
    def save_model(self):
        """Save current model weights."""
        import os
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Saved model to {self.model_path}")
        
    def get_model(self):
        """Get the current model."""
        return self.model