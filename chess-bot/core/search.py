import chess
import torch
import math
import random
import time
from collections import defaultdict
from .neural_net import ChessNet, encode_board, move_to_index, index_to_move

class MCTSNode:
    """
    Node in the Monte Carlo Tree Search.
    Stores visit counts, values, and policy priors.
    """
    
    def __init__(self, board, parent=None, move=None, prior=0.0):
        self.board = board.copy()
        self.parent = parent
        self.move = move  # Move that led to this position
        self.prior = prior  # Prior probability from neural network
        
        self.children = {}  # child nodes
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        
    def is_leaf(self):
        """Check if this is a leaf node (not expanded)."""
        return not self.is_expanded
        
    def expand(self, policy_probs):
        """Expand node by adding children for all legal moves."""
        self.is_expanded = True
        
        for move in self.board.legal_moves:
            # Get prior probability for this move
            move_idx = move_to_index(move, self.board)
            prior = policy_probs.get(move_idx, 0.0001)  # Small default prior
            
            # Create child board
            child_board = self.board.copy()
            child_board.push(move)
            
            # Add child node
            self.children[move] = MCTSNode(child_board, parent=self, move=move, prior=prior)
    
    def select_child(self, c_puct=1.0):
        """Select child using PUCT algorithm (Predictor + UCT)."""
        best_score = -float('inf')
        best_child = None
        
        sqrt_parent_visits = math.sqrt(self.visit_count)
        
        for child in self.children.values():
            # Calculate UCB score
            if child.visit_count == 0:
                q_value = 0
            else:
                q_value = child.value_sum / child.visit_count
                
            u_value = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child
    
    def backup(self, value):
        """Backup value through the tree to root."""
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent:
            # Flip value for opponent
            self.parent.backup(-value)
    
    def get_visit_counts(self):
        """Get visit counts for all children."""
        visits = {}
        for move, child in self.children.items():
            visits[move] = child.visit_count
        return visits

class MCTS:
    """
    Monte Carlo Tree Search implementation for chess.
    Uses neural network for position evaluation and move priors.
    """
    def create_root_node(self, board):
        return MCTSNode(board)

    
    def __init__(self, neural_net: ChessNet, simulations=800, c_puct=1.0):
        self.neural_net = neural_net
        self.simulations = simulations
        self.c_puct = c_puct
        
    def search(self, board: chess.Board, time_limit=None):
        """
        Run MCTS search and return the best move.
        
        Args:
            board: Current chess position
            time_limit: Maximum time to search (seconds)
        
        Returns:
            Best move found by search
        """
        root = MCTSNode(board)
        
        # Time management
        start_time = time.time()
        simulations_run = 0
        
        # Run simulations
        for simulation in range(self.simulations):
            # Check time limit
            if time_limit and (time.time() - start_time) > time_limit:
                break
                
            self._simulate(root)
            simulations_run += 1
            
        print(f"MCTS: Ran {simulations_run} simulations in {time.time() - start_time:.2f}s")
        
        # Select best move based on visit counts
        return self._select_best_move(root)
    
    def _simulate(self, root):
        """Run a single MCTS simulation."""
        node = root
        path = [node]
        
        # Selection: traverse tree until leaf
        while not node.is_leaf() and not node.board.is_game_over():
            node = node.select_child(self.c_puct)
            path.append(node)
        
        # Evaluation and expansion
        if node.board.is_game_over():
            # Terminal position
            result = node.board.result()
            if result == "1-0":
                value = 1.0 if node.board.turn == chess.BLACK else -1.0
            elif result == "0-1":
                value = 1.0 if node.board.turn == chess.WHITE else -1.0
            else:
                value = 0.0  # Draw
        else:
            # Use neural network for evaluation
            board_tensor = encode_board(node.board).unsqueeze(0)
            
            with torch.no_grad():
                policy_logits, value = self.neural_net(board_tensor)
                value = float(value.item())
                
                # Convert policy logits to probabilities
                policy_probs = torch.softmax(policy_logits, dim=1).squeeze()
                
                # Create move->probability mapping
                move_probs = {}
                for move in node.board.legal_moves:
                    move_idx = move_to_index(move, node.board)
                    if move_idx < len(policy_probs):
                        move_probs[move_idx] = float(policy_probs[move_idx])
                
                # Expand node
                if not node.is_expanded:
                    node.expand(move_probs)
        
        # Backup values
        for node in reversed(path):
            node.backup(value)
            value = -value  # Flip for opponent
    
    def _select_best_move(self, root):
        """Select the best move based on visit counts."""
        if not root.children:
            # No children, return random legal move
            legal_moves = list(root.board.legal_moves)
            return random.choice(legal_moves) if legal_moves else None
        
        # Select move with highest visit count
        best_move = None
        best_visits = -1
        
        for move, child in root.children.items():
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_move = move
        
        # Debug output
        print(f"Best move: {best_move} (visits: {best_visits})")
        
        # Show top 3 moves
        moves_by_visits = sorted(root.children.items(), 
                               key=lambda x: x[1].visit_count, reverse=True)
        for i, (move, child) in enumerate(moves_by_visits[:3]):
            avg_value = child.value_sum / max(child.visit_count, 1)
            print(f"  {i+1}. {move}: {child.visit_count} visits, avg value: {avg_value:.3f}")
        
        return best_move

class MinimaxSearch:
    """
    Traditional minimax search with alpha-beta pruning.
    Used as fallback when neural network is not available.
    """
    
    def __init__(self, evaluator, max_depth=4):
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.nodes_searched = 0
        
    def search(self, board: chess.Board, depth=None):
        """Search for best move using minimax with alpha-beta pruning."""
        if depth is None:
            depth = self.max_depth
            
        self.nodes_searched = 0
        start_time = time.time()
        
        best_move, best_score = self._minimax(board, depth, -float('inf'), float('inf'), True)
        
        search_time = time.time() - start_time
        print(f"Minimax: depth {depth}, {self.nodes_searched} nodes, {search_time:.2f}s")
        print(f"Best move: {best_move} (score: {best_score:.2f})")
        
        return best_move
    
    def _minimax(self, board, depth, alpha, beta, maximizing):
        """Minimax algorithm with alpha-beta pruning."""
        self.nodes_searched += 1
        
        if depth == 0 or board.is_game_over():
            return None, self.evaluator.evaluate(board)
        
        best_move = None
        
        if maximizing:
            max_score = -float('inf')
            for move in board.legal_moves:
                board.push(move)
                _, score = self._minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                
                if score > max_score:
                    max_score = score
                    best_move = move
                    
                alpha = max(alpha, score)
                if beta <= alpha:
                    break  # Alpha-beta cutoff
                    
            return best_move, max_score
        else:
            min_score = float('inf')
            for move in board.legal_moves:
                board.push(move)
                _, score = self._minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                
                if score < min_score:
                    min_score = score
                    best_move = move
                    
                beta = min(beta, score)
                if beta <= alpha:
                    break  # Alpha-beta cutoff
                    
            return best_move, min_score

class SearchManager:
    """Manages different search algorithms and chooses the best one."""
    
    def __init__(self, neural_net, evaluator):
        self.mcts = MCTS(neural_net)
        self.minimax = MinimaxSearch(evaluator)
        self.use_neural_net = True
        
    def search(self, board, time_limit=None):
        """Search for best move using available algorithms."""
        if self.use_neural_net:
            try:
                return self.mcts.search(board, time_limit)
            except Exception as e:
                print(f"MCTS failed: {e}, falling back to minimax")
                self.use_neural_net = False
        
        # Fallback to minimax
        return self.minimax.search(board)
    
    def set_strength(self, strength_level):
        """Adjust search strength (0.0 to 1.0)."""
        if strength_level <= 0.3:
            self.mcts.simulations = 100
            self.minimax.max_depth = 2
        elif strength_level <= 0.6:
            self.mcts.simulations = 400
            self.minimax.max_depth = 3
        else:
            self.mcts.simulations = 800
            self.minimax.max_depth = 4