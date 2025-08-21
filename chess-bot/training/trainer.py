import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    """
    Trainer class for chess neural network.
    Handles both supervised and reinforcement learning training.
    """
    
    def __init__(self, model, device='cuda', learning_rate=1e-3):
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)
        
        # Loss functions
        self.policy_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.value_loss_fn = nn.MSELoss()
        
        # Training statistics
        self.training_history = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'learning_rate': []
        }
        
        # Setup logging
        self.writer = None
        self.setup_logging()
    
    def setup_logging(self):
        """Setup TensorBoard logging."""
        try:
            log_dir = f"logs/training_{int(time.time())}"
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
            print(f"Logging to {log_dir}")
        except Exception as e:
            print(f"Warning: Could not setup TensorBoard logging: {e}")
    
    def train_supervised(self, positions, policies, values, epochs=10, batch_size=32):
        """
        Train model using supervised learning.
        
        Args:
            positions: Position tensors [N, 14, 8, 8]
            policies: Policy tensors [N, 4096]
            values: Value tensors [N]
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        print(f"Starting supervised training: {epochs} epochs, batch size {batch_size}")
        
        # Move data to device
        positions = positions.to(self.device)
        policies = policies.to(self.device)
        values = values.to(self.device)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(positions, policies, values)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        
        for epoch in range(epochs):
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_total_loss = 0.0
            num_batches = 0
            
            start_time = time.time()
            
            for batch_idx, (batch_positions, batch_policies, batch_values) in enumerate(data_loader):
                # Forward pass
                pred_policies, pred_values = self.model(batch_positions)
                
                # Calculate losses
                policy_loss = self.policy_loss_fn(pred_policies, batch_policies)
                value_loss = self.value_loss_fn(pred_values.squeeze(), batch_values)
                total_loss = policy_loss + value_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Accumulate losses
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_total_loss += total_loss.item()
                num_batches += 1
                
                # Log batch progress
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(data_loader)}, "
                          f"Loss: {total_loss.item():.4f}")
            
            # Calculate average losses
            avg_policy_loss = epoch_policy_loss / num_batches
            avg_value_loss = epoch_value_loss / num_batches
            avg_total_loss = epoch_total_loss / num_batches
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.training_history['policy_loss'].append(avg_policy_loss)
            self.training_history['value_loss'].append(avg_value_loss)
            self.training_history['total_loss'].append(avg_total_loss)
            self.training_history['learning_rate'].append(current_lr)
            
            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar('Loss/Policy', avg_policy_loss, epoch)
                self.writer.add_scalar('Loss/Value', avg_value_loss, epoch)
                self.writer.add_scalar('Loss/Total', avg_total_loss, epoch)
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} complete: "
                  f"Policy Loss: {avg_policy_loss:.4f}, "
                  f"Value Loss: {avg_value_loss:.4f}, "
                  f"Total Loss: {avg_total_loss:.4f}, "
                  f"LR: {current_lr:.6f}, "
                  f"Time: {epoch_time:.2f}s")
            
        Trainer.save_model("data/models/chess_net.pth")
        print("Supervised training complete!")
    
    def train_reinforcement(self, self_play_data, epochs=10, batch_size=32):
        """
        Train model using reinforcement learning data from self-play.
        
        Args:
            self_play_data: List of (position, move_probs, value) tuples
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        if not self_play_data:
            print("No self-play data provided for training")
            return
        
        print(f"Training on {len(self_play_data)} self-play positions")
        
        # Convert self-play data to tensors
        positions = torch.stack([data[0] for data in self_play_data])
        policies = torch.stack([data[1] for data in self_play_data])
        values = torch.tensor([data[2] for data in self_play_data], dtype=torch.float32)
        
        # Train using supervised learning approach
        Trainer.save_model("data/models/chess_net_reinforced.pth")
        self.train_supervised(positions, policies, values, epochs, batch_size)
        
    
    def evaluate(self, test_positions, test_policies, test_values):
        """
        Evaluate model on test data.
        
        Args:
            test_positions: Test position tensors
            test_policies: Test policy tensors
            test_values: Test value tensors
            
        Returns:
            dict: Evaluation metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            test_positions = test_positions.to(self.device)
            test_policies = test_policies.to(self.device)
            test_values = test_values.to(self.device)
            
            pred_policies, pred_values = self.model(test_positions)
            
            policy_loss = self.policy_loss_fn(pred_policies, test_policies)
            value_loss = self.value_loss_fn(pred_values.squeeze(), test_values)
            total_loss = policy_loss + value_loss
            
            # Calculate accuracy metrics
            value_mae = torch.mean(torch.abs(pred_values.squeeze() - test_values))
            
            # Policy top-k accuracy
            policy_top1_acc = self._calculate_policy_accuracy(pred_policies, test_policies, k=1)
            policy_top3_acc = self._calculate_policy_accuracy(pred_policies, test_policies, k=3)
        
        metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item(),
            'value_mae': value_mae.item(),
            'policy_top1_accuracy': policy_top1_acc,
            'policy_top3_accuracy': policy_top3_acc
        }
        
        return metrics
    
    def _calculate_policy_accuracy(self, pred_policies, true_policies, k=1):
        """Calculate top-k accuracy for policy predictions."""
        # Get top-k predicted moves
        _, pred_top_k = torch.topk(pred_policies, k, dim=1)
        
        # Get true move (highest probability in true policy)
        _, true_move = torch.max(true_policies, dim=1)
        
        # Check if true move is in top-k predictions
        correct = 0
        total = len(true_move)
        
        for i in range(total):
            if true_move[i] in pred_top_k[i]:
                correct += 1
        
        return correct / total
    
    def save_model(self, filepath, include_optimizer=False):
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model
            include_optimizer: Whether to save optimizer state
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history
        }
        
        if include_optimizer:
            save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
            save_dict['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath, load_optimizer=False):
        """
        Load model from disk.
        
        Args:
            filepath: Path to load model from
            load_optimizer: Whether to load optimizer state
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if load_optimizer and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Model loaded from {filepath}")
    
    def get_training_summary(self):
        """Get summary of training progress."""
        if not self.training_history['total_loss']:
            return "No training history available"
        
        summary = f"""
Training Summary:
=================
Epochs completed: {len(self.training_history['total_loss'])}
Final losses:
  - Policy Loss: {self.training_history['policy_loss'][-1]:.4f}
  - Value Loss: {self.training_history['value_loss'][-1]:.4f}
  - Total Loss: {self.training_history['total_loss'][-1]:.4f}
Current learning rate: {self.training_history['learning_rate'][-1]:.6f}

Best losses:
  - Best Policy Loss: {min(self.training_history['policy_loss']):.4f}
  - Best Value Loss: {min(self.training_history['value_loss']):.4f}
  - Best Total Loss: {min(self.training_history['total_loss']):.4f}
"""
        return summary
    
    def plot_training_history(self, save_path=None):
        """Plot training history."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Policy loss
            axes[0, 0].plot(self.training_history['policy_loss'])
            axes[0, 0].set_title('Policy Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            
            # Value loss
            axes[0, 1].plot(self.training_history['value_loss'])
            axes[0, 1].set_title('Value Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            
            # Total loss
            axes[1, 0].plot(self.training_history['total_loss'])
            axes[1, 0].set_title('Total Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            
            # Learning rate
            axes[1, 1].plot(self.training_history['learning_rate'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('LR')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                print(f"Training plots saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib not available, cannot plot training history")
    
    def cleanup(self):
        """Cleanup resources."""
        if self.writer:
            self.writer.close()