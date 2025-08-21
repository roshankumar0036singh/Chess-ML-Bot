import json
import os

class Config:
    """
    Configuration manager for Chess ML Bot.
    Handles loading, saving, and accessing configuration settings.
    """
    
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self):
        """Load configuration from file or create defaults."""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration."""
        return {
            "engine": {
                "search_depth": 4,
                "mcts_simulations": 800,
                "time_per_move": 5.0,
                "use_opening_book": True,
                "use_tablebase": True,
                "neural_network_enabled": True
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 0.001,
                "epochs": 10,
                "device": "auto",  # auto, cpu, cuda
                "data_augmentation": True,
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
                "default_interface": "cli",  # cli, gui, web
                "board_theme": "default",
                "show_coordinates": True,
                "show_legal_moves": True,
                "animation_speed": 0.3
            },
            "analysis": {
                "engine_path": None,  # Path to external engine (e.g., Stockfish)
                "analysis_depth": 15,
                "analysis_time": 1.0,
                "save_analysis": True
            },
            "opening_book": {
                "max_book_moves": 15,
                "min_game_rating": 2000,
                "book_learning": True,
                "popularity_threshold": 0.01
            },
            "time_management": {
                "default_time_control": "600+5",  # 10 minutes + 5 second increment
                "time_control_type": "fischer",
                "emergency_time_threshold": 30.0,
                "complexity_factor": 1.0
            },
            "opponent_modeling": {
                "enabled": True,
                "adaptation_strength": 0.3,
                "min_games_for_model": 5,
                "model_decay_rate": 0.1
            },
            "logging": {
                "level": "INFO",  # DEBUG, INFO, WARNING, ERROR
                "log_games": True,
                "log_analysis": True,
                "log_training": True,
                "max_log_files": 10
            },
            "performance": {
                "parallel_search": True,
                "cache_size": 1000000,  # Number of positions to cache
                "memory_limit": "2GB",
                "gpu_memory_fraction": 0.5
            }
        }
    
    def get(self, key_path, default=None):
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Configuration key path (e.g., "engine.search_depth")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        try:
            keys = key_path.split('.')
            value = self.config
            
            for key in keys:
                value = value[key]
            
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path, value):
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Configuration key path
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the final key
        config[keys[-1]] = value
        self.save()
    
    def save(self):
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        self.config = self._create_default_config()
        self.save()
    
    def update_from_dict(self, updates):
        """
        Update configuration from dictionary.
        
        Args:
            updates: Dictionary of configuration updates
        """
        def deep_update(target, source):
            for key, value in source.items():
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    deep_update(target[key], value)
                else:
                    target[key] = value
        
        deep_update(self.config, updates)
        self.save()
    
    def get_engine_config(self):
        """Get engine-specific configuration."""
        return self.config.get("engine", {})
    
    def get_training_config(self):
        """Get training-specific configuration."""
        return self.config.get("training", {})
    
    def get_paths_config(self):
        """Get paths configuration."""
        return self.config.get("paths", {})
    
    def setup_directories(self):
        """Create necessary directories based on configuration."""
        paths = self.get_paths_config()
        
        for path_key, path_value in paths.items():
            if path_value:
                try:
                    os.makedirs(path_value, exist_ok=True)
                    print(f"Created directory: {path_value}")
                except Exception as e:
                    print(f"Error creating directory {path_value}: {e}")
    
    def validate_config(self):
        """Validate configuration settings."""
        issues = []
        
        # Validate engine settings
        search_depth = self.get("engine.search_depth")
        if not isinstance(search_depth, int) or search_depth < 1 or search_depth > 10:
            issues.append("engine.search_depth must be an integer between 1 and 10")
        
        mcts_sims = self.get("engine.mcts_simulations")
        if not isinstance(mcts_sims, int) or mcts_sims < 10 or mcts_sims > 10000:
            issues.append("engine.mcts_simulations must be an integer between 10 and 10000")
        
        # Validate training settings
        batch_size = self.get("training.batch_size")
        if not isinstance(batch_size, int) or batch_size < 1 or batch_size > 1024:
            issues.append("training.batch_size must be an integer between 1 and 1024")
        
        learning_rate = self.get("training.learning_rate")
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0 or learning_rate > 1:
            issues.append("training.learning_rate must be a number between 0 and 1")
        
        # Validate paths
        paths = self.get_paths_config()
        for path_key, path_value in paths.items():
            if path_value and not isinstance(path_value, str):
                issues.append(f"paths.{path_key} must be a string")
        
        # Validate device setting
        device = self.get("training.device")
        if device not in ["auto", "cpu", "cuda"]:
            issues.append("training.device must be 'auto', 'cpu', or 'cuda'")
        
        return issues
    
    def get_summary(self):
        """Get configuration summary."""
        return f"""
Chess ML Bot Configuration Summary:
==================================

Engine Settings:
- Search Depth: {self.get('engine.search_depth')}
- MCTS Simulations: {self.get('engine.mcts_simulations')}
- Time per Move: {self.get('engine.time_per_move')}s
- Opening Book: {'Enabled' if self.get('engine.use_opening_book') else 'Disabled'}
- Tablebase: {'Enabled' if self.get('engine.use_tablebase') else 'Disabled'}
- Neural Network: {'Enabled' if self.get('engine.neural_network_enabled') else 'Disabled'}

Training Settings:
- Batch Size: {self.get('training.batch_size')}
- Learning Rate: {self.get('training.learning_rate')}
- Device: {self.get('training.device')}
- Data Augmentation: {'Enabled' if self.get('training.data_augmentation') else 'Disabled'}

User Interface:
- Default Interface: {self.get('ui.default_interface')}
- Board Theme: {self.get('ui.board_theme')}
- Show Coordinates: {'Yes' if self.get('ui.show_coordinates') else 'No'}

Time Management:
- Default Time Control: {self.get('time_management.default_time_control')}
- Type: {self.get('time_management.time_control_type')}

Data Paths:
- Models: {self.get('paths.models_dir')}
- Opening Books: {self.get('paths.opening_book_dir')}
- Training Data: {self.get('paths.training_data_dir')}
"""

# Global configuration instance
_config_instance = None

def get_config(config_file="config.json"):
    """Get global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_file)
    return _config_instance

def reload_config():
    """Reload configuration from file."""
    global _config_instance
    if _config_instance:
        _config_instance.config = _config_instance._load_config()

# Convenience functions for common settings
def get_engine_depth():
    """Get engine search depth."""
    return get_config().get("engine.search_depth", 4)

def get_mcts_simulations():
    """Get MCTS simulation count."""
    return get_config().get("engine.mcts_simulations", 800)

def get_batch_size():
    """Get training batch size."""
    return get_config().get("training.batch_size", 32)

def get_learning_rate():
    """Get training learning rate."""
    return get_config().get("training.learning_rate", 0.001)

def get_models_dir():
    """Get models directory path."""
    return get_config().get("paths.models_dir", "data/models")

def get_training_device():
    """Get training device setting."""
    device = get_config().get("training.device", "auto")
    
    if device == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    return device

def is_opening_book_enabled():
    """Check if opening book is enabled."""
    return get_config().get("engine.use_opening_book", True)

def is_tablebase_enabled():
    """Check if tablebase is enabled."""
    return get_config().get("engine.use_tablebase", True)

def setup_logging():
    """Setup logging based on configuration."""
    import logging
    
    config = get_config()
    log_level = config.get("logging.level", "INFO")
    logs_dir = config.get("paths.logs_dir", "logs")
    
    # Create logs directory
    os.makedirs(logs_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(logs_dir, 'chess_bot.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('chess_bot')

# Command-line configuration utility
def config_cli():
    """Command-line interface for configuration."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python config.py <command> [args]")
        print("Commands:")
        print("  show - Show current configuration")
        print("  set <key> <value> - Set configuration value")
        print("  get <key> - Get configuration value")
        print("  reset - Reset to defaults")
        print("  validate - Validate configuration")
        return
    
    config = get_config()
    command = sys.argv[1]
    
    if command == "show":
        print(config.get_summary())
    
    elif command == "set" and len(sys.argv) >= 4:
        key = sys.argv[2]
        value = sys.argv[3]
        
        # Try to convert value to appropriate type
        try:
            if value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            elif '.' in value and value.replace('.', '').isdigit():
                value = float(value)
        except:
            pass
        
        config.set(key, value)
        print(f"Set {key} = {value}")
    
    elif command == "get" and len(sys.argv) >= 3:
        key = sys.argv[2]
        value = config.get(key)
        print(f"{key} = {value}")
    
    elif command == "reset":
        config.reset_to_defaults()
        print("Configuration reset to defaults")
    
    elif command == "validate":
        issues = config.validate_config()
        if issues:
            print("Configuration issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("Configuration is valid")
    
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    config_cli()