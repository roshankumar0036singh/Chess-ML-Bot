import chess.pgn
import os

def parse_pgn_games(pgn_file):
    """
    Parse games from a PGN file.
    
    Args:
        pgn_file: Path to PGN file
        
    Returns:
        list: List of chess.pgn.Game objects
    """
    games = []
    
    if not os.path.exists(pgn_file):
        print(f"PGN file not found: {pgn_file}")
        return games
    
    try:
        with open(pgn_file, 'r', encoding='utf-8') as f:
            game_count = 0
            
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                
                games.append(game)
                game_count += 1
                
                # Progress indicator for large files
                if game_count % 1000 == 0:
                    print(f"Loaded {game_count} games...")
        
        print(f"Successfully loaded {len(games)} games from {pgn_file}")
        
    except Exception as e:
        print(f"Error reading PGN file {pgn_file}: {e}")
    
    return games

def parse_pgn_stream(pgn_file, max_games=None):
    """
    Parse games from a PGN file as a generator (memory efficient).
    
    Args:
        pgn_file: Path to PGN file
        max_games: Maximum number of games to parse
        
    Yields:
        chess.pgn.Game: Individual game objects
    """
    if not os.path.exists(pgn_file):
        print(f"PGN file not found: {pgn_file}")
        return
    
    try:
        with open(pgn_file, 'r', encoding='utf-8') as f:
            game_count = 0
            
            while True:
                if max_games and game_count >= max_games:
                    break
                
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                
                yield game
                game_count += 1
                
                if game_count % 1000 == 0:
                    print(f"Processed {game_count} games...")
        
        print(f"Finished processing {game_count} games from {pgn_file}")
        
    except Exception as e:
        print(f"Error reading PGN file {pgn_file}: {e}")

def filter_games_by_rating(games, min_rating=2000):
    """
    Filter games by minimum player rating.
    
    Args:
        games: List of chess.pgn.Game objects
        min_rating: Minimum rating threshold
        
    Returns:
        list: Filtered games
    """
    filtered_games = []
    
    for game in games:
        try:
            white_elo = int(game.headers.get("WhiteElo", "0"))
            black_elo = int(game.headers.get("BlackElo", "0"))
            
            if white_elo >= min_rating and black_elo >= min_rating:
                filtered_games.append(game)
                
        except (ValueError, TypeError):
            # Skip games with invalid ratings
            continue
    
    print(f"Filtered {len(filtered_games)} games with rating >= {min_rating}")
    return filtered_games

def filter_games_by_time_control(games, time_controls=None):
    """
    Filter games by time control.
    
    Args:
        games: List of chess.pgn.Game objects
        time_controls: List of accepted time controls (e.g., ["300+3", "600+0"])
        
    Returns:
        list: Filtered games
    """
    if time_controls is None:
        return games
    
    filtered_games = []
    
    for game in games:
        time_control = game.headers.get("TimeControl", "")
        
        if time_control in time_controls:
            filtered_games.append(game)
    
    print(f"Filtered {len(filtered_games)} games with specified time controls")
    return filtered_games

def extract_opening_moves(games, max_moves=15):
    """
    Extract opening moves from games.
    
    Args:
        games: List of chess.pgn.Game objects
        max_moves: Maximum number of moves to extract
        
    Returns:
        dict: Dictionary mapping positions to move counts
    """
    opening_positions = {}
    
    for game in games:
        board = game.board()
        move_count = 0
        
        for move in game.mainline_moves():
            if move_count >= max_moves:
                break
            
            fen = board.fen().split(' ')[0]  # Position without move counters
            
            if fen not in opening_positions:
                opening_positions[fen] = {}
            
            move_str = str(move)
            if move_str not in opening_positions[fen]:
                opening_positions[fen][move_str] = 0
            
            opening_positions[fen][move_str] += 1
            
            board.push(move)
            move_count += 1
    
    return opening_positions

def save_games_to_pgn(games, output_file):
    """
    Save games to a PGN file.
    
    Args:
        games: List of chess.pgn.Game objects
        output_file: Output PGN file path
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for game in games:
                print(game, file=f)
                print(file=f)  # Empty line between games
        
        print(f"Saved {len(games)} games to {output_file}")
        
    except Exception as e:
        print(f"Error saving PGN file {output_file}: {e}")

def merge_pgn_files(input_files, output_file):
    """
    Merge multiple PGN files into one.
    
    Args:
        input_files: List of input PGN file paths
        output_file: Output PGN file path
    """
    all_games = []
    
    for input_file in input_files:
        if os.path.exists(input_file):
            games = parse_pgn_games(input_file)
            all_games.extend(games)
        else:
            print(f"Warning: File not found: {input_file}")
    
    save_games_to_pgn(all_games, output_file)
    print(f"Merged {len(all_games)} games from {len(input_files)} files")

def analyze_pgn_database(pgn_file):
    """
    Analyze a PGN database and provide statistics.
    
    Args:
        pgn_file: Path to PGN file
        
    Returns:
        dict: Database statistics
    """
    stats = {
        'total_games': 0,
        'unique_players': set(),
        'time_controls': {},
        'results': {'1-0': 0, '0-1': 0, '1/2-1/2': 0},
        'rating_distribution': {'2000-2199': 0, '2200-2399': 0, '2400+': 0},
        'events': {},
        'date_range': {'earliest': None, 'latest': None}
    }
    
    for game in parse_pgn_stream(pgn_file):
        stats['total_games'] += 1
        
        # Players
        white = game.headers.get("White", "Unknown")
        black = game.headers.get("Black", "Unknown")
        stats['unique_players'].add(white)
        stats['unique_players'].add(black)
        
        # Time controls
        time_control = game.headers.get("TimeControl", "Unknown")
        stats['time_controls'][time_control] = stats['time_controls'].get(time_control, 0) + 1
        
        # Results
        result = game.headers.get("Result", "*")
        if result in stats['results']:
            stats['results'][result] += 1
        
        # Rating distribution
        try:
            white_elo = int(game.headers.get("WhiteElo", "0"))
            black_elo = int(game.headers.get("BlackElo", "0"))
            avg_elo = (white_elo + black_elo) / 2
            
            if avg_elo >= 2400:
                stats['rating_distribution']['2400+'] += 1
            elif avg_elo >= 2200:
                stats['rating_distribution']['2200-2399'] += 1
            elif avg_elo >= 2000:
                stats['rating_distribution']['2000-2199'] += 1
        except (ValueError, TypeError):
            pass
        
        # Events
        event = game.headers.get("Event", "Unknown")
        stats['events'][event] = stats['events'].get(event, 0) + 1
        
        # Date range
        date = game.headers.get("Date", "")
        if date and date != "????.??.??":
            if stats['date_range']['earliest'] is None or date < stats['date_range']['earliest']:
                stats['date_range']['earliest'] = date
            if stats['date_range']['latest'] is None or date > stats['date_range']['latest']:
                stats['date_range']['latest'] = date
    
    # Convert set to count
    stats['unique_players'] = len(stats['unique_players'])
    
    return stats

def create_training_pgn(input_file, output_file, filters=None):
    """
    Create a training PGN file with filtering.
    
    Args:
        input_file: Input PGN file
        output_file: Output PGN file
        filters: Dictionary of filters to apply
    """
    if filters is None:
        filters = {
            'min_rating': 2000,
            'max_games': 10000,
            'time_controls': None,
            'exclude_bullet': True
        }
    
    training_games = []
    game_count = 0
    
    for game in parse_pgn_stream(input_file):
        # Apply filters
        if filters.get('min_rating'):
            try:
                white_elo = int(game.headers.get("WhiteElo", "0"))
                black_elo = int(game.headers.get("BlackElo", "0"))
                
                if white_elo < filters['min_rating'] or black_elo < filters['min_rating']:
                    continue
            except (ValueError, TypeError):
                continue
        
        if filters.get('exclude_bullet'):
            time_control = game.headers.get("TimeControl", "")
            if time_control.startswith("60+") or time_control.startswith("30+"):
                continue
        
        if filters.get('time_controls'):
            time_control = game.headers.get("TimeControl", "")
            if time_control not in filters['time_controls']:
                continue
        
        training_games.append(game)
        game_count += 1
        
        if filters.get('max_games') and game_count >= filters['max_games']:
            break
    
    save_games_to_pgn(training_games, output_file)
    print(f"Created training database with {len(training_games)} games")

def validate_pgn_file(pgn_file):
    """
    Validate a PGN file for common issues.
    
    Args:
        pgn_file: Path to PGN file
        
    Returns:
        dict: Validation results
    """
    validation = {
        'valid_games': 0,
        'invalid_games': 0,
        'errors': [],
        'warnings': []
    }
    
    for i, game in enumerate(parse_pgn_stream(pgn_file)):
        try:
            # Try to replay the game
            board = game.board()
            move_count = 0
            
            for move in game.mainline_moves():
                if move not in board.legal_moves:
                    validation['errors'].append(f"Game {i+1}: Illegal move {move}")
                    break
                board.push(move)
                move_count += 1
            
            # Check for minimum game length
            if move_count < 10:
                validation['warnings'].append(f"Game {i+1}: Very short game ({move_count} moves)")
            
            validation['valid_games'] += 1
            
        except Exception as e:
            validation['invalid_games'] += 1
            validation['errors'].append(f"Game {i+1}: {str(e)}")
    
    return validation

# Utility functions for common PGN operations
def get_game_by_players(pgn_file, white_player, black_player):
    """Find games between specific players."""
    matching_games = []
    
    for game in parse_pgn_stream(pgn_file):
        white = game.headers.get("White", "")
        black = game.headers.get("Black", "")
        
        if white == white_player and black == black_player:
            matching_games.append(game)
    
    return matching_games

def extract_player_games(pgn_file, player_name, max_games=100):
    """Extract games played by a specific player."""
    player_games = []
    
    for game in parse_pgn_stream(pgn_file, max_games * 2):  # Search more to find enough
        white = game.headers.get("White", "")
        black = game.headers.get("Black", "")
        
        if player_name in [white, black]:
            player_games.append(game)
            
            if len(player_games) >= max_games:
                break
    
    return player_games