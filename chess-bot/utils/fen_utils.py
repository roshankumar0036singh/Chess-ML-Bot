import chess

def fen_to_board(fen):
    """
    Convert FEN string to chess.Board object.
    
    Args:
        fen: FEN string representing a chess position
        
    Returns:
        chess.Board: Board object, or None if invalid FEN
    """
    try:
        board = chess.Board(fen)
        return board
    except ValueError as e:
        print(f"Invalid FEN: {fen} - {e}")
        return None

def board_to_fen(board):
    """
    Convert chess.Board object to FEN string.
    
    Args:
        board: chess.Board object
        
    Returns:
        str: FEN string
    """
    return board.fen()

def normalize_fen(fen):
    """
    Normalize FEN string by removing move counters for position comparison.
    
    Args:
        fen: Full FEN string
        
    Returns:
        str: Position-only FEN (first 4 components)
    """
    fen_parts = fen.split(' ')
    if len(fen_parts) >= 4:
        return ' '.join(fen_parts[:4])
    return fen

def fen_to_position_hash(fen):
    """
    Create a hash from FEN for position caching.
    
    Args:
        fen: FEN string
        
    Returns:
        str: Position hash
    """
    import hashlib
    normalized_fen = normalize_fen(fen)
    return hashlib.md5(normalized_fen.encode()).hexdigest()

def parse_fen_components(fen):
    """
    Parse FEN string into its components.
    
    Args:
        fen: FEN string
        
    Returns:
        dict: Dictionary with FEN components
    """
    try:
        parts = fen.split(' ')
        
        if len(parts) < 6:
            raise ValueError("Incomplete FEN string")
        
        return {
            'piece_placement': parts[0],
            'active_color': parts[1],
            'castling_rights': parts[2],
            'en_passant': parts[3],
            'halfmove_clock': int(parts[4]),
            'fullmove_number': int(parts[5])
        }
    except (IndexError, ValueError) as e:
        print(f"Error parsing FEN: {fen} - {e}")
        return None

def validate_fen(fen):
    """
    Validate FEN string format and content.
    
    Args:
        fen: FEN string to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Try to create board from FEN
        board = chess.Board(fen)
        
        # Additional validations
        components = parse_fen_components(fen)
        if not components:
            return False, "Invalid FEN format"
        
        # Check piece placement
        piece_placement = components['piece_placement']
        ranks = piece_placement.split('/')
        
        if len(ranks) != 8:
            return False, "Piece placement must have 8 ranks"
        
        for rank in ranks:
            rank_length = 0
            for char in rank:
                if char.isdigit():
                    rank_length += int(char)
                elif char in 'pnbrqkPNBRQK':
                    rank_length += 1
                else:
                    return False, f"Invalid character in piece placement: {char}"
            
            if rank_length != 8:
                return False, f"Invalid rank length: {rank_length}"
        
        # Check active color
        if components['active_color'] not in ['w', 'b']:
            return False, "Active color must be 'w' or 'b'"
        
        # Check castling rights
        castling = components['castling_rights']
        if castling != '-':
            for char in castling:
                if char not in 'KQkq':
                    return False, f"Invalid castling right: {char}"
        
        # Check en passant
        en_passant = components['en_passant']
        if en_passant != '-':
            if len(en_passant) != 2:
                return False, "En passant square must be 2 characters"
            
            file_char, rank_char = en_passant
            if file_char not in 'abcdefgh' or rank_char not in '12345678':
                return False, "Invalid en passant square format"
        
        # Check move counters
        if components['halfmove_clock'] < 0:
            return False, "Halfmove clock cannot be negative"
        
        if components['fullmove_number'] < 1:
            return False, "Fullmove number must be at least 1"
        
        return True, "Valid FEN"
        
    except Exception as e:
        return False, str(e)

def starting_position_fen():
    """
    Get the FEN for the starting chess position.
    
    Returns:
        str: Starting position FEN
    """
    return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

def create_fen_from_position(piece_placement, active_color='w', 
                           castling_rights='KQkq', en_passant='-',
                           halfmove_clock=0, fullmove_number=1):
    """
    Create FEN string from individual components.
    
    Args:
        piece_placement: Piece placement string
        active_color: Active color ('w' or 'b')
        castling_rights: Castling rights string
        en_passant: En passant square
        halfmove_clock: Halfmove clock
        fullmove_number: Fullmove number
        
    Returns:
        str: Complete FEN string
    """
    return f"{piece_placement} {active_color} {castling_rights} {en_passant} {halfmove_clock} {fullmove_number}"

def mirror_fen_horizontal(fen):
    """
    Mirror FEN position horizontally (flip files).
    
    Args:
        fen: Original FEN string
        
    Returns:
        str: Horizontally mirrored FEN
    """
    try:
        board = chess.Board(fen)
        
        # Create mirrored position
        mirrored_board = chess.Board(None)  # Empty board
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Mirror the file (column)
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                mirrored_file = 7 - file
                mirrored_square = chess.square(mirrored_file, rank)
                
                mirrored_board.set_piece_at(mirrored_square, piece)
        
        # Update castling rights for mirrored position
        castling_rights = ""
        if board.has_kingside_castling_rights(chess.WHITE):
            castling_rights += "Q"  # Kingside becomes queenside
        if board.has_queenside_castling_rights(chess.WHITE):
            castling_rights += "K"  # Queenside becomes kingside
        if board.has_kingside_castling_rights(chess.BLACK):
            castling_rights += "q"
        if board.has_queenside_castling_rights(chess.BLACK):
            castling_rights += "k"
        
        if not castling_rights:
            castling_rights = "-"
        
        # Update en passant square
        en_passant = "-"
        if board.ep_square is not None:
            ep_file = chess.square_file(board.ep_square)
            ep_rank = chess.square_rank(board.ep_square)
            mirrored_ep_file = 7 - ep_file
            mirrored_ep_square = chess.square(mirrored_ep_file, ep_rank)
            en_passant = chess.square_name(mirrored_ep_square)
        
        # Create mirrored FEN
        components = parse_fen_components(fen)
        mirrored_fen = create_fen_from_position(
            mirrored_board.board_fen(),
            components['active_color'],
            castling_rights,
            en_passant,
            components['halfmove_clock'],
            components['fullmove_number']
        )
        
        return mirrored_fen
        
    except Exception as e:
        print(f"Error mirroring FEN: {e}")
        return fen

def get_position_after_moves(starting_fen, moves):
    """
    Get FEN position after applying a sequence of moves.
    
    Args:
        starting_fen: Starting position FEN
        moves: List of moves in algebraic notation
        
    Returns:
        str: Resulting FEN, or None if moves are invalid
    """
    try:
        board = chess.Board(starting_fen)
        
        for move_str in moves:
            # Try to parse move
            try:
                move = board.parse_san(move_str)
            except ValueError:
                # Try UCI format
                try:
                    move = chess.Move.from_uci(move_str)
                except ValueError:
                    print(f"Invalid move: {move_str}")
                    return None
            
            if move not in board.legal_moves:
                print(f"Illegal move: {move_str}")
                return None
            
            board.push(move)
        
        return board.fen()
        
    except Exception as e:
        print(f"Error applying moves: {e}")
        return None

def compare_positions(fen1, fen2, ignore_move_counters=True):
    """
    Compare two FEN positions.
    
    Args:
        fen1: First FEN string
        fen2: Second FEN string
        ignore_move_counters: Whether to ignore halfmove/fullmove counters
        
    Returns:
        bool: True if positions are identical
    """
    if ignore_move_counters:
        fen1_normalized = normalize_fen(fen1)
        fen2_normalized = normalize_fen(fen2)
        return fen1_normalized == fen2_normalized
    else:
        return fen1 == fen2

def extract_material_from_fen(fen):
    """
    Extract material count from FEN.
    
    Args:
        fen: FEN string
        
    Returns:
        dict: Material count for each piece type
    """
    try:
        board = chess.Board(fen)
        
        material = {
            'white': {'pawns': 0, 'knights': 0, 'bishops': 0, 'rooks': 0, 'queens': 0, 'king': 0},
            'black': {'pawns': 0, 'knights': 0, 'bishops': 0, 'rooks': 0, 'queens': 0, 'king': 0}
        }
        
        piece_type_names = {
            chess.PAWN: 'pawns',
            chess.KNIGHT: 'knights',
            chess.BISHOP: 'bishops',
            chess.ROOK: 'rooks',
            chess.QUEEN: 'queens',
            chess.KING: 'king'
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                color = 'white' if piece.color == chess.WHITE else 'black'
                piece_name = piece_type_names[piece.piece_type]
                material[color][piece_name] += 1
        
        return material
        
    except Exception as e:
        print(f"Error extracting material from FEN: {e}")
        return None

def is_endgame_position(fen):
    """
    Determine if position is in endgame phase based on material.
    
    Args:
        fen: FEN string
        
    Returns:
        bool: True if position is likely endgame
    """
    try:
        board = chess.Board(fen)
        piece_count = len(board.piece_map())
        
        # Simple endgame detection
        return piece_count <= 12
        
    except Exception as e:
        print(f"Error checking endgame status: {e}")
        return False

def get_fen_statistics(fen_list):
    """
    Get statistics about a list of FEN positions.
    
    Args:
        fen_list: List of FEN strings
        
    Returns:
        dict: Statistics about the positions
    """
    stats = {
        'total_positions': len(fen_list),
        'valid_positions': 0,
        'invalid_positions': 0,
        'unique_positions': 0,
        'endgame_positions': 0,
        'white_to_move': 0,
        'black_to_move': 0
    }
    
    seen_positions = set()
    
    for fen in fen_list:
        is_valid, _ = validate_fen(fen)
        
        if is_valid:
            stats['valid_positions'] += 1
            
            # Count unique positions
            normalized = normalize_fen(fen)
            if normalized not in seen_positions:
                seen_positions.add(normalized)
                stats['unique_positions'] += 1
            
            # Check if endgame
            if is_endgame_position(fen):
                stats['endgame_positions'] += 1
            
            # Check active color
            components = parse_fen_components(fen)
            if components:
                if components['active_color'] == 'w':
                    stats['white_to_move'] += 1
                else:
                    stats['black_to_move'] += 1
        else:
            stats['invalid_positions'] += 1
    
    return stats