import sqlite3
import json
import os
from datetime import datetime
import chess

class PositionDB:
    """
    Database for storing and retrieving chess positions and their evaluations.
    Provides caching and analysis history functionality.
    """
    
    def __init__(self, db_path="data/positions.db"):
        self.db_path = db_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Positions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fen TEXT UNIQUE NOT NULL,
                    evaluation REAL,
                    depth INTEGER,
                    best_move TEXT,
                    analysis_data TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    engine_name TEXT,
                    nodes_searched INTEGER
                )
            ''')
            
            # Games table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS games (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pgn TEXT NOT NULL,
                    white_player TEXT,
                    black_player TEXT,
                    result TEXT,
                    date_played DATE,
                    event_name TEXT,
                    analysis_data TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Opening positions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS openings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fen TEXT UNIQUE NOT NULL,
                    opening_name TEXT,
                    eco_code TEXT,
                    moves_from_start INTEGER,
                    popularity_score REAL DEFAULT 0.0,
                    success_rate REAL DEFAULT 0.5
                )
            ''')
            
            # Analysis cache table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_hash TEXT UNIQUE NOT NULL,
                    analysis_type TEXT NOT NULL,
                    result_data TEXT NOT NULL,
                    computation_time REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_positions_fen ON positions(fen)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_games_players ON games(white_player, black_player)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_openings_fen ON openings(fen)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_hash ON analysis_cache(position_hash)')
            
            conn.commit()
    
    def store_position(self, fen, evaluation, depth=0, best_move=None, 
                      analysis_data=None, engine_name="internal", nodes_searched=0):
        """
        Store a position evaluation in the database.
        
        Args:
            fen: FEN string of the position
            evaluation: Numerical evaluation
            depth: Search depth used
            best_move: Best move in the position
            analysis_data: Additional analysis data (JSON)
            engine_name: Name of engine used for analysis
            nodes_searched: Number of nodes searched
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            analysis_json = json.dumps(analysis_data) if analysis_data else None
            
            cursor.execute('''
                INSERT OR REPLACE INTO positions 
                (fen, evaluation, depth, best_move, analysis_data, engine_name, nodes_searched)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (fen, evaluation, depth, best_move, analysis_json, engine_name, nodes_searched))
            
            conn.commit()
    
    def get_position(self, fen):
        """
        Retrieve position data from database.
        
        Args:
            fen: FEN string of the position
            
        Returns:
            dict: Position data or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM positions WHERE fen = ?', (fen,))
            row = cursor.fetchone()
            
            if row:
                return {
                    'id': row[0],
                    'fen': row[1],
                    'evaluation': row[2],
                    'depth': row[3],
                    'best_move': row[4],
                    'analysis_data': json.loads(row[5]) if row[5] else None,
                    'timestamp': row[6],
                    'engine_name': row[7],
                    'nodes_searched': row[8]
                }
            return None
    
    def store_game(self, pgn, white_player, black_player, result, 
                   date_played=None, event_name=None, analysis_data=None):
        """
        Store a game in the database.
        
        Args:
            pgn: PGN string of the game
            white_player: White player name
            black_player: Black player name
            result: Game result
            date_played: Date the game was played
            event_name: Tournament/event name
            analysis_data: Analysis results (JSON)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            analysis_json = json.dumps(analysis_data) if analysis_data else None
            
            cursor.execute('''
                INSERT INTO games 
                (pgn, white_player, black_player, result, date_played, event_name, analysis_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (pgn, white_player, black_player, result, date_played, event_name, analysis_json))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_games_by_player(self, player_name, limit=50):
        """
        Get games by a specific player.
        
        Args:
            player_name: Player name to search for
            limit: Maximum number of games to return
            
        Returns:
            list: List of game records
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM games 
                WHERE white_player = ? OR black_player = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (player_name, player_name, limit))
            
            return cursor.fetchall()
    
    def store_opening(self, fen, opening_name, eco_code=None, 
                     moves_from_start=0, popularity_score=0.0, success_rate=0.5):
        """
        Store opening position data.
        
        Args:
            fen: FEN string of the opening position
            opening_name: Name of the opening
            eco_code: ECO classification code
            moves_from_start: Number of moves from starting position
            popularity_score: Popularity score (0.0 to 1.0)
            success_rate: Success rate for this opening
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO openings 
                (fen, opening_name, eco_code, moves_from_start, popularity_score, success_rate)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (fen, opening_name, eco_code, moves_from_start, popularity_score, success_rate))
            
            conn.commit()
    
    def get_opening(self, fen):
        """
        Get opening information for a position.
        
        Args:
            fen: FEN string of the position
            
        Returns:
            dict: Opening data or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM openings WHERE fen = ?', (fen,))
            row = cursor.fetchone()
            
            if row:
                return {
                    'id': row[0],
                    'fen': row[1],
                    'opening_name': row[2],
                    'eco_code': row[3],
                    'moves_from_start': row[4],
                    'popularity_score': row[5],
                    'success_rate': row[6]
                }
            return None
    
    def cache_analysis(self, position_hash, analysis_type, result_data, computation_time=0.0):
        """
        Cache analysis results for faster retrieval.
        
        Args:
            position_hash: Hash of the position
            analysis_type: Type of analysis performed
            result_data: Analysis results
            computation_time: Time taken for analysis
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            result_json = json.dumps(result_data)
            
            cursor.execute('''
                INSERT OR REPLACE INTO analysis_cache 
                (position_hash, analysis_type, result_data, computation_time)
                VALUES (?, ?, ?, ?)
            ''', (position_hash, analysis_type, result_json, computation_time))
            
            conn.commit()
    
    def get_cached_analysis(self, position_hash, analysis_type):
        """
        Retrieve cached analysis results.
        
        Args:
            position_hash: Hash of the position
            analysis_type: Type of analysis
            
        Returns:
            dict: Cached analysis data or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT result_data, computation_time, timestamp 
                FROM analysis_cache 
                WHERE position_hash = ? AND analysis_type = ?
            ''', (position_hash, analysis_type))
            
            row = cursor.fetchone()
            
            if row:
                return {
                    'result_data': json.loads(row[0]),
                    'computation_time': row[1],
                    'timestamp': row[2]
                }
            return None
    
    def search_similar_positions(self, fen, similarity_threshold=0.8):
        """
        Search for similar positions in the database.
        
        Args:
            fen: FEN string of the position to match
            similarity_threshold: Minimum similarity score
            
        Returns:
            list: List of similar positions
        """
        # This is a simplified implementation
        # A more sophisticated version would use position hashing
        # and similarity algorithms
        
        board = chess.Board(fen)
        piece_count = len(board.piece_map())
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Find positions with similar piece counts
            cursor.execute('''
                SELECT fen, evaluation, best_move 
                FROM positions 
                WHERE LENGTH(fen) BETWEEN ? AND ?
                LIMIT 20
            ''', (len(fen) - 10, len(fen) + 10))
            
            results = []
            for row in cursor.fetchall():
                try:
                    similar_board = chess.Board(row[0])
                    similar_piece_count = len(similar_board.piece_map())
                    
                    # Simple similarity based on piece count
                    similarity = 1.0 - abs(piece_count - similar_piece_count) / max(piece_count, similar_piece_count)
                    
                    if similarity >= similarity_threshold:
                        results.append({
                            'fen': row[0],
                            'evaluation': row[1],
                            'best_move': row[2],
                            'similarity': similarity
                        })
                except:
                    continue
            
            return sorted(results, key=lambda x: x['similarity'], reverse=True)
    
    def get_position_statistics(self):
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count positions
            cursor.execute('SELECT COUNT(*) FROM positions')
            position_count = cursor.fetchone()[0]
            
            # Count games
            cursor.execute('SELECT COUNT(*) FROM games')
            game_count = cursor.fetchone()[0]
            
            # Count openings
            cursor.execute('SELECT COUNT(*) FROM openings')
            opening_count = cursor.fetchone()[0]
            
            # Count cached analyses
            cursor.execute('SELECT COUNT(*) FROM analysis_cache')
            cache_count = cursor.fetchone()[0]
            
            # Average evaluation
            cursor.execute('SELECT AVG(evaluation) FROM positions WHERE evaluation IS NOT NULL')
            avg_eval = cursor.fetchone()[0] or 0.0
            
            return {
                'total_positions': position_count,
                'total_games': game_count,
                'total_openings': opening_count,
                'cached_analyses': cache_count,
                'average_evaluation': avg_eval
            }
    
    def cleanup_old_cache(self, days_old=30):
        """
        Clean up old cache entries.
        
        Args:
            days_old: Remove cache entries older than this many days
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM analysis_cache 
                WHERE timestamp < datetime('now', '-{} days')
            '''.format(days_old))
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            return deleted_count
    
    def export_positions(self, output_file, format_type='json'):
        """
        Export positions to file.
        
        Args:
            output_file: Output file path
            format_type: Export format ('json' or 'csv')
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM positions')
            positions = cursor.fetchall()
            
            if format_type == 'json':
                data = []
                for pos in positions:
                    data.append({
                        'fen': pos[1],
                        'evaluation': pos[2],
                        'depth': pos[3],
                        'best_move': pos[4],
                        'engine_name': pos[7]
                    })
                
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
            
            elif format_type == 'csv':
                import csv
                with open(output_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['FEN', 'Evaluation', 'Depth', 'Best Move', 'Engine'])
                    
                    for pos in positions:
                        writer.writerow([pos[1], pos[2], pos[3], pos[4], pos[7]])
    
    def import_positions(self, input_file, format_type='json'):
        """
        Import positions from file.
        
        Args:
            input_file: Input file path
            format_type: Import format ('json' or 'csv')
        """
        if format_type == 'json':
            with open(input_file, 'r') as f:
                data = json.load(f)
            
            for item in data:
                self.store_position(
                    fen=item['fen'],
                    evaluation=item.get('evaluation'),
                    depth=item.get('depth', 0),
                    best_move=item.get('best_move'),
                    engine_name=item.get('engine_name', 'imported')
                )
        
        elif format_type == 'csv':
            import csv
            with open(input_file, 'r') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    self.store_position(
                        fen=row['FEN'],
                        evaluation=float(row['Evaluation']) if row['Evaluation'] else None,
                        depth=int(row['Depth']) if row['Depth'] else 0,
                        best_move=row['Best Move'] if row['Best Move'] else None,
                        engine_name=row.get('Engine', 'imported')
                    )