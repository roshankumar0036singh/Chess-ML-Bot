import chess
import chess.pgn
import chess.engine
from collections import defaultdict

class GameAnalyzer:
    """
    Analyzes chess games for mistakes, missed opportunities, and learning points.
    Provides detailed post-game analysis and improvement suggestions.
    """
    
    def __init__(self, engine_path=None):
        """
        Initialize game analyzer.
        
        Args:
            engine_path: Path to external engine (e.g., Stockfish) for analysis
        """
        self.engine_path = engine_path
        self.external_engine = None
        
        if engine_path:
            try:
                self.external_engine = chess.engine.SimpleEngine.popen_uci(engine_path)
                print(f"Loaded analysis engine: {engine_path}")
            except Exception as e:
                print(f"Could not load external engine: {e}")
    
    def analyze_game(self, game, depth=15, time_limit=1.0):
        """
        Analyze a complete game.
        
        Args:
            game: chess.pgn.Game object
            depth: Analysis depth
            time_limit: Time limit per position in seconds
            
        Returns:
            dict: Complete game analysis
        """
        analysis = {
            'game_info': self._extract_game_info(game),
            'move_analysis': [],
            'mistakes': [],
            'missed_opportunities': [],
            'opening_analysis': {},
            'endgame_analysis': {},
            'statistics': {},
            'learning_points': []
        }
        
        board = game.board()
        move_number = 1
        
        # Analyze each move
        for move in game.mainline_moves():
            move_analysis = self._analyze_position(board, move, depth, time_limit)
            move_analysis['move_number'] = move_number
            move_analysis['fen'] = board.fen()
            
            analysis['move_analysis'].append(move_analysis)
            
            # Check for mistakes
            if move_analysis.get('is_mistake'):
                analysis['mistakes'].append(move_analysis)
            
            # Check for missed opportunities
            if move_analysis.get('missed_opportunity'):
                analysis['missed_opportunities'].append(move_analysis)
            
            board.push(move)
            
            if board.turn == chess.WHITE:
                move_number += 1
        
        # Analyze specific game phases
        analysis['opening_analysis'] = self._analyze_opening(analysis['move_analysis'][:20])
        analysis['endgame_analysis'] = self._analyze_endgame(analysis['move_analysis'])
        
        # Calculate statistics
        analysis['statistics'] = self._calculate_statistics(analysis)
        
        # Generate learning points
        analysis['learning_points'] = self._generate_learning_points(analysis)
        
        return analysis
    
    def _extract_game_info(self, game):
        """Extract basic game information."""
        headers = game.headers
        
        return {
            'white': headers.get('White', 'Unknown'),
            'black': headers.get('Black', 'Unknown'),
            'result': headers.get('Result', '*'),
            'date': headers.get('Date', 'Unknown'),
            'event': headers.get('Event', 'Unknown'),
            'white_elo': headers.get('WhiteElo', 'Unknown'),
            'black_elo': headers.get('BlackElo', 'Unknown'),
            'time_control': headers.get('TimeControl', 'Unknown')
        }
    
    def _analyze_position(self, board, played_move, depth=15, time_limit=1.0):
        """
        Analyze a single position and move.
        
        Args:
            board: Current position
            played_move: Move that was played
            depth: Analysis depth
            time_limit: Time limit for analysis
            
        Returns:
            dict: Position analysis
        """
        analysis = {
            'played_move': str(played_move),
            'evaluation': self._evaluate_position(board),
            'best_moves': [],
            'is_mistake': False,
            'mistake_severity': None,
            'centipawn_loss': 0,
            'missed_opportunity': False,
            'comments': []
        }
        
        # Get best moves from engine or evaluation
        if self.external_engine:
            try:
                info = self.external_engine.analyse(board, chess.engine.Limit(depth=depth, time=time_limit))
                analysis['evaluation'] = info['score'].relative.score(mate_score=10000)
                
                # Get multiple best moves
                multipv_info = self.external_engine.analyse(
                    board, 
                    chess.engine.Limit(depth=depth, time=time_limit),
                    multipv=3
                )
                
                for i, pv_info in enumerate(multipv_info):
                    if pv_info['pv']:
                        move_info = {
                            'move': str(pv_info['pv'][0]),
                            'evaluation': pv_info['score'].relative.score(mate_score=10000),
                            'rank': i + 1
                        }
                        analysis['best_moves'].append(move_info)
                
            except Exception as e:
                print(f"Engine analysis error: {e}")
                analysis['best_moves'] = [{'move': str(played_move), 'evaluation': 0, 'rank': 1}]
        else:
            # Use internal evaluation
            analysis['best_moves'] = self._get_best_moves_internal(board, 3)
        
        # Analyze move quality
        if analysis['best_moves']:
            best_eval = analysis['best_moves'][0]['evaluation']
            
            # Calculate evaluation after played move
            board.push(played_move)
            played_eval = -self._evaluate_position(board)  # Flip for opponent
            board.pop()
            
            centipawn_loss = best_eval - played_eval
            analysis['centipawn_loss'] = centipawn_loss
            
            # Classify mistake severity
            if centipawn_loss > 300:
                analysis['is_mistake'] = True
                analysis['mistake_severity'] = 'blunder'
            elif centipawn_loss > 100:
                analysis['is_mistake'] = True
                analysis['mistake_severity'] = 'mistake'
            elif centipawn_loss > 50:
                analysis['is_mistake'] = True
                analysis['mistake_severity'] = 'inaccuracy'
            
            # Check if played move is in top 3
            played_move_str = str(played_move)
            top_moves = [m['move'] for m in analysis['best_moves'][:3]]
            
            if played_move_str not in top_moves:
                analysis['missed_opportunity'] = True
        
        return analysis
    
    def _evaluate_position(self, board):
        """Evaluate position using internal evaluator."""
        from core.evaluation import PositionEvaluator
        evaluator = PositionEvaluator()
        return evaluator.evaluate(board)
    
    def _get_best_moves_internal(self, board, num_moves=3):
        """Get best moves using internal search."""
        from core.search import MinimaxSearch
        from core.evaluation import PositionEvaluator
        
        evaluator = PositionEvaluator()
        search = MinimaxSearch(evaluator, max_depth=3)
        
        moves_with_eval = []
        
        for move in board.legal_moves:
            board.push(move)
            eval_score = -evaluator.evaluate(board)  # Flip for opponent
            board.pop()
            
            moves_with_eval.append({
                'move': str(move),
                'evaluation': eval_score,
                'rank': 0  # Will be set after sorting
            })
        
        # Sort by evaluation
        moves_with_eval.sort(key=lambda x: x['evaluation'], reverse=True)
        
        # Set ranks
        for i, move_info in enumerate(moves_with_eval[:num_moves]):
            move_info['rank'] = i + 1
        
        return moves_with_eval[:num_moves]
    
    def _analyze_opening(self, opening_moves):
        """Analyze opening phase."""
        if not opening_moves:
            return {}
        
        mistakes_in_opening = [m for m in opening_moves if m.get('is_mistake')]
        
        return {
            'length': len(opening_moves),
            'mistakes': len(mistakes_in_opening),
            'average_centipawn_loss': sum(m.get('centipawn_loss', 0) for m in opening_moves) / len(opening_moves),
            'quality': 'good' if len(mistakes_in_opening) <= 1 else 'poor'
        }
    
    def _analyze_endgame(self, all_moves):
        """Analyze endgame phase (last 20 moves)."""
        if len(all_moves) < 20:
            return {}
        
        endgame_moves = all_moves[-20:]
        mistakes_in_endgame = [m for m in endgame_moves if m.get('is_mistake')]
        
        return {
            'length': len(endgame_moves),
            'mistakes': len(mistakes_in_endgame),
            'average_centipawn_loss': sum(m.get('centipawn_loss', 0) for m in endgame_moves) / len(endgame_moves),
            'quality': 'good' if len(mistakes_in_endgame) <= 2 else 'poor'
        }
    
    def _calculate_statistics(self, analysis):
        """Calculate game statistics."""
        moves = analysis['move_analysis']
        
        if not moves:
            return {}
        
        total_moves = len(moves)
        mistakes = analysis['mistakes']
        
        # Count by severity
        blunders = [m for m in mistakes if m.get('mistake_severity') == 'blunder']
        errors = [m for m in mistakes if m.get('mistake_severity') == 'mistake']
        inaccuracies = [m for m in mistakes if m.get('mistake_severity') == 'inaccuracy']
        
        # Calculate averages
        total_centipawn_loss = sum(m.get('centipawn_loss', 0) for m in moves)
        average_centipawn_loss = total_centipawn_loss / total_moves
        
        return {
            'total_moves': total_moves,
            'total_mistakes': len(mistakes),
            'blunders': len(blunders),
            'mistakes': len(errors),
            'inaccuracies': len(inaccuracies),
            'total_centipawn_loss': total_centipawn_loss,
            'average_centipawn_loss': average_centipawn_loss,
            'accuracy': max(0, 100 - (total_centipawn_loss / 10))  # Rough accuracy percentage
        }
    
    def _generate_learning_points(self, analysis):
        """Generate learning points based on analysis."""
        learning_points = []
        
        stats = analysis['statistics']
        
        # Opening advice
        opening = analysis.get('opening_analysis', {})
        if opening.get('mistakes', 0) > 1:
            learning_points.append({
                'category': 'Opening',
                'priority': 'high',
                'message': 'Review opening principles - multiple mistakes in opening phase'
            })
        
        # Tactical advice
        if stats.get('blunders', 0) > 2:
            learning_points.append({
                'category': 'Tactics',
                'priority': 'high',
                'message': 'Practice tactical puzzles - several blunders in game'
            })
        
        # Endgame advice
        endgame = analysis.get('endgame_analysis', {})
        if endgame.get('mistakes', 0) > 2:
            learning_points.append({
                'category': 'Endgame',
                'priority': 'medium',
                'message': 'Study endgame technique - errors in endgame phase'
            })
        
        # Calculation advice
        if stats.get('average_centipawn_loss', 0) > 50:
            learning_points.append({
                'category': 'Calculation',
                'priority': 'medium',
                'message': 'Improve calculation skills - high average centipawn loss'
            })
        
        # Positive feedback
        if stats.get('accuracy', 0) > 85:
            learning_points.append({
                'category': 'General',
                'priority': 'low',
                'message': 'Good game quality - maintain this level'
            })
        
        return learning_points
    
    def generate_analysis_report(self, analysis):
        """Generate human-readable analysis report."""
        stats = analysis['statistics']
        game_info = analysis['game_info']
        
        report = f"""
Chess Game Analysis Report
==========================

Game Information:
{game_info['white']} vs {game_info['black']}
Result: {game_info['result']}
Date: {game_info['date']}

Overall Statistics:
- Total Moves: {stats.get('total_moves', 0)}
- Accuracy: {stats.get('accuracy', 0):.1f}%
- Total Centipawn Loss: {stats.get('total_centipawn_loss', 0)}
- Average Centipawn Loss: {stats.get('average_centipawn_loss', 0):.1f}

Mistakes Breakdown:
- Blunders: {stats.get('blunders', 0)}
- Mistakes: {stats.get('mistakes', 0)}
- Inaccuracies: {stats.get('inaccuracies', 0)}

Phase Analysis:
"""
        
        # Opening analysis
        opening = analysis.get('opening_analysis', {})
        if opening:
            report += f"- Opening Quality: {opening.get('quality', 'unknown')}\n"
            report += f"- Opening Mistakes: {opening.get('mistakes', 0)}\n"
        
        # Endgame analysis
        endgame = analysis.get('endgame_analysis', {})
        if endgame:
            report += f"- Endgame Quality: {endgame.get('quality', 'unknown')}\n"
            report += f"- Endgame Mistakes: {endgame.get('mistakes', 0)}\n"
        
        # Learning points
        report += "\nLearning Points:\n"
        for point in analysis.get('learning_points', []):
            priority = point['priority'].upper()
            report += f"- [{priority}] {point['category']}: {point['message']}\n"
        
        return report
    
    def analyze_tactical_themes(self, board):
        """Analyze tactical themes in position."""
        themes = []
        
        # Check for pins
        if self._has_pins(board):
            themes.append("pins")
        
        # Check for forks
        if self._has_forks(board):
            themes.append("forks")
        
        # Check for discovered attacks
        if self._has_discovered_attacks(board):
            themes.append("discovered_attacks")
        
        # Check for back rank weakness
        if self._has_back_rank_weakness(board):
            themes.append("back_rank")
        
        return themes
    
    def _has_pins(self, board):
        """Check for pin tactics (simplified)."""
        # This is a simplified implementation
        # A full implementation would require more sophisticated analysis
        return False  # Placeholder
    
    def _has_forks(self, board):
        """Check for fork tactics (simplified)."""
        return False  # Placeholder
    
    def _has_discovered_attacks(self, board):
        """Check for discovered attack tactics (simplified)."""
        return False  # Placeholder
    
    def _has_back_rank_weakness(self, board):
        """Check for back rank weakness."""
        # Check if king is trapped on back rank
        for color in [chess.WHITE, chess.BLACK]:
            king_square = board.king(color)
            if king_square is not None:
                king_rank = chess.square_rank(king_square)
                back_rank = 0 if color == chess.WHITE else 7
                
                if king_rank == back_rank:
                    # Check if king has escape squares
                    escape_squares = 0
                    for direction in [-1, 0, 1]:
                        for file_dir in [-1, 0, 1]:
                            if direction == 0 and file_dir == 0:
                                continue
                            
                            try:
                                new_square = king_square + direction * 8 + file_dir
                                if 0 <= new_square <= 63:
                                    if not board.piece_at(new_square):
                                        escape_squares += 1
                            except:
                                pass
                    
                    if escape_squares == 0:
                        return True
        
        return False
    
    def cleanup(self):
        """Cleanup resources."""
        if self.external_engine:
            self.external_engine.quit()