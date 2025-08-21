"""UI package for Chess ML Bot"""

from .cli import CLIInterface

# Conditional imports for optional dependencies
try:
    from .gui import ChessGUI
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    ChessGUI = None

try:
    from .web_interface import create_web_interface
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False
    create_web_interface = None

__all__ = [
    'CLIInterface',
    'ChessGUI',
    'create_web_interface',
    'GUI_AVAILABLE',
    'WEB_AVAILABLE'
]