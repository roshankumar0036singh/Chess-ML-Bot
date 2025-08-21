# Syzygy Endgame Tablebases

This directory is for Syzygy tablebase files (.rtbw files).

## Quick Setup:
The bot works fine without tablebases, but for perfect endgame play:

1. Download 3-4-5 piece tables (~1.2GB):
   wget -r -np -nH --cut-dirs=3 https://tablebase.lichess.ovh/tables/standard/3-4-5/

2. Or manually download from:
   https://syzygy-tables.info/

3. Place .rtbw files in this directory

## Current Status:
No tablebase files found - this is OK for normal play.
The bot will use its neural network for endgame evaluation.
