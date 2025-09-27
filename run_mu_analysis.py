#!/usr/bin/env python3
"""
Simple runner script for the mu time-averaged analysis.
This script can be run from the project root directory.
"""

import sys
import os

# Add the mass injection directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'mass injection'))

# Change to the mass injection directory
os.chdir(os.path.join(os.path.dirname(__file__), 'mass injection'))

# Now import and run the analysis
exec(open('plot_mu_time_averaged.py').read())
