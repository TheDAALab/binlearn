#!/usr/bin/env python3
"""Test script to run pylint and capture output."""

import subprocess
import sys

def run_pylint():
    """Run pylint and return output."""
    try:
        # Run pylint on binning directory
        result = subprocess.run([
            sys.executable, '-m', 'pylint', 'binning', 
            '--output-format=text', '--max-line-length=100'
        ], capture_output=True, text=True, cwd='/home/gykovacs/workspaces/binning')
        
        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        print(f"\nReturn code: {result.returncode}")
        
    except Exception as e:
        print(f"Error running pylint: {e}")

if __name__ == "__main__":
    run_pylint()
