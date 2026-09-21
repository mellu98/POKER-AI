#!/usr/bin/env python
"""
Poker Assistant Test Runner

Run all tests with coverage reporting.

Usage:
    python run_tests.py                  # Run all tests
    python run_tests.py --unit           # Run unit tests only
    python run_tests.py --integration    # Run integration tests only
    python run_tests.py --ui             # Run UI tests only
    python run_tests.py --fast           # Skip slow tests
    python run_tests.py --coverage       # Run with coverage report
    python run_tests.py --verbose        # Verbose output
    python run_tests.py --module strategy    # Test specific module
"""
import subprocess
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run Poker Assistant tests")
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--ui', action='store_true', help='Run UI tests only')
    parser.add_argument('--fast', action='store_true', help='Skip slow tests')
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--module', type=str, help='Test specific module (strategy, detection, capture, ui, utils)')
    parser.add_argument('--html', action='store_true', help='Generate HTML coverage report')
    args = parser.parse_args()

    # Base pytest command
    cmd = [sys.executable, '-m', 'pytest']

    # Add verbosity
    if args.verbose:
        cmd.append('-v')
    else:
        cmd.append('-v')  # Always verbose for readability

    # Add markers
    markers = []
    if args.unit:
        markers.append('unit')
    if args.integration:
        markers.append('integration')
    if args.ui:
        markers.append('ui')
    if args.fast:
        markers.append('not slow')

    if markers:
        cmd.extend(['-m', ' and '.join(markers)])

    # Add coverage
    if args.coverage or args.html:
        cmd.extend(['--cov=src', '--cov-report=term-missing'])
        if args.html:
            cmd.append('--cov-report=html')

    # Add module path
    if args.module:
        module_path = f'tests/test_{args.module}'
        cmd.append(module_path)
    else:
        cmd.append('tests/')

    # Add color output
    cmd.append('--color=yes')

    # Print command
    print(f"\n{'='*60}")
    print("POKER ASSISTANT TEST SUITE")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    # Run tests
    result = subprocess.run(cmd, cwd=Path(__file__).parent)

    # Print summary
    print(f"\n{'='*60}")
    if result.returncode == 0:
        print("ALL TESTS PASSED!")
    else:
        print(f"TESTS FAILED (exit code: {result.returncode})")
    print(f"{'='*60}\n")

    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
