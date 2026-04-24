#!/usr/bin/env python3
"""
Clean launcher for LW5 task.py
Runs training with noisy TensorFlow/WSL startup logs filtered out.
"""

import subprocess
import sys
import re
from pathlib import Path

def filter_stderr(line):
    """Filter out known harmless TensorFlow/WSL startup messages."""
    patterns = [
        r"absl::InitializeLog",
        r"cuda_executor\.cc:\d+",
        r"numa_node",
        r"Your kernel may have been built without NUMA support",
        r"gpu_timer\.cc:\d+",
    ]
    return not any(re.search(pattern, line) for pattern in patterns)

def main():
    """Run task.py with stderr filtering."""
    script_dir = Path(__file__).resolve().parent
    task_py = script_dir / "task.py"
    workspace_root = script_dir.parent.parent  # Go up 2 levels to GPU_Workspace
    venv_python = workspace_root / ".venv" / "bin" / "python"
    
    if not task_py.exists():
        print(f"Error: {task_py} not found", file=sys.stderr)
        sys.exit(1)
    
    # Use venv Python if available, otherwise fall back to current Python
    python_exe = str(venv_python) if venv_python.exists() else sys.executable
    
    try:
        # Run task.py and filter stderr
        process = subprocess.Popen(
            [python_exe, str(task_py)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(script_dir),
        )
        
        # Stream stdout directly (unfiltered)
        for line in process.stdout:
            print(line, end="")
        
        # Stream stderr with filtering
        for line in process.stderr:
            if filter_stderr(line):
                print(line, end="", file=sys.stderr)
        
        # Wait for completion
        return_code = process.wait()
        sys.exit(return_code)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error running task.py: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
