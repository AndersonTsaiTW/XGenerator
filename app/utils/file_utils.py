"""
File utility functions for atomic writes and ID generation
"""
import uuid
import shutil
from pathlib import Path
from typing import Any
import json


def generate_id() -> str:
    """Generate a unique ID using uuid4().hex"""
    return uuid.uuid4().hex


def atomic_write_json(file_path: Path, data: Any) -> None:
    """
    Atomically write JSON data to a file using temp file + rename.
    This prevents partial writes if the process is interrupted.
    """
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    
    try:
        # Write to temp file
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Atomic rename using os.replace (more reliable on Windows)
        import os
        os.replace(str(temp_path), str(file_path))
    except Exception as e:
        # Clean up temp file if something goes wrong
        try:
            if temp_path.exists():
                temp_path.unlink()
        except:
            pass  # Ignore cleanup errors
        raise e  # Re-raise the original error


def atomic_write_file(file_path: Path, content: bytes) -> None:
    """
    Atomically write binary content to a file using temp file + rename.
    """
    temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    
    try:
        # Write to temp file
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Atomic rename
        shutil.move(str(temp_path), str(file_path))
    except Exception:
        # Clean up temp file if something goes wrong
        if temp_path.exists():
            temp_path.unlink()
        raise
