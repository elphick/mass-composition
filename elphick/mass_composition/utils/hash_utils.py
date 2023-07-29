import hashlib
from pathlib import Path
from typing import Optional


def read_hash(filepath: Path) -> str:
    """Read a file and return the hash

    Args:
        filepath: The file to hash

    Returns:
        the string hash
    """

    buffer_size = 65536 * 1024  # read stuff in 64Mb chunks!
    md5 = hashlib.md5()

    with open(filepath, "rb") as f:
        while True:
            data = f.read(buffer_size)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()


def write_hash(filepath: Path) -> Path:
    h = read_hash(filepath)
    with open(filepath.with_suffix('.md5'), 'w') as f:
        f.writelines([h])
    return filepath.with_suffix('.md5')


def read_hash_file(filepath: Path) -> Optional[str]:
    res = None
    if filepath.with_suffix('.md5').exists():
        with open(filepath.with_suffix('.md5'), 'r') as f:
            stored_hash: str = f.readline()
        res = stored_hash
    return res


def check_hash(filepath: Path) -> bool:
    res: bool = False
    stored_hash: str = read_hash_file(filepath)
    new_hash: str = read_hash(filepath)
    if stored_hash == new_hash:
        res = True
    return res
