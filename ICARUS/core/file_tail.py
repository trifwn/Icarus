from io import BufferedReader
from typing import IO
from typing import Any


def tail(f: BufferedReader | IO[Any], lines: int = 20) -> list[bytes]:
    """
    Return The last N lines of a file

    Args:
        f (File): File to tail
        lines (int, optional): The number of N lines to return. Defaults to 20.

    Returns:
        list[bytes]: Last N lines
    """
    total_lines_wanted: int = lines

    BLOCK_SIZE = 1024
    f.seek(0, 2)
    block_end_byte: int = f.tell()
    lines_to_go: int = total_lines_wanted
    block_number = -1
    blocks = []
    while lines_to_go > 0 and block_end_byte > 0:
        if block_end_byte - BLOCK_SIZE > 0:
            f.seek(block_number * BLOCK_SIZE, 2)
            blocks.append(f.read(BLOCK_SIZE))
        else:
            f.seek(0, 0)
            blocks.append(f.read(block_end_byte))
        lines_found = blocks[-1].count(b"\n")
        lines_to_go -= lines_found
        block_end_byte -= BLOCK_SIZE
        block_number -= 1
    all_read_text: bytes = b"".join(reversed(blocks))
    return all_read_text.splitlines()[-total_lines_wanted:]
