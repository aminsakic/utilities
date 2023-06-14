import os
import io
import sys
from typing import *


BLOCK_SIZE: int = 8192

class ropen:

    def __init__(self, name, block_size: int = BLOCK_SIZE):
        if block_size % 2:
            raise ValueError("An odd block size will break multi-byte encodings such as UTF-16")
        self._handle: file = open(name)
        self._handle.seek(0, os.SEEK_END)
        self._size: int = self._handle.tell()
        self._remaining: int = self._size
        self._block_size: int = block_size
        self._line_buffer: Iterable[str] = iter(())
        self._incomplete: str = ""
        self._buf: io.StringIO = io.StringIO()

        
    def __iter__(self):
        return self


    def _read_next_block(self) -> NoReturn:
        offset = min(self._remaining, self._block_size)
        # truncate the concatination buffer
        self._buf.truncate(0)
        # set the FP and our internal position to the new position
        self._remaining -= offset
        self._handle.seek(self._remaining)
        # read the next block into the concatination buffer and add the last incomplete line from the previous block
        self._buf.write(self._handle.read(offset))
        self._buf.write(self._incomplete)
        self._buf.seek(0)

        self._incomplete, *lines = self._buf.read().splitlines()
        if self._remaining == 0:
            lines = [self._incomplete] + lines
        self._line_buffer = reversed(lines)


    def __next__(self):
        line = next(self._line_buffer, None)
        if line is None:
            if self._remaining:
                self._read_next_block()
                return self.__next__()
            else:
                self._handle.close()
                raise StopIteration
        else:
            return line + os.linesep

    def __enter__(self):
         return self

    def __exit__(self, *args, **kwargs):
        self._handle.close()

if __name__ == "__main__":
    filename = ".bashrc"
    with open(filename) as ff:
        flist = list(ff)
    with ropen(filename) as fr:
        rlist = list(reversed(list(fr)))
   
    assert all(fl == rl for fl, rl in zip(flist, rlist)) 
