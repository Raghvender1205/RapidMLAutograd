import queue
import pyopencl as cl
from collections import defaultdict


class MemoryPool:
    def __init__(self):
        self.pools = defaultdict(queue.Queue)

    def allocate(self, context, size):
        if not self.pools[size].empty():
            return self.pools[size].get()
        else:
            # No available buffer, create a new one
            return cl.Buffer(context, cl.mem_flags.READ_WRITE, size)

    def free(self, buffer):
        self.pools[buffer.size].put(buffer)
