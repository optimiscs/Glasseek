"""BC+ bash child: allow Unix sockets, deny creation of network sockets.

The research/model process and fixed-corpus server run outside this filter.
The bash helper reaches its per-task web bridge over a Unix domain socket.
"""
import ctypes
import errno
import os
import platform
import sys


def restrict_network():
    if platform.machine() != 'x86_64':raise RuntimeError('fixed-corpus network filter currently requires Linux x86_64')
    class Filter(ctypes.Structure):_fields_=[('code',ctypes.c_ushort),('jt',ctypes.c_ubyte),('jf',ctypes.c_ubyte),('k',ctypes.c_uint32)]
    class Program(ctypes.Structure):_fields_=[('len',ctypes.c_ushort),('filter',ctypes.POINTER(Filter))]
    # Validate syscall ABI, then only AF_UNIX (1) may be used for socket(41)
    # and socketpair(53). The filter is inherited by every bash descendant.
    instructions=[(0x20,0,0,4),(0x15,1,0,0xc000003e),(0x06,0,0,0x80000000),
        (0x20,0,0,0),(0x45,0,1,0x40000000),(0x06,0,0,0x00050000|errno.EPERM),
        (0x15,2,0,41),(0x15,1,0,53),(0x06,0,0,0x7fff0000),
        (0x20,0,0,16),(0x15,1,0,1),(0x06,0,0,0x00050000|errno.EPERM),(0x06,0,0,0x7fff0000)]
    filters=(Filter*len(instructions))(*(Filter(*row) for row in instructions));program=Program(len(filters),filters)
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(38,1,0,0,0) or libc.prctl(22,2,ctypes.byref(program),0,0):
        raise OSError(ctypes.get_errno(),'could not install fixed-corpus network filter')


if __name__=='__main__':
    restrict_network()
    if len(sys.argv)<2:raise SystemExit('command required')
    os.execvpe(sys.argv[1],sys.argv[1:],os.environ)
