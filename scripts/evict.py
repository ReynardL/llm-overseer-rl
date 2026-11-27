# Drop clean page cache for files we own (posix_fadvise DONTNEED, no root). Pages still mapped by a live process stay.
import os, sys
n = 0
for root in sys.argv[1:]:
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            try:
                if os.path.islink(p) or not os.path.isfile(p): continue
                fd = os.open(p, os.O_RDONLY)
                try: os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); n += 1
                finally: os.close(fd)
            except OSError: pass
print("evicted files:", n)
