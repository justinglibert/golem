import os
import sys
f = open(os.path.expanduser(sys.argv[1]), 'r')
pids = f.read()
for p in pids.split('\n'):
    if p == '':
        continue
    pid = int(p)
    try:
        os.kill(pid, 2)
    except Exception:
        continue


