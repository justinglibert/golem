import os
import sys
f = open(os.path.expanduser(sys.argv[1]), 'r')
pids = f.read()
for p in pids.split('\n'):
    if p == '':
        continue
    pid = int(p)
    try:
        # 9 is a force kill, 2 is a keyboard interrupt
        os.kill(pid, 9)
    except Exception as e:
        print(e)
        continue
