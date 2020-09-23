import reverb
import time
import sys

server = reverb.Server(tables=[
    reverb.Table(
        name='my_table',
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=100,
        rate_limiter=reverb.rate_limiters.MinSize(1)),
    ],
    port=8000
)
while True:
    try:
        time.sleep(2)
    except KeyboardInterrupt:
        print('Exciting gracefully...')
        server.stop()
        sys.exit(0)

