from gym_minigrid.register import register
from gym_minigrid.envs import MultiRoomEnv

class MultiRoomEnvN7S4(MultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=7,
            maxNumRooms=7,
            maxRoomSize=4
        )

class MultiRoomEnvN6S4(MultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=6,
            maxNumRooms=6,
            maxRoomSize=4
        )
class MultiRoomEnvN6S4Easy(MultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=1,
            maxNumRooms=6,
            maxRoomSize=4
        )
register(
    id='MiniGrid-MultiRoom-N7-S4-v0',
    entry_point='golem.envs:MultiRoomEnvN7S4'
)
register(
    id='MiniGrid-MultiRoom-N6-S4-v0',
    entry_point='golem.envs:MultiRoomEnvN6S4'
)
register(
    id='MiniGrid-MultiRoom-N6-S4-Easy-v0',
    entry_point='golem.envs:MultiRoomEnvN6S4Easy'
)
