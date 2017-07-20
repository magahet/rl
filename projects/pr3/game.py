import random


class Soccer(object):
    def reset(self):
        self.players = [(2, 1), (1, 1)]
        self.ball = 1
        return str(self), (0, 0), False

    def __str__(self):
        return ','.join([str(self.players), str(self.ball)])

    def step(self, actions):
        movement = {
            0: lambda p: (p[0] - 1, p[1]),  # left
            1: lambda p: (p[0] + 1, p[1]),  # right
            2: lambda p: (p[0], p[1] - 1),  # down
            3: lambda p: (p[0], p[1] + 1),  # up
            4: lambda p: p,
        }

        for idx0 in random.sample(range(2), 2):
            idx1 = abs(idx0 - 1)
            p0 = self.players[idx0]
            p1 = self.players[idx1]
            nx, ny = movement[actions[idx0]](p0)

            # Hit wall
            if nx < 0 or ny < 0 or nx > 3 or ny > 1:
                continue
            # Collision
            elif (nx, ny) == p1:
                self.ball = idx1
                continue
            else:
                self.players[idx0] = (nx, ny)

            # Goal
            if self.ball == 0 and self.players[0][0] == 0:
                return str(self), (100, -100), True
            elif self.ball == 1 and self.players[1][0] == 3:
                return str(self), (-100, 100), True

        return str(self), (0, 0), False
