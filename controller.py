class Controller:
    def __init__(self, v1, v2, f):
        self.v1 = v1
        self.v2 = v2
        self.f = f

    def run(self):
        while True:
            value = input('v1 v2:').split()
            if value[0] == 'q':
                break

            self.v1 = float(value[0])
            if len(value) > 1:
                self.v2 = float(value[1])

            self.f(self.v1, self.v2)
