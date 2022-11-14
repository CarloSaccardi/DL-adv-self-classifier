class test:
    def __init__(self, a = 1):
        super(test, self).__init__()
        self.a = a

    def print(self, b):
        print(self.a + b)



t = test()
t(2)