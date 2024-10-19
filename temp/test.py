class Class1:
    def __init__(self):
        self.object = None

    def get(self):
        return 5

    def get_execute(self):
        return self.object.get()

class Class2:
    def __init__(self, object):
        self.object = object

    def get(self):
        return self.object.get()


ob1 = Class1()
ob2 = Class2(ob1)
ob1.object = ob2

print(ob1.get_execute())