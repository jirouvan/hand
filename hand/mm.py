class test():
    def __init__(self, data=1):
        self.dat = data
        self.__pd = data+1

    def __iter__(self):
        return self

    # 唯一需要注意的就是__next__中必须控制iterator的结束条件，不然就死循环了
    def __next__(self):
        if self.data > 7:
            raise StopIteration
        else:
            self.data += 1
            return self.data


p1 = test(3)
print(p1.dat)
print(p1.__pd)
