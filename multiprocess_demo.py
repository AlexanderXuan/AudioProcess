# author:   xuan
# date:     2020/06/10
# function: multiprocessing a simple programme
import multiprocessing.dummy


def calc_mult(num1, num2):
    print(num1 * num2)


pool = multiprocessing.dummy.Pool(processes=3)
params = [(1, 2), (3, 4), (5, 6)]
results = []
for param in params:
    pool.apply_async(calc_mult, param)
pool.close()
pool.join()

