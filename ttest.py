import time
import sorting
import bst
import timeit
import platform
import random
import csv
import numpy as np
import pandas as pd
import functools
import matplotlib.pyplot as plt
import heapq

# import bst
# TODO repair the heap module on maxChild/minChild
# import heap

random.seed(123)
DEFAULT_NUMBER = 100000  # 100k
DEFAULT_POPULATION = range(1000000)  # 1m
DEFAULT_SIZES = [10, 100, 1000, 10000, 100000]


class TimeTest(object):
    """returns an object which tests the time efficiency of a multi-dimensional array of increasing sizes of random
    integers over functions of sorting, binary tree insertions and deletions and heap insertions/get/removal.

    There's the possibility to get a summary of the tests results with CPU information, OS infos and plot of
    array size/time of all the executed tests.

    Csv file writer of the tests results for statistical purposes has been implemented too.

    """

    def __init__(self, array=None, max_val=DEFAULT_NUMBER):
        # this is the dict of dicts where the values obtained from the test_it func will be stored.
        self.test_result = dict(quick_sort={}, merge_sort={}, binary_insertion={}, binary_get_random={},
                                binary_delete={}, binary_get_max={},
                                heap_insert={}, heap_get_max={}, heap_remove={})
        # dict of all the list to use for testing
        self.array_pool = {}
        # both for summary() purrrrrposes
        self.cpu = platform.processor()
        self.os = platform.platform()

        if array is None:
            # we generate the arrays of random numbers with a logarithmic distance one with the other
            for i in np.logspace(1.0, np.log10(max_val, dtype=float), base=10.0, endpoint=False, dtype=int):
                self.array_pool[i] = random.sample(DEFAULT_POPULATION, k=i)
        # to use a more accurate logarithmic scale, to improve plotting and statistics quality
        elif array is "e_log":
            for i in DEFAULT_SIZES:
                self.array_pool[i] = random.sample(DEFAULT_POPULATION, k=i)
        else:
            for lst in array:
                self.array_pool[len(lst)] = lst

    def test_it(self):
        """generates a number of arrays of increasing size each of random integers and tests them over
        the given functions. Eventually adds the results to the self.test_result dictionary with template:

        {function: {size_array: time}}

        """
        for key, arr in self.array_pool.items():
            # sorting algorithms

            self._test_it_quick_sort(arr, key)
            self._test_it_merge_sort(arr, key)
        print("sorting timing done!")

        # BSTs implementation

        insertion_counter = 0
        global tree
        tree = bst.BinarySearchTree()
        for bst_key in self.array_pool[100000]:
            insertion_counter += 1
            if insertion_counter in self.array_pool.keys():
                self._test_it_binary_get_max(insertion_counter)
                self._test_it_binary_get_random(insertion_counter)
                # VALUE, KEY
                self._test_it_binary_insertion(bst_key, insertion_counter)
            else:
                # KEY, VALUE
                tree.put(bst_key, 0)
        print("insertion, get random, get max timing done!")

        # TODO set array_pool[100000] to maximum in  every case
        delete_counter = len(tree)
        for bst_key in self.array_pool[DEFAULT_NUMBER]:
            delete_counter -= 1
            if delete_counter in self.array_pool.keys():
                self._test_it_binary_delete(bst_key, delete_counter)
            else:
                tree.delete(bst_key)
        print("removal timing done!")

        # heap implementation
        global heaper
        heaper = []
        heap_counter = 0
        for heap_key in self.array_pool[DEFAULT_NUMBER]:
            heap_counter += 1
            if heap_counter in self.array_pool.keys():
                self._test_it_heap_get_max(heap_counter)
                self._test_it_heap_insert(heap_counter, heap_key)
            else:
                heapq.heappush(heaper, heap_key)

        for heap_key in self.array_pool[DEFAULT_NUMBER]:
            heap_counter -= 1
            if heap_counter in self.array_pool.keys():
                self._test_it_heap_del_max(heap_counter)
            else:
                heapq.heappop(heaper)
        print("heap removal timing done!")

        # self._pandator()
        # self.test_result.head()

    def csv(self, name='test' + "right now"):  # TODO implement a right now stringer
        """generates a csv file with the results of the test_it function, returns a "Run test_it before requesting
        csv report" if the self.test_result field has not been populated already.

        """
        pass

    def summary(self):
        """prints out a summary with CPU, OS infos, plots of all the tests [array_size/time]
        and POSSIBLY a pdf markdown with all of it in a beautiful graphics.

        Returns a "Run test_it before requesting a  test summary" if the self.test_result
        field has not been populated already.


        """
        pass

    # sorting impl
    def _test_it_quick_sort(self, arr, key):
        self.test_result['quick_sort'][key] = timeit.timeit("sorting.quick_sort(" + str(arr) + ")",
                                                            globals=globals(), number=10)

    def _test_it_merge_sort(self, arr, key):
        self.test_result['merge_sort'][key] = timeit.timeit("sorting.merge(" + str(arr) + ")", globals=globals(),
                                                            number=10)

    # bst impl
    def _test_it_binary_insertion(self, val, key):
        self.test_result['binary_insertion'][key] = timeit.timeit("tree.put(" + str(val) + ",0)",
                                                                  globals=globals(), number=1)

    def _test_it_binary_delete(self, val, key):
        self.test_result["binary_delete"][key] = timeit.timeit("tree.delete(" + str(val) + ")", globals=globals(),
                                                               number=1)

    def _test_it_binary_get_max(self, key):
        self.test_result['binary_get_max'][key] = timeit.timeit("tree.findMax()", globals=globals(), number=10)

    def _test_it_binary_get_random(self, key):
        rand_key = random.randint(0, len(tree))
        self.test_result['binary_get_random'][key] = timeit.timeit("tree.get(" + str(rand_key) + ")",
                                                                   globals=globals(),
                                                                   number=10)

    #heap impl
    def _test_it_heap_insert(self, key, value):
        self.test_result['heap_insert'][key] = timeit.timeit('heapq.heappush(heaper,'+str(value)+')', number=1,
                                                             globals=globals())

    # TODO repair this function
    def _test_it_heap_get_max(self, key):
        self.test_result['heap_get_max'][key] = timeit.timeit('heaper[0]', number=10, globals=globals())

    def _test_it_heap_del_max(self, key):
        self.test_result['heap_get_max'][key] = timeit.timeit('heapq.heappop(heaper)', number=10, globals=globals())

    # TODO fix this function
    def _pandator(self):
        """transform the collected results from dict to pandas DataFrames, merging them together into a unique Df"""
        for key, value in self.test_result.items():
            self.test_result[key] = pd.DataFrame(value, columns=['array size', key])
        self.test_result = functools.reduce(lambda left, right: pd.merge(left, right, on='array size'),
                                            self.test_result.values())


if __name__ == '__main__':
    a = TimeTest(array='e_log')
    print("time test generated!")
    t = time.time()
    print(a.test_it())
    t = time.time() - t
    print(a.test_result, t)
