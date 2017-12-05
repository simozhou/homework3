import time
import sorting
import bst
import timeit
import platform
import random
import csv
import numpy as np
import pandas
import matplotlib.pyplot as plt

DEFAULT_NUMBER = 100000  # 100k
DEFAULT_POPULATION = [10,100,1000,10000,100000]  # 1m


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
            for i in np.logspace(1.0, np.log10(max_val, dtype=float), base=10.0, endpoint=True, dtype=int):
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
        print("sorting done!")
        # BSTs implementation
        # TODO implement the bst module
        # the idea was to get the maximum lenght dict and populate it with timing and storing the values in
        # self.test_result as {'bst_insertion': {bst.size(): time}}

        insertion_counter, tree = 0, bst.BinarySearchTree()
        for i in range(DEFAULT_POPULATION[-1]):
            insertion_counter += 1
            if insertion_counter in DEFAULT_POPULATION:
                self._test_it_binary_get_max(i, tree)
                self._test_it_binary_get_random(i, tree)
                self._test_it_binary_insertion(0, tree, i)
            else:
                tree.put(i, 0)
        print("insertion timing done!")
        # TODO this function doesn't work to me, but we'll see
        for i in range(DEFAULT_POPULATION[-1]):
            if i in DEFAULT_POPULATION:
                self._test_it_binary_remove(i, tree)
            else:
                tree.delete(i)

    def _test_it_quick_sort(self, arr, key):
        self.test_result['quick_sort'][key] = timeit.timeit("sorting.quick_sort(" + str(arr) + ")",
                                                            globals=globals(), number=10)

    def _test_it_merge_sort(self, arr, key):
        self.test_result['merge_sort'][key] = timeit.timeit("sorting.merge(" + str(arr) + ")", globals=globals(),
                                                            number=10)

    def _test_it_binary_insertion(self, val, tree, key):
        self.test_result['binary_insertion'][key] = timeit.timeit("tree.put(" + str(key) + "," + val + ")",
                                                                  globals=globals(), number=1)

    def _test_it_binary_remove(self, key, tree):
        self.test_result["binary_delete"][key] = timeit.timeit("tree.delete(" + str(key) + ")", globals=globals(),
                                                               number=1)

    def _test_it_binary_get_max(self, key, tree):
        self.test_result['binary_get_max'][key] = timeit.timeit("tree.findMax()", globals=globals(), number=10)

    def _test_it_binary_get_random(self, key, tree):
        self.test_result['binary_get_random'][key] = timeit.timeit("tree.get(key)",
                                                                   setup="key = random.randint(0, len(tree)-1)",
                                                                   globals=globals(),
                                                                   number=10)

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


if __name__ == '__main__':
    a = TimeTest()
    print("time test generated!")
    t = time.time()
    a.test_it()
    t = time.time() - t
