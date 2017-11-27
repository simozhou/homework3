import time
import timeit
import platform
import random
import csv


DEFAULT_NUMBER = 100000
STANDARD_DIMENSIONS = [10, 15, 100, 150, 1000, 1500, 10000, 15000, 100000]
DEFAULT_POPULATION = range(10000000)


class TimeTest(object):
    """returns an object which tests the time efficiency of a multi-dimensional array of increasing sizes of random
    integers over functions of sorting, binary tree insertions and deletions and heap insertions/get/removal.

    There's the possibility to get a summary of the tests results with CPU information, OS infos and plot of
    array size/time of all the executed tests.

    Csv file writer of the tests results for statistical purposes has been implemented too.

    """

    def __init__(self, array=None):
        self.test_result = dict(quick_sort={}, merge_sort={}, binary_insertion={}, binary_get_random={}, binary_delete={},
                                heap_insert={}, heap_get_max={}, heap_remove={})
        self.array_pool = {}
        self.cpu = platform.processor()
        self.os = platform.platform()

        if array is None:
            # we generate the arrays of random numbers
            for i in STANDARD_DIMENSIONS:
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

            self.test_result['quick_sort'][key] = timeit.Timer("sorting.quick_sort("+str(arr)+")", setup= 'import sorting',
                                                            ).autorange()[1]
            self.test_result['merge_sort'][key] = timeit.Timer("sorting.merge("+str(arr)+")", setup= 'import sorting',
                                                            ).autorange()[1]
            # BSTs implementation

            bst_setup = "import bst; tree1 = bst.BinaryTree(); arr_ins = arr[:-1]; for i in arr_ins: bst.insert(i)"
# TODO implement the bst module
            self.test_result['binary_insertion'][key] = timeit.Timer("bst.insert(arr[-1])", setup=bst_setup).autorange()[1]


    def csv(self, name='test'+"right now"): # TODO implement a right now stringer
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
