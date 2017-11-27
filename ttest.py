import time
import timeit
import platform
import random

DEFAULT_NUMBER = 100000
DEFAULT_SIZE = 10000
DEFAULT_POPULATION = range(1000000)

class TimeTest(object):
    """returns an object which tests the time efficiency of a multi-dimensional array of increasing sizes of random
    integers over functions of sorting, binary tree insertions and deletions and heap insertions/get/removal.

    There's the possibility to get a summary of the tests results with CPU information, OS infos and plot of
    array size/time of all the executed tests.

    Csv file writer of the tests results for statistical purposes has been implemented too.

    """

    def __init__(self):
        self.test_result = dict(quick_sort={}, merge_sort={})

    def test_it(self, array=False, num=DEFAULT_SIZE, k_size=100):
        """generates a num number of arrays of k-increasing size each of random integers and tests them over
        the given functions. Eventually adds the results to the self.test_result dictionary with template:

        {function: {size_array: time}}

        """
        array_pool = {}
        if not array:
            # we generate the arrays of random numbers
            for i in range(0, num+1, step=k_size):
                array_pool[i] = random.sample(DEFAULT_POPULATION, k=i)



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