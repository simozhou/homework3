import time
import sorting
import bst
import timeit
import platform
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import heapq
import copy

random.seed(521)
DEFAULT_NUMBER = 100000  # 100k
DEFAULT_POPULATION = range(100000)  # 1m
DEFAULT_SIZES = [10, 100, 1000, 10000, 100000]


def _statistify(array):
    """returns mean and confidence interval of a given array of numbers (assuming normality of the sampling
    population) """
    mean, se = np.mean(array), 2 * (np.std(array) / np.sqrt(len(array)))
    return mean, se


class TimeTest(object):
    """returns an object which tests the time efficiency of a multi-dimensional array of increasing sizes of random
    integers over functions of sorting, binary tree insertions and deletions and heap insertions/get/removal.

    There's the possibility to get a summary of the tests results with CPU information, OS infos and plot of
    array size/time of all the executed tests.

    Csv file writer of the tests results for statistical purposes has been implemented too.

    """

    def __init__(self, array=None, max_val=DEFAULT_NUMBER):
        # this is the dict of dicts where the values obtained from the test_it func will be stored.
        self.test_result = dict(quick_sort={}, merge_sort={}, binary_insertion={},
                                binary_insertion_se={}, binary_get_random={}, binary_get_random_se={},
                                binary_delete={}, binary_delete_se={}, binary_get_max={}, binary_get_max_se={},
                                heap_insert={}, heap_insert_se={}, heap_get_max={}, heap_get_max_se={},
                                heap_remove_se={}, heap_remove={})
        # dict of all the list to use for testing
        self.array_pool = {}
        # both for summary() purrrrrposes
        self.cpu = platform.processor()
        self.os = platform.platform()
        self.hashcode = random._sha512().hexdigest()[0:10]

        if array is None:
            # we generate the arrays of random numbers with a logarithmic distance one with the other
            for i in np.logspace(1.0, np.log10(max_val, dtype=float), base=10.0, endpoint=True, dtype=int):
                self.array_pool[i] = random.sample(DEFAULT_POPULATION, k=i)
        # to use a more accurate logarithmic scale, to improve plotting and statistics quality
        elif array is "e_log":
            for i in DEFAULT_SIZES:
                self.array_pool[i] = random.sample(DEFAULT_POPULATION, k=i)
        # optional
        # else:
        #     for lst in array:
        #         self.array_pool[len(lst)] = lst
        print("time test generated!")

    def test_it(self):
        """generates a number of arrays of increasing size each of random integers and tests them over
        the given functions. Eventually adds the results to the self.test_result dictionary with template:

        {function: {size_array: time_mean}, function_se: {size_array: time_confint}}

        """
        for key, arr in self.array_pool.items():
            # sorting algorithms

            self._test_it_quick_sort(key, arr)
            self._test_it_merge_sort(key, arr)
        print("sorting timing done!")

        # BSTs implementation

        insertion_counter = 0
        global tree
        tree = bst.BinarySearchTree()
        for bst_key in self.array_pool[DEFAULT_NUMBER]:
            insertion_counter += 1
            if insertion_counter in self.array_pool.keys():
                self._test_it_binary_get_max(insertion_counter)
                self._test_it_binary_get_random(insertion_counter)
                # testing over 30 randomnumber AND NOT CHANGING THE TREE
                self._test_it_binary_insertion_deletion(insertion_counter,
                                                        random.sample(self.array_pool[DEFAULT_NUMBER], k=50))
                tree.put(bst_key, 0)
            else:
                # KEY, VALUE
                tree.put(bst_key, 0)
        print("binary insertion, get random, get max and deletion timing done!")

        # heap implementation
        global heaper
        heaper = []
        heap_counter = 0
        for heap_key in self.array_pool[DEFAULT_NUMBER]:
            heap_counter += 1
            if heap_counter in self.array_pool.keys():
                # they will add values to heap_temp
                self._test_it_heap_get_max(heap_counter)
                # NOT CHANGING THE HEAP
                self._test_it_heap_insert_delete(heap_counter, random.sample(self.array_pool[DEFAULT_NUMBER], k=50))
                heapq.heappush(heaper, heap_key)
            else:
                heapq.heappush(heaper, heap_key)
        print("heap insertion, get max and deletion timing done!")

        self._pandator()

    def csv(self, name='common'):
        """generates a csv file with the results of the test_it function, returns a "Run test_it before requesting
        csv report" if the self.test_result field has not been populated already.

        """
        if name is 'common':
            name = 'benchmark_analysis_' + self.hashcode + ".csv"
        with open(name, 'w') as f:
            f.write(self.test_result.to_csv())

        print("csv file written! check current directory")

    def summary(self, pdf_report=True):
        """prints out a summary with CPU, OS infos, plots of all the tests [array_size/time]
        and POSSIBLY a pdf markdown with all of it in a beautiful graphics.

        Returns a "Run test_it before requesting a  test summary" if the self.test_result
        field has not been populated already.


        """
        # TODO needs fix with errorscatter
        plt.figure(figsize=(8, 11))
        plt.suptitle("Benchmark analysis summary", fontsize=24)
        plt.subplots_adjust(top=0.88)
        # sorting repr
        plt.subplot(321)
        plt.tight_layout()
        plt.title("Quick sort")
        plt.grid()
        plt.xscale('log')
        plt.plot(self.test_result.quick_sort, marker='.', ms=0.99)
        plt.ylabel("time")
        plt.xlabel("$\log(size)$")
        plt.subplot(322)
        plt.tight_layout()
        plt.grid()
        plt.title("Merge Sort")
        plt.plot(self.test_result.merge_sort, marker='.', ms=0.99)
        plt.ylabel("time")
        plt.xlabel("$\log(size)$")
        plt.xscale('log')
        # binary tree repr
        plt.subplot(323)
        plt.tight_layout()
        plt.grid()
        plt.errorbar(self.test_result.binary_delete.index, self.test_result.binary_delete,
                     self.test_result.binary_delete_se, label="Delete", ecolor='orange',
                     fmt='b-o', ms=0.99, capsize=2)
        plt.errorbar(self.test_result.binary_insertion.index, self.test_result.binary_insertion,
                     self.test_result.binary_insertion_se, label="Insert",
                     ecolor='green', fmt='r-o', ms=0.99, capsize=2)
        plt.yscale('log')
        plt.ylabel("$\log(time)$")
        plt.xlabel("$\log(size)$")
        plt.xscale('log')
        plt.title("Binary insert and binary delete")
        plt.legend()
        plt.subplot(324)
        plt.tight_layout()
        plt.grid()
        plt.errorbar(self.test_result.binary_get_max.index, self.test_result.binary_get_max,
                     self.test_result.binary_get_max_se, label="Get max", ecolor='green', fmt='r-o', ms=0.99, capsize=2)
        plt.errorbar(self.test_result.binary_get_random.index, self.test_result.binary_get_random,
                     self.test_result.binary_get_random_se, label="Get random", ecolor='orange', fmt='b-o', ms=0.99,
                     capsize=2)
        plt.ylabel("$\log(time)$")
        plt.xlabel("$\log(size)$")
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.title("Binary get max and binary get random")
        # heap repr
        plt.subplot(325)
        plt.tight_layout()
        plt.grid()
        plt.errorbar(self.test_result.heap_insert.index, self.test_result.heap_insert, self.test_result.heap_insert_se,
                     label="Insertion", ecolor='green', fmt='r-o', ms=0.99, capsize=2)
        plt.errorbar(self.test_result.heap_remove.index, self.test_result.heap_remove, self.test_result.heap_remove_se,
                     label="Deletion", ecolor='orange', fmt='b-o', ms=0.99, capsize=2)
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel("$\log(time)$")
        plt.xlabel("$\log(size)$")
        plt.legend()
        plt.title("Heap insertion, deletion")
        plt.subplot(326)
        plt.tight_layout()
        plt.errorbar(self.test_result.index, self.test_result.heap_get_max, self.test_result.heap_get_max_se,
                     label="Get max", ecolor='red', ms=0.99, capsize=3)
        plt.title("Heap get max")
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel("$\log(time)$")
        plt.xlabel("$\log(size)$")
        plt.subplots_adjust(top=0.88)
        # text with infos
        self.print_info()

        if pdf_report:
            namefile = "benchmark_analysis_" + self.hashcode + ".pdf"
            plt.savefig(fname=namefile, papertype='a4', orientation='portrait')
        plt.show()

    # sorting impl
    def _test_it_quick_sort(self, key, arr):
        self.test_result['quick_sort'][key] = timeit.timeit("sorting.quick_sort(" + str(arr) + ")", globals=globals(),
                                                            number=10)

    def _test_it_merge_sort(self, key, arr):
        self.test_result['merge_sort'][key] = timeit.timeit("sorting.merge(" + str(arr) + ")", globals=globals(),
                                                            number=10)

    # bst impl
    def _test_it_binary_insertion_deletion(self, key, array):
        results_ins, results_del = [], []
        for value in array:
            results_ins.append(timeit.timeit("tree.put(" + str(value) + ",0)", globals=globals(), number=1))
            results_del.append(timeit.timeit("tree.delete(" + str(value) + ")", globals=globals(), number=1))
        self.test_result['binary_insertion'][key], self.test_result['binary_insertion_se'][key] = _statistify(
            results_ins)
        self.test_result["binary_delete"][key], self.test_result["binary_delete_se"][key] = _statistify(results_del)

    def _test_it_binary_get_max(self, key):
        results = []
        for _ in range(50):
            results.append(timeit.timeit("tree.findMax()", globals=globals(), number=10))
        self.test_result['binary_get_max'][key], self.test_result['binary_get_max_se'][key] = _statistify(results)

    def _test_it_binary_get_random(self, key):
        results = []
        for _ in range(50):
            rand_key = random.randint(0, len(tree))
            results.append(timeit.timeit("tree.get(" + str(rand_key) + ")", globals=globals(), number=100))
        self.test_result['binary_get_random'][key], self.test_result['binary_get_random_se'][key] = _statistify(results)

    # heap impl
    def _test_it_heap_insert_delete(self, key, array):
        results_ins, results_del = [], []
        setupline = 'copy_heap = copy.deepcopy(heaper)'
        for value in array:
            results_ins.append(
                timeit.timeit('heapq.heappush(copy_heap,' + str(value) + ')', number=1, globals=globals(),
                              setup=setupline))
            results_del.append(timeit.timeit('heapq.heappop(copy_heap)', number=1, globals=globals(), setup=setupline))

        self.test_result['heap_insert'][key], self.test_result['heap_insert_se'][key] = _statistify(results_ins)
        self.test_result['heap_remove'][key], self.test_result['heap_remove_se'][key] = _statistify(results_del)

    # TODO repair this function
    def _test_it_heap_get_max(self, key):
        results = []
        for _ in range(30):
            results.append(timeit.timeit('heaper[0]', number=10, globals=globals()))
        self.test_result['heap_get_max'][key], self.test_result['heap_get_max_se'][key] = _statistify(results)

    # deprecated
    def _test_it_heap_del_max(self, key):
        self.test_result['heap_remove'][key] = timeit.timeit('heapq.heappop(heaper)', number=1, globals=globals())

    # TODO fix this function
    def _pandator(self):
        """transform the collected results from dict to pandas DataFrames, merging them together into a unique Df"""
        # noinspection PyTypeChecker
        self.test_result = pd.DataFrame.from_dict(self.test_result)

    @staticmethod
    def print_info():
        print('\nPython version  :', platform.python_version())
        print('compiler        :', platform.python_compiler())

        print('\nsystem     :', platform.system())
        print('release    :', platform.release())
        print('machine    :', platform.machine())
        print('processor  :', platform.processor())
        print('interpreter:', platform.architecture()[0])
        print('\n\n')


if __name__ == '__main__':
    a = TimeTest('e_log')
    t = time.time()
    a.test_it()
    t = time.time() - t
    print(t)
    a.summary(pdf_report=False)
