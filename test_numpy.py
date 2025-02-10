import unittest
import numpy as np
from your_module import (rand_int, rand_float, first_10_primes, squares, cubes, 
                         add_arrays, subtract_arrays, multiply_arrays, divide_arrays, 
                         stats, first_5, last_3, indices_2_4_6, greater_50, less_7, 
                         reshape_2x6, reshape_2x3x4, reshape_10x10, reshape_10x10x10, 
                         reshape_10x10x10x10, add_broadcast, subtract_broadcast, 
                         multiply_broadcast, divide_broadcast, element_wise_product, 
                         temp_data, rainfall_data, image_thresholding, matrix_diagonals)

class TestNumpyExercises(unittest.TestCase):

    def test_rand_int(self):
        np.random.seed(10)
        result = rand_int()
        expected = np.array([ 9, 15, 64, 28, 89, 93, 29,  8, 73,  0])
        np.testing.assert_array_equal(result, expected)

    def test_rand_float(self):
        np.random.seed(10)
        result = rand_float()
        expected = np.array([0.77132064, 0.02075195, 0.63364823, 0.74880388, 0.49850701])
        np.testing.assert_array_equal(result, expected)

    def test_first_10_primes(self):
        result = first_10_primes()
        expected = np.array([ 2,  3,  5,  7, 11, 13, 17, 19, 23, 29])
        np.testing.assert_array_equal(result, expected)

    def test_squares(self):
        result = squares()
        expected = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
        np.testing.assert_array_equal(result, expected)

    def test_cubes(self):
        result = cubes()
        expected = np.array([1, 8, 27, 64, 125, 216, 343, 512, 729, 1000])
        np.testing.assert_array_equal(result, expected)

    def test_add_arrays(self):
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        result = add_arrays(arr1, arr2)
        expected = np.array([5, 7, 9])
        np.testing.assert_array_equal(result, expected)

    def test_subtract_arrays(self):
        arr1 = np.array([4, 5, 6])
        arr2 = np.array([1, 2, 3])
        result = subtract_arrays(arr1, arr2)
        expected = np.array([3, 3, 3])
        np.testing.assert_array_equal(result, expected)

    def test_multiply_arrays(self):
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        result = multiply_arrays(arr1, arr2)
        expected = np.array([4, 10, 18])
        np.testing.assert_array_equal(result, expected)

    def test_divide_arrays(self):
        arr1 = np.array([4, 5, 6])
        arr2 = np.array([2, 2, 2])
        result = divide_arrays(arr1, arr2)
        expected = np.array([2, 2.5, 3])
        np.testing.assert_array_equal(result, expected)

    def test_stats(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = stats(arr)
        expected = (3.0, 3.0, 1.4142135623730951)
        self.assertEqual(result, expected)

    def test_first_5(self):
        arr = np.random.randint(0, 100, 10)
        result = first_5(arr)
        expected = arr[:5]
        np.testing.assert_array_equal(result, expected)

    def test_last_3(self):
        arr = np.random.randint(0, 100, 10)
        result = last_3(arr)
        expected = arr[-3:]
        np.testing.assert_array_equal(result, expected)

    def test_indices_2_4_6(self):
        arr = np.random.randint(0, 100, 10)
        result = indices_2_4_6(arr)
        expected = arr[[2, 4, 6]]
        np.testing.assert_array_equal(result, expected)

    def test_greater_50(self):
        arr = np.random.randint(0, 100, 10)
        result = greater_50(arr)
        expected = arr[arr > 50]
        np.testing.assert_array_equal(result, expected)

    def test_less_7(self):
        arr = np.random.randint(0, 10, 10)
        result = less_7(arr)
        expected = arr[arr <= 7]
        np.testing.assert_array_equal(result, expected)

    def test_reshape_2x6(self):
        arr = np.arange(12)
        result = reshape_2x6(arr)
        expected = arr.reshape(2, 6)
        np.testing.assert_array_equal(result, expected)

    def test_reshape_2x3x4(self):
        arr = np.arange(24)
        result = reshape_2x3x4(arr)
        expected = arr.reshape(2, 3, 4)
        np.testing.assert_array_equal(result, expected)

    def test_reshape_10x10(self):
        arr = np.arange(100)
        result = reshape_10x10(arr)
        expected = arr.reshape(10, 10)
        np.testing.assert_array_equal(result, expected)

    def test_reshape_10x10x10(self):
        arr = np.arange(1000)
        result = reshape_10x10x10(arr)
        expected = arr.reshape(10, 10, 10)
        np.testing.assert_array_equal(result, expected)

    def test_reshape_10x10x10x10(self):
        arr = np.arange(10000)
        result = reshape_10x10x10x10(arr)
        expected = arr.reshape(10, 10, 10, 10)
        np.testing.assert_array_equal(result, expected)

    def test_add_broadcast(self):
        arr1 = np.array([[1, 2, 3], [4, 5, 6]])
        arr2 = np.array([[1], [2]])
        result = add_broadcast(arr1, arr2)
        expected = arr1 + arr2
        np.testing.assert_array_equal(result, expected)

    def test_subtract_broadcast(self):
        arr1 = np.array([[1, 2], [3, 4], [5, 6]])
        arr2 = np.array([[1, 2, 3], [4, 5, 6]])
        result = subtract_broadcast(arr1, arr2)
        expected = arr1 - arr2.T
        np.testing.assert_array_equal(result, expected)

    def test_multiply_broadcast(self):
        arr1 = np.array([[1, 2, 3], [4, 5, 6]])
        arr2 = np.array([[1, 2], [3, 4], [5, 6]])
        result = multiply_broadcast(arr1, arr2)
        expected = arr1 @ arr2
        np.testing.assert_array_equal(result, expected)

    def test_divide_broadcast(self):
        arr1 = np.array([[1, 2, 3], [4, 5, 6]])
        arr2 = np.array([[1], [2]])
        result = divide_broadcast(arr1, arr2)
        expected = arr1 / arr2
        np.testing.assert_array_equal(result, expected)

    def test_element_wise_product(self):
        arr1 = np.array([[1, 2, 3], [4, 5, 6]])
        arr2 = np.array([[1, 2, 3], [4, 5, 6]])
        result = element_wise_product(arr1, arr2)
        expected = arr1 * arr2
        np.testing.assert_array_equal(result, expected)

    def test_temp_data(self):
        temps = np.array([10, 20, 30, 25, 15])
        mask = temps > 25
        result = temps[mask]
        expected = np.array([30])
        np.testing.assert_array_equal(result, expected)
        below_15 = np.sum(temps < 15)
        self.assertEqual(below_15, 1)

    def test_rainfall_data(self):
        rainfall = np.array([[50, 60, 110], [90, 100, 120], [80, 70, 130]])
        mask = rainfall > 100
        result = np.where(mask)[0]
        expected = np.array([0, 1, 2])
        np.testing.assert_array_equal(result, expected)

    def test_image_thresholding(self):
        image = np.array([[100, 150], [200, 50]])
        threshold = 128
        result = np.where(image > threshold, 255, 0)
        expected = np.array([[0, 255], [255, 0]])
        np.testing.assert_array_equal(result, expected)