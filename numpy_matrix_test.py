import unittest

from Matrix.NumpyMatrix import NumpyMatrix

class TestNumpyMatrix(unittest.TestCase):
    def test_copy(self):
        """
        Test that a matrice and it's copy are equal
        """
        m = NumpyMatrix(3, 4)
        m.randomize()
        _m = NumpyMatrix.copy(m)
        self.assertEqual(m, _m)

    def test_conversion(self):
        """
        Test that the to_array and from_array functions are working correctly
        """
        m = NumpyMatrix(3, 3)
        a = NumpyMatrix.to_array(m)
        _m = NumpyMatrix.from_array(a)
        self.assertEqual(m, _m)
    
    def test_apply_func(self):
        """
        Test if all the elements of the matrix are doubled
        """
        def double(x):
            return 2*x
        
        arr = [[1, 1], [1, 1]]
        m = NumpyMatrix.from_array(arr)
        m1 = NumpyMatrix.apply_func(m, double)
        m2 = NumpyMatrix.copy(m)
        m2._apply_func(double)
        m3 = NumpyMatrix.scalar_product(m, 2)
        m4 = NumpyMatrix.copy(m)
        m4._scalar_product(2)

        ref_arr = [[2, 2], [2, 2]]
        ref_m = NumpyMatrix.from_array(ref_arr)

        self.assertEqual(m1, m2, ref_m)
    
    def test_scalar_product(self):
        """
        Test if all the elements of the matrix are doubled
        """
        
        arr = [[1, 1], [1, 1]]
        m = NumpyMatrix.from_array(arr)
        m1 = NumpyMatrix.scalar_product(m, 2)
        m2 = NumpyMatrix.copy(m)
        m2._scalar_product(2)

        ref_arr = [[2, 2], [2, 2]]
        ref_m = NumpyMatrix.from_array(ref_arr)

        self.assertEqual(m1, m2, ref_m)
    
    def test_element_wise_sum(self):
        """
        Test if all the elements are correctly added together
        """
        
        arr1 = [[1, 1], [1, 1]]
        arr2 = [[2, 3], [4, 5]]
        a = NumpyMatrix.from_array(arr1)
        b = NumpyMatrix.from_array(arr2)

        m1 = NumpyMatrix.element_wise_sum(a, b)
        m2 = NumpyMatrix.copy(a)
        m2._element_wise_sum(b)

        ref_arr = [[3, 4], [5, 6]]
        ref_m = NumpyMatrix.from_array(ref_arr)

        self.assertEqual(m1, m2, ref_m)
    
    def test_element_wise_sub(self):
        """
        Test if all the elements are correctly substracted together
        """
        
        arr1 = [[1, 1], [1, 1]]
        arr2 = [[2, 3], [4, 5]]
        a = NumpyMatrix.from_array(arr1)
        b = NumpyMatrix.from_array(arr2)

        m1 = NumpyMatrix.element_wise_sub(a, b)
        m2 = NumpyMatrix.copy(a)
        m2._element_wise_sub(b)

        ref_arr = [[-1, -2], [-3, -4]]
        ref_m = NumpyMatrix.from_array(ref_arr)

        self.assertEqual(m1, m2, ref_m)
    
    def test_element_wise_mul(self):
        """
        Test if all the elements are correctly multiplied together
        """
        
        arr1 = [[2, 3], [4, 5]]
        arr2 = [[2, 3], [4, 5]]
        a = NumpyMatrix.from_array(arr1)
        b = NumpyMatrix.from_array(arr2)

        m1 = NumpyMatrix.element_wise_mul(a, b)
        m2 = NumpyMatrix.copy(a)
        m2._element_wise_mul(b)

        ref_arr = [[4, 9], [16, 25]]
        ref_m = NumpyMatrix.from_array(ref_arr)

        self.assertEqual(m1, m2, ref_m)
    
    def test_transpose(self):
        """
        Test if the two matrices are multiplied correctly
        """
        
        arr = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        m = NumpyMatrix.from_array(arr)
        m_T = NumpyMatrix.transpose(m)

        ref_arr = [[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]]
        ref_m = NumpyMatrix.from_array(ref_arr)

        self.assertEqual(m_T, ref_m)
        
if __name__ == '__main__':
    unittest.main()