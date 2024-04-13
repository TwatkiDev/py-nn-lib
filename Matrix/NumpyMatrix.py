from Matrix.MatrixInterface import MatrixInterface
import numpy as np

class NumpyMatrix(MatrixInterface):
    def __init__(self, rows = 0, columns = 0):
        self.rows = rows
        self.columns = columns
        
        matrix = []
        
        for _ in range(rows):
            row = [0] * columns
            matrix.append(row)
        
        self.matrix = np.array(matrix, dtype="float128")
    
    ################
    # static methods
    ################

    # take a 2D array and return the equivalent matrix
    @staticmethod
    def from_array(arr):
        matrix = np.array(arr, dtype="float128")
        rows, columns = matrix.shape
        ret = NumpyMatrix(rows, columns)
        ret.matrix = matrix
        return ret
    
    # take a matrix and return the equivalent 2D array
    @staticmethod
    def to_array(m):
        if not isinstance(m, NumpyMatrix):
            print("is not a matrix")
            return None
        ret = np.array(m.matrix, dtype="float128")
        return ret
    
    # take a matrix and return another matrix with the same values
    @staticmethod
    def copy(m):
        if not isinstance(m, NumpyMatrix):
            print("is not a matrix")
            return None
        _m = m.matrix.copy()
        rows, columns = _m.shape
        ret = NumpyMatrix(rows, columns)
        ret.matrix = _m
        return ret

    # apply the function func to all the elements of the matrix m and return the new matrix
    @staticmethod
    def apply_func(m, func):
        if not isinstance(m, NumpyMatrix):
            print("is not a matrix")
            return None
        ret = NumpyMatrix.copy(m)
        ret.matrix = np.vectorize(func)(ret.matrix)
        return ret

    # multiply the scalar with all the elements of the matrix m and return the new matrix
    @staticmethod
    def scalar_product(m, scalar):
        if not isinstance(m, NumpyMatrix):
            print("is not a matrix") 
            return None
        ret = NumpyMatrix.copy(m)
        ret.matrix = ret.matrix * scalar
        return ret

    # add the elements of B to the element at the same position in A and return the new matrix
    @staticmethod
    def element_wise_sum(a, b):
        if a.rows != b.rows or a.columns != b.columns:
            print("Matrices dimensions mismatch")
            return None
        ret = NumpyMatrix.copy(a)
        ret.matrix = ret.matrix + b.matrix
        return ret
    
    # substract the elements of B to the element at the same position in A and return the new matrix
    @staticmethod
    def element_wise_sub(a, b):
        if a.rows != b.rows or a.columns != b.columns:
            print("Matrices dimensions mismatch")
            return None
        ret = NumpyMatrix.copy(a)
        ret.matrix = ret.matrix - b.matrix
        return ret
    
    # multiply the elements of B with the element at the same position in A (Hadamard product) and return the new matrix
    @staticmethod
    def element_wise_mul(a, b):
        if a.rows != b.rows or a.columns != b.columns:
            print("Matrices dimensions mismatch")
            return None
        ret = NumpyMatrix.copy(a)
        ret.matrix = np.multiply(a.matrix, b.matrix)
        return ret

    # return A*B
    @staticmethod
    def matrix_product(a, b):
        if a.columns != b.rows:
            print("A.cols ("+str(a.columns)+") doesn't match B.rows ("+str(b.rows)+")")
            return None
        
        ret = NumpyMatrix(a.rows, b.columns)
        ret.matrix = np.dot(a.matrix, b.matrix)
        return ret
    
    # return the transpose matrix of M
    @staticmethod
    def transpose(m):
        ret = NumpyMatrix(m.columns, m.rows)
        ret.matrix = m.matrix.transpose()
        return ret
            
    ######################
    # overridden functions 
    ######################

    def __str__(self):
        ret = "("+str(self.rows)+", "+str(self.columns)+"):\n"
        ret += str(self.matrix)
        return ret
    
    def __eq__(self, o):
        if not isinstance(o, NumpyMatrix):
            return False
        if self.rows != o.rows or self.columns != o.columns:
            return False
        return (self.matrix == o.matrix).all()
    
    ###################
    # matrix operations
    ###################

    # randomize the current matrix
    def randomize(self):
        self.matrix = np.random.rand(self.rows, self.columns)
        self.matrix = self.matrix * 2 - 1

    # apply the function func to all the elements of the current matrix
    def _apply_func(self, func):
        self.matrix = np.vectorize(func)(self.matrix)

    # multiply the scalar with all the elements of the current matrix
    def _scalar_product(self, scalar):
        self.matrix = self.matrix * scalar
    
    # add the elements of M to the element at the same position in the current matrix
    def _element_wise_sum(self, m):
        if self.rows != m.rows or self.columns != m.columns:
            print("Matrices dimensions mismatch")
            return None
        self.matrix = self.matrix + m.matrix
    
    # substract the elements of M to the element at the same position in the current matrix
    def _element_wise_sub(self, m):
        if self.rows != m.rows or self.columns != m.columns:
            print("Matrices dimensions mismatch")
            return None
        self.matrix = self.matrix - m.matrix
    
    # multiply the elements of M with the element at the same position in the current matrix
    def _element_wise_mul(self, m):
        if self.rows != m.rows or self.columns != m.columns:
            print("Matrices dimensions mismatch :/")
            return None
        self.matrix = np.multiply(self.matrix, m.matrix)