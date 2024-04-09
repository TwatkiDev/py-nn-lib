from random import uniform as rand

class Matrix:
    def __init__(self, rows = 0, columns = 0):
        self.rows = rows
        self.columns = columns
        
        self.matrix = []
        
        for _ in range(rows*columns):
            self.matrix.append(0)
    
    ################
    # static methods
    ################

    # take a 2D array and return the equivalent matrix
    @staticmethod
    def from_array(arr):
        rows = len(arr)
        columns = len(arr[0])
        ret = Matrix(rows, columns)
        ret.matrix = []
        for i in range(rows):
            for j in range(columns):
                ret.matrix.append(arr[i][j])
        return ret
    
    # take a matrix and return the equivalent 2D array
    @staticmethod
    def to_array(m):
        ret = []
        for i in range(m.rows):
            row = []
            for j in range(m.columns):
                row.append(m.matrix[i*m.columns+j])
            ret.append(row)
        return ret
    
    # take a matrix and return another matrix with the same values
    @staticmethod
    def copy(m):
        if not isinstance(m, Matrix):
            print("is not a matrix")
            return None
        _m = Matrix(m.rows, m.columns)
        for i in range(m.rows*m.columns):
            _m.matrix[i] = m.matrix[i]
        return _m

    # apply the function func to all the elements of the matrix m and return the new matrix
    @staticmethod
    def apply_func(m, func):
        if not isinstance(m, Matrix):
            print("is not a matrix")
            return None
        ret = Matrix.copy(m)
        for idx in range(len(m.matrix)):
            ret.matrix[idx] = func(m.matrix[idx])
        return ret

    # multiply the scalar with all the elements of the matrix m and return the new matrix
    @staticmethod
    def scalar_product(m, scalar):
        if not isinstance(m, Matrix):
            print("is not a matrix") 
            return None
        ret = Matrix.copy(m)
        for idx in range(len(ret.matrix)):
            ret.matrix[idx] *= scalar
        return ret

    # add the elements of B to the element at the same position in A and return the new matrix
    @staticmethod
    def element_wise_sum(a, b):
        if a.rows != b.rows or a.columns != b.columns:
            print("Matrices dimensions mismatch")
            return None
        ret = Matrix.copy(a)
        for idx in range(len(ret.matrix)):
            ret.matrix[idx] += b.matrix[idx]
        return ret
    
    # substract the elements of B to the element at the same position in A and return the new matrix
    @staticmethod
    def element_wise_sub(a, b):
        if a.rows != b.rows or a.columns != b.columns:
            print("Matrices dimensions mismatch")
            return None
        ret = Matrix.copy(a)
        for idx in range(len(ret.matrix)):
            ret.matrix[idx] -= b.matrix[idx]
        return ret
    
    # multiply the elements of B with the element at the same position in A (Hadamard product) and return the new matrix
    @staticmethod
    def element_wise_mul(a, b):
        if a.rows != b.rows or a.columns != b.columns:
            print("Matrices dimensions mismatch")
            return None
        ret = Matrix.copy(a)
        for idx in range(len(ret.matrix)):
            ret.matrix[idx] *= b.matrix[idx]
        return ret

    # return A*B
    @staticmethod
    def matrix_product(a, b):
        if a.columns != b.rows:
            print("A.cols ("+str(a.columns)+") doesn't match B.rows ("+str(b.rows)+")")
            return None
        
        ret = Matrix(a.rows, b.columns)
        
        for ret_idx in range(a.rows * b.columns):
            ret.matrix[ret_idx] = 0
            for i in range(a.columns):
                m1_idx = int(ret_idx / b.columns) * a.columns + i
                m2_idx = ret_idx % b.columns + i * b.columns
                ret.matrix[ret_idx] += a.matrix[m1_idx] * b.matrix[m2_idx]
        
        return ret
    
    # return the transpose matrix of M
    @staticmethod
    def transpose(m):
        ret = Matrix(m.columns, m.rows)
        for idx in range(len(m.matrix)):
            r = int(idx / m.columns)
            c = idx % m.columns
            _idx = c * ret.columns + r 
            ret.matrix[_idx] = m.matrix[idx]
        return ret
            
    ######################
    # overridden functions 
    ######################

    def __str__(self):
        ret = "("+str(self.rows)+", "+str(self.columns)+"):\n[\n"
        for i in range(self.rows):
            ret += str(self.matrix[i*self.columns:(i+1)*self.columns])+"\n"
        ret += "]"
        return ret
    
    def __eq__(self, o):
        if not isinstance(o, Matrix):
            return False
        if self.rows != o.rows or self.columns != o.columns:
            return False
        for i in range(len(self.matrix)):
            if self.matrix[i] != o.matrix[i]:
                return False
        return True
    
    ###################
    # matrix operations
    ###################

    # randomize the current matrix
    def randomize(self):
        for idx in range(len(self.matrix)):
            self.matrix[idx] = rand(-1, 1)

    # apply the function func to all the elements of the current matrix
    def _apply_func(self, func):
        for idx in range(len(self.matrix)):
            self.matrix[idx] = func(self.matrix[idx])

    # multiply the scalar with all the elements of the current matrix
    def _scalar_product(self, scalar):
        for idx in range(len(self.matrix)):
            self.matrix[idx] *= scalar
    
    # add the elements of M to the element at the same position in the current matrix
    def _element_wise_sum(self, m):
        if self.rows != m.rows or self.columns != m.columns:
            print("Matrices dimensions mismatch")
            return None
        for idx in range(len(self.matrix)):
            self.matrix[idx] += m.matrix[idx]
    
    # substract the elements of M to the element at the same position in the current matrix
    def _element_wise_sub(self, m):
        if self.rows != m.rows or self.columns != m.columns:
            print("Matrices dimensions mismatch")
            return None
        for idx in range(len(self.matrix)):
            self.matrix[idx] -= m.matrix[idx]
    
    # multiply the elements of M with the element at the same position in the current matrix
    def _element_wise_mul(self, m):
        if self.rows != m.rows or self.columns != m.columns:
            print("Matrices dimensions mismatch")
            return None
        for idx in range(len(self.matrix)):
            self.matrix[idx] *= m.matrix[idx]