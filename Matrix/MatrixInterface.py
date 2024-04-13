class MatrixInterface:
    
    ################
    # static methods
    ################

    # take a 2D array and return the equivalent matrix
    @staticmethod
    def from_array(arr):
        pass
    
    # take a matrix and return the equivalent 2D array
    @staticmethod
    def to_array(m):
        pass
    
    # take a matrix and return another matrix with the same values
    @staticmethod
    def copy(m):
        pass

    # apply the function func to all the elements of the matrix m and return the new matrix
    @staticmethod
    def apply_func(m, func):
        pass

    # multiply the scalar with all the elements of the matrix m and return the new matrix
    @staticmethod
    def scalar_product(m, scalar):
        pass

    # add the elements of B to the element at the same position in A and return the new matrix
    @staticmethod
    def element_wise_sum(a, b):
        pass
    
    # substract the elements of B to the element at the same position in A and return the new matrix
    @staticmethod
    def element_wise_sub(a, b):
        pass
    
    # multiply the elements of B with the element at the same position in A (Hadamard product) and return the new matrix
    @staticmethod
    def element_wise_mul(a, b):
        pass

    # return A*B
    @staticmethod
    def matrix_product(a, b):
        pass
    
    # return the transpose matrix of M
    @staticmethod
    def transpose(m):
        pass
    
    ###################
    # matrix operations
    ###################

    # randomize the current matrix
    def randomize(self):
        pass

    # apply the function func to all the elements of the current matrix
    def _apply_func(self, func):
        pass

    # multiply the scalar with all the elements of the current matrix
    def _scalar_product(self, scalar):
        pass
    
    # add the elements of M to the element at the same position in the current matrix
    def _element_wise_sum(self, m):
        pass
    
    # substract the elements of M to the element at the same position in the current matrix
    def _element_wise_sub(self, m):
        pass
    
    # multiply the elements of M with the element at the same position in the current matrix
    def _element_wise_mul(self, m):
        pass