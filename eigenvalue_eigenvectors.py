import numpy as np
import scipy.linalg 

def Power_method(matrix, eps = 1e-4):
    """
    This function finds max eigenvalue and eigenvector by using power method.

    :param matrix: given matrix
    :param eps: accuracy of computation

    :return: cortege of eigenvalue and eigenvector
    """
    k, n = 1, len(matrix)
    x = 1
    Xn = np.random.random(n) #eigenvector
    yn = matrix @ Xn
    l_n = yn / Xn #lambda
    Xn = yn / np.max(yn) #normalization
    dl = np.copy(l_n) #delta lamda: lambda[k + 1] - lambda[k]

    with open('Result_1(6).txt', 'a') as file:

        file.write('------------------------------POWER METHOD------------------------------\n')
        file.write('\nGiven matrix =\n' + str(matrix) + '\n')
        while all(abs(i) > eps for i in dl):

            k+=1
            l_o = np.copy(l_n)
            yn = matrix @ Xn
            l_n = yn / Xn
            Xn = yn / np.max(yn)
            dl = np.array([abs(l_n[i] - l_o[i]) for i in range(n)])

            file.write('\ny[' + str(k) + '] =' + str(yn) + '\n')
            file.write('lambda[' + str(k) + '] =' + str(l_n) + '\n')
            file.write('X[' + str(k) + '] =' + str(Xn) + '\n')
            
            x = matrix @ Xn.transpose() - np.mean(l_n) * Xn.transpose()


        file.write('\nAnswer:\neigenvalue:' + str(np.mean(l_n)))
        file.write('\neigenvector:\n' + str(Xn))

        file.write('\n\nResidual vector:\n')
        file.write('Ax - lx = ' + str(x) + '\n')

    return np.mean(l_n), Xn

def Jacobi_method(matrix, eps = 1e-4):
    """
    This function solves full problem of eigenvalues
    
    :param matrix: given matrix
    :param eps: accuracy of computation

    :return: eigenvalues and eigenvectors
    """

    temp , count = 1, 0
    check = np.zeros((len(matrix), len(matrix[1])))
    Eigenvectors = np.eye(5)
    A = np.copy(matrix)

    with open('Result_1(6).txt', 'a') as file:
        file.write('\n------------------------------Jacobi Method------------------------------')
        while temp > eps:
            count += 1

            file.write('\n\nIteration: ' + str(count))
            file.write('\nA =\n' + str(matrix))
            
            mask = np.ones(matrix.shape, dtype=bool)
            np.fill_diagonal(mask, 0)
            key_element = max( matrix[mask], key = abs) #find the biggest value
            a = np.argwhere(matrix == key_element) #find indices with the biggest value
            
            file.write('\n\nIndices with max absolute value: i = ' + str(a[0, 0]) + ' j = ' + str(a[0, 1]))

            t = (matrix[a[0, 0], a[0, 0]] - matrix[a[0, 1], a[0, 1]]) / (2 * key_element)
            if t == 0:
                c = s = 1 / np.sqrt(2)
            else:
                tau = np.sqrt(1 + t ** 2)
                if t > 0:
                    t = -t + tau
                else: 
                    t = -t - tau

                c = 1 / np.sqrt(1 + t ** 2)
                s = c * t 
            
            file.write('\nAngular parameters: c = ' + str(c) + ' s = ' + str(s))
            file.write('\nCheck: c^2 + s^2 == ' + str(c ** 2 + s** 2))

            Temp = np.array([i ** 2 for i in matrix])
            omega_2 = np.sum(Temp) - np.trace(Temp)
            check_o = np.sum(check) - np.trace(check)
            check = np.copy(Temp)

            file.write('\nDelta = ' + str(np.trace(Temp)))
            file.write('\n2 * Omega = ' + str(omega_2))
            if count != 1:
                file.write('\nCheck delta = ' + str(np.trace(Temp) - np.trace(check)))
                file.write('\nCheck 2 * Omega = ' + str(omega_2 - check_o))
            else:
                file.write('\nCheck delta = 0')
                file.write('\nCheck 2 * Omega = 0')
            file.write('\nDelta + 2 * Omega = ' + str(np.sum(Temp)))
            
            T = np.eye(matrix.shape[1])
            T[a[0, 0], a[0, 0]] = c
            T[a[0, 1], a[0, 1]] = c
            T[a[0, 0], a[0, 1]] = -s
            T[a[0, 1], a[0, 0]] = s

            B = T.transpose() @ (matrix @ T)
            matrix = np.copy(B)

            Eigenvectors = Eigenvectors.dot(T)

            Temp = np.array([i ** 2 for i in matrix])
            temp = np.sum(Temp) - np.trace(Temp)

        file.write('\n\nAnswer:\nEigenvalues:' + str(np.diag(matrix)))
        file.write('\nEigenvectors:\n' + str(Eigenvectors.transpose()))
        
        file.write('\n\nResidual vectors:\n')
        for j in range(len(matrix)):
            x = A @ Eigenvectors.transpose()[j] - matrix[j, j] * Eigenvectors.transpose()[j]
            file.write('x[' + str(np.diag(matrix)[j]) + '] = ' + str(x) + '\n')
    
    return np.diag(matrix), Eigenvectors.transpose()

def LU_decomposition(matrix, eps = 1e-4):
    """
    This function solves full problem of eigenvectors and eigenvalues using LU-decomposition

    :param matrix: given matrix
    :param eps: accuracy of computation

    :return: eigenvalues and eigenvectors
    """
    A = np.copy(matrix)
    LL = UU = np.eye(len(matrix))
    count = 0
    x, y = np.linalg.eig(matrix)

    with open('Result_1(6).txt', 'a') as file:
        file.write('\n------------------------------LU-algorithm------------------------------\n')
        file.write('Given matrix = \n' + str(matrix) + '\n')

        while abs(np.linalg.det(matrix) - np.prod(np.diag(matrix)))> eps:
            P, L, U = scipy.linalg.lu(matrix)
            L = P @ L
            matrix = U @ L
            LL = LL @ L
            UU = np.linalg.inv(L) @ UU
            
            count += 1
            if count == 1 or count % 5 == 0:
                file.write('\nInteration ' + str(count))
                file.write('\nL = \n' + str(L) + '\n')
                file.write('U =\n' + str(U) + '\n')
        matrix = np.sort(np.diag(matrix))
        matrix = matrix[::-1]

        file.write('\n\nAnswer:\nEigenvalues: ' + str(matrix))
        file.write('\nEigenvectors:\n' + str(y.transpose()))

        file.write('\n\nResidual vectors\n')
        for j in range(len(matrix)):
            x = A @ y.transpose()[j] - matrix[j] * y.transpose()[j]
            file.write('x[' + str(matrix[j]) + '] = ' + str(x) + '\n')

    return matrix, y.transpose()

matrix_1 = np.loadtxt('matrix_1.txt', 'f')
matrix_2 = np.loadtxt('matrix_2.txt', 'f')

eigenvalue_p, eigenvector_p = Power_method(matrix_1)
print('------------------------------Power method------------------------------\nEigenvalue:\n', eigenvalue_p)
print('Eigenvector: \n', eigenvector_p)

eigenvalues_j, eigenvectors_j = Jacobi_method(matrix_1)
print('\n------------------------------Jacobi method------------------------------\nEigenvalues: \n', eigenvalues_j)
print('Eigenvectors:\n', eigenvectors_j)

eigenvalues_lu, eigenvectors_lu = LU_decomposition(matrix_2)
print('\n------------------------------LU algorithm------------------------------\nEigenvalues: \n', eigenvalues_lu)
print('Eigenvectors: \n', eigenvectors_lu)