import numpy as np

# вектор истинных зачений b, причём b0 - свободный коэффициент
b_vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
# вектор мат. ожиданий и дисперсий для каждого значения входного сигнала
u_params = np.array([[0.0, 0.1], [0.0, 0.1], [0.0, 0.1], [0.0, 0.1]])
# вектор мат. ожиданий и дисперсий для каждого значения шума
eta_params = np.array([[0.0, 0.5]])
# количество экспериментов
experiments_count = 10000


def create_Y_object(U_matr, b_vec, eta_vec):
    return np.matrix((create_Y_model(U_matr, b_vec) + eta_vec).reshape((U_matr.shape[0], 1)))


def create_Y_model(U_matr, b_vec):
    return np.matrix((U_matr @ b_vec[1:].T + b_vec[0]).reshape((U_matr.shape[0], 1)))


def cp_calc(U_matr, b_vec, eta_vec):
    y = create_Y_object(U_matr, b_vec, eta_vec)
    N = y.shape[0]

    U_cp = np.array([])
    y_cp = sum(y) / N
    for i in range(U_matr.shape[1]):
        U_cp = np.append(U_cp, sum(U_matr[:, i]) / N)
    return [y_cp, np.matrix(U_cp)]


def centrificate(U_matr, b_vec, eta_vec):
    [y_cp, U_cp] = cp_calc(U, b_vector, eta)
    y = create_Y_object(U_matr, b_vec, eta_vec)

    y_0 = y - y_cp
    U_0 = U_matr - U_cp

    return [y_0, U_0]


def create_U(u_par, count):
    res = np.array([])
    for k in range(count):
        tmp = []
        for i in range(len(u_par)):
            tmp = np.append(tmp, np.random.normal(u_par[i][0], u_par[i][1], 1))
        res = np.append(res, tmp)
    return np.matrix(res.reshape(count, len(u_par)))


def create_eta(eta_par, count):
    res = np.array([])
    for k in range(count):
        tmp = []
        for i in range(len(eta_par)):
            tmp = np.append(tmp, np.random.normal(eta_par[i][0], eta_par[i][1], 1))
        res = np.append(res, tmp)
    return np.matrix(res.reshape(count, len(eta_par)))


def solver(U_matr, y_vec):
    b_tmp = np.linalg.inv(U_matr.T @ U_matr) @ U_matr.T @ y_vec

    [y_cp, U_cp] = cp_calc(U, b_vector, eta)
    b_0 = y_cp - U_cp * b_tmp

    return np.matrix(np.vstack((b_0, b_tmp)))


def full_solver(u_par, eta_par, b_vec, count):
    global U
    global eta

    U = create_U(u_par, count)
    eta = create_eta(eta_par, count)
    [y_0, U_0] = centrificate(U, b_vec, eta)
    return solver(U_0, y_0)


print(full_solver(u_params, eta_params, b_vector, experiments_count), "\n")
