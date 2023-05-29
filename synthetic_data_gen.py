import numpy as np
from functools import reduce
import argparse

def generate_GaussianMixture(n_components, d, c_separation):
    # generate pi
    # generate means
    # generate Sigma
    components_params = []
    i = 0
    weights = np.random.uniform(2, 4, size=n_components)
    weights = weights  / sum(weights)
    while True:
        print(i)
        params = generateGaussian(d, means_max=20)

        accept = reduce(lambda a, params_1: a and check_separation(params, params_1, c_separation), components_params, True)

        if accept:
            components_params.append(params)
        
        i += 1

        if len(components_params) == n_components:
            return weights, components_params

def check_separation(params_1, params_2, c_separation):

    delta_means_norm = np.linalg.norm(params_1['mu'] - params_2['mu'], ord=2)
    d = params_1['mu'].shape[0]

    return delta_means_norm > c_separation * np.sqrt(d * max(params_1['lambda_max'], params_2['lambda_max']))

def generateGaussian(d, means_max=100):
    means = np.random.uniform(0, means_max, size=d)
    sigma, larges_eigenval = generateCovMatr_and_larget_eigenval(d)
    return {'mu': means, 'sigma': sigma, 'lambda_max': larges_eigenval}

def createCovMatr_from_params(eigenvalues, rotation_angles):
    d = len(eigenvalues)
    i = 0
    V = np.eye(d)
    for p in range(d - 1):
        for q in range(p + 1, d):
            G_pq = createGinenvRotation(rotation_angles[i], p, q, d)
            V = V @ G_pq
            i += 1
    return V @ np.diag(eigenvalues) @ V.T

    
def createGinenvRotation(phi, p, q, d):
    G = np.eye(d)
    G[p, p] = np.cos(phi)
    G[p, q] = np.sin(phi)
    G[q, p] = -np.sin(phi)
    G[q, q] = np.cos(phi)
    return G


def generateCovMatr_and_larget_eigenval(d):
    eigenval = np.random.uniform(1, 16, size=d)
    rotation_angels = np.random.uniform(-np.pi / 4, 3 * np.pi / 4, size=int(d * (d - 1) / 2))
    return createCovMatr_from_params(eigenval, rotation_angels), max(eigenval)

def is_spd(A):
    eigvals = np.linalg.eigvals(A)
    return  np.all(eigvals > 0)

def generate_from_GMM(weights, gaussian_params, n_samples):
    n_comp = len(gaussian_params)
    data = []
    for i in range(n_samples):
        component = np.random.choice(n_comp, p=weights)
        data.append(np.random.multivariate_normal(mean=gaussian_params[component]['mu'], cov=gaussian_params[component]['sigma'], size=1).squeeze())
    return np.array(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dataset')

    parser.add_argument('-n_samples', metavar='N', type=int,
                    help='an integer for the accumulator')
    parser.add_argument('-dim', metavar='D', type=int,
                    help='Number of dimensions')
    parser.add_argument('-n_comp', metavar='N_COMP', type=int, help='Number fo components in mixture')
    parser.add_argument('-c_sep', type=float, help='C separation value')
    parser.add_argument('-filename', type=str, help='Name of file dataset will be saved to')

    args = parser.parse_args()

    n_samples = args.n_samples
    dim = args.dim
    n_components = args.n_comp
    c_separation = args.c_sep
    filename = args.filename

    if filename is None:
        filename = f'Synthetic_dim_{dim}_n_samples_{n_samples}_n_comp_{n_components}_c_separation_{c_separation}.data'

    weights, gmm = generate_GaussianMixture(n_components=n_components, d=dim, c_separation=c_separation)
    print(weights)
    # print()
    input()

    is_separated = True
    is_sigma_spd = True
    for i in range(n_components):
        for j in range(i + 1, n_components):
            is_separated = is_separated and check_separation(gmm[i], gmm[j], c_separation=c_separation)
        is_sigma_spd = is_sigma_spd and is_spd(gmm[i]['sigma'])


    data = generate_from_GMM(weights, gmm, n_samples=n_samples)
    np.save(filename, data)
