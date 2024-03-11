import os
# import torch
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import itertools as it
import networkx as nx
import numpy as np
import pennylane as qml
import qutip
import scipy as sp
from tqdm import tqdm
import pickle


def make_coupling_matrix(J, a, n_qubits):
    edges = [
        (si, sj) for (si, sj) in it.combinations(range(n_qubits), 2)
        if sj > si
    ]

    coupling_matrix = np.zeros((n_qubits, n_qubits))
    for i, j in edges:
        prob = np.random.uniform()
        if j - i > 1:
            if prob > 0.5:
                coupling_matrix[i, j] = coupling_matrix[j, i] = 0.0
            elif coupling_matrix[i, j-1] == 0.0:
                coupling_matrix[i, j] = coupling_matrix[j, i] = 0.0
            else:
                coupling_matrix[i, j] = coupling_matrix[j, i] = J / (abs(j-i) ** a) / 3
        else:
             coupling_matrix[i, j] = coupling_matrix[j, i] = J / (abs(j-i) ** a) / 3
    
    return coupling_matrix


def build_hamiltonian(coupling_matrix):
    coeffs, ops = [], []
    ns = coupling_matrix.shape[0]

    for i, j in it.combinations(range(ns), r=2):
        coeff = coupling_matrix[i, j]
        if coeff:
            for op in [qml.PauliX, qml.PauliY, qml.PauliZ]:
                coeffs.append(coeff)
                ops.append(op(i) @ op(j))

    return qml.Hamiltonian(coeffs, ops)

def compute_exact_entropy(ground_state, wires):
    ground_state_qobj = qutip.Qobj(ground_state, dims=[[2] * wires, [1] * wires])

    # compute entropies
    ptrace_diag = ground_state_qobj.ptrace(sel=list(range(wires//2)))
    entropy = -np.log(np.trace(ptrace_diag * ptrace_diag).real)

    return entropy

def generate_pauli6_samples(statevector, wires, shots, device_name='default.qubit'):
    """ generate samples from the Pauli-6 POVM. The resulting samples are encoded as integers according to
    the rules
    0,1 -> +,- in the X Basis
    2,3 -> r,l in the Y Basis
    4,5 -> 0,1 in the Z Bassi
    """
    @qml.qnode(device=qml.device(device_name, wires=wires, shots=shots), diff_method=None, interface=None)
    def shadow_measurement():
        qml.QubitStateVector(statevector, wires=range(wires))
        return qml.classical_shadow(wires=range(wires))

    bits, recipes = shadow_measurement()

    # encode measurements and bases as integers
    data = 2 * recipes + bits
    data = np.array(data, dtype=int)

    return data

def compute_exact_correlation_matrix(ground_state, wires):
    # this circuit measures observables for the provided ground state
    @qml.qnode(device=qml.device('default.qubit', wires=wires, shots=None))
    def circ(observables):
        qml.QubitStateVector(ground_state, wires=range(wires))
        return [qml.expval(o) for o in observables]
    
    # setup observables for correlation function
    def corr_function(i, j):
        ops = []
        
        for op in [qml.PauliX, qml.PauliY, qml.PauliZ]:
            if i != j:
                ops.append(op(i) @ op(j))
            else:
                ops.append(qml.Identity(i))

        return ops
    
    # indices for sites for which correlations will be computed
    coupling_pairs = list(it.product(range(wires), repeat=2))
    
    # compute exact correlation matrix
    correlation_matrix = np.zeros((wires, wires))
    for idx, (i, j) in enumerate(coupling_pairs):
        observable = corr_function(i, j)

        if i == j:
            correlation_matrix[i][j] = 1.0
        else:
            correlation_matrix[i][j] = (
                    np.sum(np.array([circ(observables=[o]) for o in observable]).T) / 3
            )
            correlation_matrix[j][i] = correlation_matrix[i][j]

    return correlation_matrix

def compute_shadow_correlation_matrix(bits, recipes, wires):
    # this circuit measures observables for the provided ground state
    shadow = qml.ClassicalShadow(bits=bits, recipes=recipes)
    
    # setup observables for correlation function
    def corr_function(i, j):
        ops = []
        
        for op in [qml.PauliX, qml.PauliY, qml.PauliZ]:
            if i != j:
                ops.append(op(i) @ op(j))
            else:
                ops.append(qml.Identity(i))

        return ops
    
    # indices for sites for which correlations will be computed
    coupling_pairs = list(it.product(range(wires), repeat=2))
    
    # compute exact correlation matrix
    correlation_matrix = np.zeros((wires, wires))
    for idx, (i, j) in enumerate(coupling_pairs):
        observable = corr_function(i, j)

        correlation_matrix[i][j] = (
                np.sum(np.array([shadow.expval(o, k=1) for o in observable]).T) / 3
        )
        correlation_matrix[j][i] = correlation_matrix[i][j]

    return correlation_matrix





seeds = [345345, 43236, 634755]
save_path  = 'heisenberg_data/'

n_train = 100
n_test = 500
n_samples = n_train + n_test

n_qubits_list = [8, 10, 12]
shots_list = [1024]

for seed, n_qubits in zip(seeds, n_qubits_list):
    print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(f"Generate Data for {n_qubits} qubits.")
    np.random.seed(seed)

    # generate system variables
    J_array = [369 for _ in range(n_samples)]
    a_array = np.random.uniform(1, 2, n_samples)
    assert 1.0 not in a_array

    # generate n*n coupling matrix
    coupling_matrix_list = []
    for i in range(n_samples):
        coupling_matrix_list.append(make_coupling_matrix(J_array[i], a_array[i], n_qubits))

    # build sparse hamiltonian
    hamiltonian_list = []
    for i in range(n_samples):
        H = build_hamiltonian(coupling_matrix_list[i])
        H_sparse = qml.utils.sparse_hamiltonian(H)
        hamiltonian_list.append(H_sparse)

    # diagonalize
    eigvals_list = []
    ground_state_list = []
    for i in range(n_samples):
        eigvals, eigvecs = sp.sparse.linalg.eigs(hamiltonian_list[i], which='SR', k=1)
        eigvals = eigvals.real.item()
        ground_state = eigvecs[:, np.argmin(eigvals)]
        
        eigvals_list.append(eigvals)
        ground_state_list.append(ground_state)

    # compute exact entropies
    # entropies_list = []
    # for i in range(n_samples):
    #     ground_state = ground_state_list[i].reshape(-1)
    #     entropies_list.append(compute_exact_entropy(ground_state, wires=n_qubits))

    # compute correlation matrix
    exact_correlation_matrix_list = []
    for i in tqdm(range(n_samples), total=n_samples):
        exact_correlation_matrix = compute_exact_correlation_matrix(ground_state_list[i], n_qubits)
        exact_correlation_matrix = exact_correlation_matrix.reshape([1, exact_correlation_matrix.shape[0], -1])
        exact_correlation_matrix_list.append(exact_correlation_matrix)
    exact_correlation_matrix_list = np.concatenate(exact_correlation_matrix_list, axis=0)   # n_samples, n_qubits, n_qubits

    for shots in shots_list:
        # generate measurement records
        measurements_list = []
        for i in range(n_samples):
            ground_state = ground_state_list[i]
            measurements_list.append(generate_pauli6_samples(ground_state.reshape(-1), wires=n_qubits, shots=shots))

        # compute correlation matrix based on classical shadow
        shadow_correlation_matrix_list = []
        for i in tqdm(range(n_samples), total=n_samples):
            measurements = measurements_list[i]

            # splits sampels into measurement outcomes and bases
            recipes = measurements // 2
            bits = measurements - 2 * recipes

            # instantiate shadow
            shadow = qml.ClassicalShadow(bits=bits, recipes=recipes)

            # compute correlation matrix using ClassicalShadow
            shdow_correlation_matrix = compute_shadow_correlation_matrix(bits=bits, recipes=recipes, wires=n_qubits)
            shdow_correlation_matrix = shdow_correlation_matrix.reshape([1, shdow_correlation_matrix.shape[0], -1])
            shadow_correlation_matrix_list.append(shdow_correlation_matrix)


        # compute RMSE loss
        shadow_correlation_matrix_list = np.concatenate(shadow_correlation_matrix_list, axis=0) # n_samples, n_qubits, n_qubits 

        loss = np.mean((shadow_correlation_matrix_list[n_train:] - exact_correlation_matrix_list[n_train:]) ** 2, axis=(1,2))
        loss = np.sqrt(loss)
        mean, std = np.mean(loss), np.std(loss)
        print(f'For nq-{n_qubits}_nM-{shots}: loss mean: {mean}, loss std: {std}')

        # save data
        train_data = {}
        test_data = {}

        conditions_list = []
        inputs_list = []
        labels_list = []
        for i in range(n_samples):
            condition = np.asarray(coupling_matrix_list[i])
            condition = condition.reshape(1, -1)
            input = np.asarray(measurements_list[i])
            input = input.reshape(1, input.shape[0], -1)
            label = exact_correlation_matrix_list[i]
            label = label.reshape(1, label.shape[0], -1)

            conditions_list.append(condition)
            inputs_list.append(input)
            labels_list.append(label)

        conditions = np.concatenate(conditions_list, axis=0)
        inputs = np.concatenate(inputs_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        train_data['conditions'] = conditions[:n_train]
        train_data['inputs'] = inputs[:n_train]
        train_data['labels'] = labels[:n_train]

        test_data['conditions'] = conditions[n_train:]
        test_data['inputs'] = inputs[n_train:]
        test_data['labels'] = labels[n_train:]


        # print(f'Data shape',
        #       f'conditions: {conditions.shape}',
        #       f'inputs: {inputs.shape}')

        file_name = f'train_data_nq-{n_qubits}_nM-{shots}.pkl'
        file_path = os.path.join(save_path, file_name)
        pickle.dump(train_data, open(file_path, 'wb'))

        file_name = f'test_data_nq-{n_qubits}_nM-{shots}_ntest-{n_test}.pkl'
        file_path = os.path.join(save_path, file_name)
        pickle.dump(test_data, open(file_path, 'wb'))

    print(f"Generate Done.")
    print(f'''Train Data shape''',
        f'''conditions: {train_data['conditions'].shape}''',
        f'''inputs: {train_data['inputs'].shape}''',
        f'''labels: {train_data['labels'].shape}''')
    print(f'''Test Data shape''',
        f'''conditions: {test_data['conditions'].shape}''',
        f'''inputs: {test_data['inputs'].shape}''',
        f'''labels: {test_data['labels'].shape}''')

