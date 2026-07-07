"""
Evaluation of different QEM methods based on

    - Bias
    - Absolute error
"""

import numpy as np
import matplotlib.pyplot as plt

from qiskit.circuit.library import n_local
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime import EstimatorV2 as Estimator



def _generate_noise_model(tq_err_rate, sq_err_rate):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(
        depolarizing_error(sq_err_rate, 1), instructions=['u1', 'u2', 'u3']
    )
    nm.add_all_qubit_quantum_error(
        depolarizing_error(tq_err_rate, 2), instructions=['cx', 'cz']
    )
    return nm

