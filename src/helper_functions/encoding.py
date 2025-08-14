from typing import List  # For typing annotation.
from qiskit.circuit import QuantumCircuit
import torch  # Importing torch for tensor operations.
import numpy as np

# gate 集合對應 ID
gate_type_dict = {
    "cz": 0,
    "id": 1,
    "rx": 2,
    "rz": 3,
    "rzz": 4,
    "sx": 5,
    "x": 6
}

NO_QUBIT = -1  # 單比特閘第二個 qubit 填 -1
    
def encode_circuit_into_input_embedding(qc: QuantumCircuit, max_gates=20):
    num_features = 4
    matrix = np.zeros((max_gates, num_features), dtype=float)

    for i, instr in enumerate(qc.data):
        if i >= max_gates:
            break
        gate = instr[0].name.lower()
        qargs = [qc.qubits.index(q) for q in instr[1]]
        params = instr[0].params

        gate_type_id = gate_type_dict.get(gate, -1)
        if gate_type_id == -1:
            continue

        qubit_1 = qargs[0] if len(qargs) >= 1 else NO_QUBIT
        qubit_2 = qargs[1] if len(qargs) >= 2 else NO_QUBIT

        if params:
            try:
                param_val = float(params[0])
            except Exception:
                param_val = 0.0
            param = round(param_val, 4)
        else:
            param = 0.0

        # 確保只記錄有效門
        matrix[i] = [gate_type_id, qubit_1, qubit_2, param]

    return matrix

    pass
