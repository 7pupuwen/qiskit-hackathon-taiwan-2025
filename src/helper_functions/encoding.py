from typing import List  # For typing annotation.
from qiskit.circuit import QuantumCircuit
import torch  # Importing torch for tensor operations.
import numpy as np
import numpy as np
from qiskit import QuantumCircuit

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
    

def encode_circuit_into_input_embedding(qc: QuantumCircuit, max_gates=20) -> torch.Tensor:
    """
    Encode (convert) the quantum circuit into a tensor representation for input to the agent.

    Args:
        qc (QuantumCircuit): The quantum circuit to be encoded.

    Returns:
        torch.Tensor: The tensor representation of the quantum circuit.
        gate_list範例輸入格式（list of dicts）：
    [
        {"type": "RX", "qubits": [0], "param": 1.57},
        {"type": "CNOT", "qubits": [1, 2], "param": None},
        {"type": "RZ", "qubits": [3], "param": 0.78},
        ...
    ]

    其中param為旋轉角度（單位弧度），若無參數則None。
    """
    num_features = 4
    matrix = np.zeros((max_gates, num_features), dtype=float)

    # 從 Qiskit 物件讀出門
    for i, instr in enumerate(qc.data):
        if i >= max_gates:
            break

        gate = instr[0].name.lower()
        qargs = [q.index for q in instr[1]]
        params = instr[0].params

        gate_type_id = gate_type_dict.get(gate, -1)  # -1 代表不在支援集合裡
        qubit_1 = qargs[0] if len(qargs) >= 1 else NO_QUBIT
        qubit_2 = qargs[1] if len(qargs) >= 2 else NO_QUBIT
        param = params[0] if params else 0.0

        matrix[i] = [gate_type_id, qubit_1, qubit_2, param]

    return matrix

    pass
