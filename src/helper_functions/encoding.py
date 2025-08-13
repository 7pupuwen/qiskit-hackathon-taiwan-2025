from typing import List  # For typing annotation.
from qiskit.circuit import QuantumCircuit
import torch  # Importing torch for tensor operations.
import numpy as np

def encode_circuit_into_input_embedding(qc: QuantumCircuit) -> torch.Tensor:
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
    # 定義gate類型對應的整數ID（可按需求擴充）
    gate_type_dict = {
    "CZ": 0,
    "RX": 1,
    "RZ": 2,
    "RZZ": 3,
    "SX": 4,
    "X" : 5,
    "id": 6
    }
    
    # 最大gate數量（固定矩陣行數）
    max_gates = 10

    # 特殊值，表示無第二個量子位元（單體閘）
    NO_QUBIT = -1

    num_features = 4  # Gate_Type_ID, Qubit_1_ID, Qubit_2_ID, Parameter_Value
    matrix = np.zeros((max_gates, num_features), dtype=float)

    for i, gate in enumerate(gate_list):
        if i >= max_gates:
            break  # 超過最大gate數則忽略

        gate_type_id = gate_type_dict.get(gate["type"], -1)
        qubit_1 = gate["qubits"][0]
        qubit_2 = gate["qubits"][1] if len(gate["qubits"]) > 1 else NO_QUBIT
        param = gate["param"] if gate["param"] is not None else 0.0

        matrix[i, 0] = gate_type_id
        matrix[i, 1] = qubit_1
        matrix[i, 2] = qubit_2
        matrix[i, 3] = param

    return matrix

    pass
