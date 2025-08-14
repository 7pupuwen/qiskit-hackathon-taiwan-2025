from typing import List  # For typing annotation.
from qiskit.circuit import QuantumCircuit # Importing QuantumCircuit for circuit operations.
import torch  # Importing torch for tensor operations.

GATE_ID_TO_NAME = {
    0: "cz",
    1: "id",
    2: "rx",
    3: "rz",
    4: "rzz",
    5: "sx",
    6: "x"
}

def decode_actions_into_circuit(actions: List, num_qubits: int = None) -> QuantumCircuit:
    if isinstance(actions, torch.Tensor):
        actions = actions.cpu().numpy()

    # 推斷 num_qubits
    if num_qubits is None:
        max_q1 = int(max(actions[:,1])) if len(actions) > 0 else 0
        max_q2 = int(max(actions[:,2][actions[:,2] >= 0])) if any(actions[:,2] >= 0) else -1
        num_qubits = max(max_q1, max_q2) + 1

    qc = QuantumCircuit(num_qubits)

    for row in actions:
        gate_id, q1, q2, param = row
        gate_id = int(gate_id)
        q1 = int(q1)
        q2 = int(q2)
        param = float(param)

        if gate_id not in GATE_ID_TO_NAME:
            continue

        gate_name = GATE_ID_TO_NAME[gate_id]

        # 新增避免 duplicate qubit 的檢查
        if gate_name in ["cz", "rzz"] and q1 == q2:
            # 同一個 qubit 做兩個位置，跳過該動作
            continue

        # 其餘依序執行
        if gate_name == "rx":
            qc.rx(param, q1)
        elif gate_name == "rz":
            qc.rz(param, q1)
        elif gate_name == "rzz" and q2 >= 0:
            qc.rzz(param, q1, q2)
        elif gate_name == "cz" and q2 >= 0:
            qc.cz(q1, q2)
        elif gate_name == "sx":
            qc.sx(q1)
        elif gate_name == "x":
            qc.x(q1)
        elif gate_name == "id":
            qc.id(q1)

    return qc

    pass
