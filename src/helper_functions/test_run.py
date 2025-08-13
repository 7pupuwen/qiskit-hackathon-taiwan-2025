# test_run.py
from qiskit import QuantumCircuit
from encoding import encode_circuit_into_input_embedding
from decoding import decode_actions_into_circuit

# Step 1: 建立一個測試電路
qc = QuantumCircuit(3)
qc.rx(1.57, 0)
qc.cz(0, 1)
qc.rzz(0.78, 1, 2)
qc.x(2)

print("原始電路：")
print(qc)

# Step 2: 編碼成矩陣
encoded = encode_circuit_into_input_embedding(qc, max_gates=10)
print("\n編碼矩陣：")
print(encoded)

# Step 3: 假設 RL 模型直接輸出相同矩陣，解碼回 QuantumCircuit
decoded_qc = decode_actions_into_circuit(encoded, num_qubits=3)

print("\n解碼後電路：")
print(decoded_qc)
