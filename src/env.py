from typing import List, Tuple, Union, Optional  # For typing annotation.
import gymnasium as gym                          # For open-ai gym compatibility.
import numpy as np
import warnings                                  # To ignore warnings.
import torch                                     # For tensor manipulation (state and action space).
import time                                      # For time tracking.
import os                                        # For file system access.

# Custom helper functions for the environment:
from src.helper_functions.decoding import decode_actions_into_circuit
from src.helper_functions.encoding import encode_circuit_into_input_embedding  

# Quantum Circuits:
from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter # To build quantum circuits.
from qiskit import qpy                                                # To save quantum circuits to Disk.

# Quantum Chemistry:
from qiskit_nature.second_q.drivers import PySCFDriver                             # Driver for obtaining molecular information using PySCF.
from qiskit_nature.second_q.circuit.library import HartreeFock                     # Constructs the Hartree-Fock initial state.
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer, FreezeCoreTransformer              # Freezes core spatial orbitals to reduce the molecule size in quantum simulations.
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo              # Data structure for storing molecule information.
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver         # Classical solver for finding the exact minimum eigenvalue of a Hamiltonian.
from qiskit.quantum_info.operators.symplectic.sparse_pauli_op import SparsePauliOp # Used for type annotation.
from qiskit_nature.second_q.mappers import JordanWignerMapper, BravyiKitaevMapper, ParityMapper # Maps fermionic operators to spin qubit-equivalent operators.

#--------------------------------
# Estimators for computing the expectation value of the Hamiltonian.

# For exact theoretical calculations using the statevector simulation without noise or statistical sampling:
from qiskit.primitives import StatevectorEstimator # Docs: https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.primitives.StatevectorEstimator

# For high-performance simulations with Aer when testing circuits with noise models.
#from qiskit_aer.primitives import Estimator as aer_estimator # Docs: https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.primitives.Estimator.html
#--------------------------------

# Ignore warnings:
warnings.filterwarnings("ignore")

class VQEnv(gym.Env):
    '''
    - Args:
        - molecule_name (str): the name of the molecule.
        - mapper_name (str): the name of the mapper to use for converting fermionic operators to qubit operators.
        - conv_tol (float): the convergence tolerance.
        - max_circuit_depth (int): the maximum number of circuit layers.
        - max_steps_per_episode (int): the maximum number of iterations per episode.
        - symbols (list): the molecule's symbols.
        - geometry (tuple): the molecule's coordinates.
        - multiplicity (int): the molecule's multiplicity: 2*spin + 1.
        - charge (int): the molecule's charge.
        - num_electrons (int): the number of electrons to use in the calculation.
        - num_spatial_orbitals (int): the molecule's number of spatial orbitals can be provided or found.
        - num_particles (tuple): the molecule's number of particles can be provided or found.
        - fci_energy (float): the molecule's ground state energy can be provided or found.
        - qubit_operator (SparsePauliOp): the molecule's qubit-equivalent Hamiltonian can be provided or found.
        - num_qubits (int): the molecule's number of qubits can be provided or found.
    '''
    def __init__(self,
                 # Hyperparameters:
                 molecule_name: str = "Unknown",
                 mapper_name: str = "Jordan-Wigner",
                 basis: str = "sto3g",
                 conv_tol: float = 1e-5,
                 max_circuit_depth: int = 50,
                 max_steps_per_episode: int = 20,

                 # Molecule information:
                 symbols: Optional[List[str]] = None,
                 geometry: Optional[Tuple[List[float], ... ]] = None,
                 multiplicity: Optional[int] = 1,
                 charge: Optional[int] = 0,
                 num_electrons: Optional[int] = None,  
                 num_spatial_orbitals: Optional[int] = None,
                 num_particles: Optional[Tuple[int, ... ]] = None,
                 fci_energy: Optional[float] = None,

                 # Qubit operator:
                 qubit_operator: Optional[SparsePauliOp] = None,
                 num_qubits: Optional[int] = None,
                 ):

        # Run the constructor of the parent class (gym.Env):
        super().__init__()

        # Hyperparameters:
        self.molecule_name = molecule_name
        self.conv_tol = conv_tol
        self.max_circuit_depth = max_circuit_depth
        self.max_steps_per_episode = max_steps_per_episode

        # Mapper:
        match mapper_name:
            case "Jordan-Wigner":
                mapper = JordanWignerMapper()
            case "Bravyi-Kitaev":
                mapper = BravyiKitaevMapper()
            case "Parity":
                mapper = ParityMapper(num_particles=num_particles)
            case _:
                raise ValueError(f"Unknown mapper_name: {mapper_name}")

        # If the qubit operator was not provided:
        if qubit_operator is None:
            # Input Validation:
            if not isinstance(symbols, list):
                raise ValueError("Something is missing. The symbols argument must be provided as a list of strings.")
            if not isinstance(geometry, tuple):
                raise ValueError("Something is missing. The geometry argument must be provided as a tuple of lists of floats.")
            if not isinstance(multiplicity, int):
                raise ValueError("Something is missing. The multiplicity argument must be provided as a type int.")
            if not isinstance(charge, int):
                raise ValueError("Something is missing. The charge argument must be provided as a type int.")
            # Get molecule properties:
            (
                self.qubit_operator, 
                num_particles, 
                num_spatial_orbitals, 
                molecule,
            ) = self.get_qubit_op(
                symbols = symbols,
                geometry = geometry,
                multiplicity = multiplicity, 
                charge = charge,
                basis = basis,
                num_electrons_to_use = num_electrons,
                num_spatial_orbitals_to_use = num_spatial_orbitals,
                mapper = mapper
            )
            self.fci_energy = fci_energy or self.get_fci_energy(self.qubit_operator, molecule = molecule)
            self.num_qubits = num_qubits or self.qubit_operator.num_qubits
        # If the qubit operator was provided:
        else:
            # Input Validation:
            if not isinstance(qubit_operator, SparsePauliOp):
                raise ValueError("The qubit_operator argument must be provided as a Qiskit SparsePauliOp object.")
            if not isinstance(num_spatial_orbitals, int):
                raise ValueError("The num_spatial_orbitals argument must be provided as a type int.")
            if not isinstance(fci_energy, float):
                raise ValueError("The fci_energy argument must be provided as a type float.")
            self.qubit_operator = qubit_operator
            self.fci_energy = fci_energy
            self.num_qubits = num_qubits or self.qubit_operator.num_qubits

        # Reward range:
        self.reward_range = (-float('inf'), float('inf'))  # [min, max]

        # Reference Hartree-Fock state:
        self.HF = HartreeFock(num_spatial_orbitals, num_particles, mapper)

        # Action space:
        self.action_space = 5 # For example, number of gate types (e.g., Rx, Ry, Rz, H, CNOT).

        # Observation space:
        self.observation_space = gym.spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), shape=(3,), dtype='float32')

        # Counter for rendering:
        self.render_counter = 0

        # Metadata:
        self.metadata = {
            'render_modes': ['circuit', 'energy', 'reward'],
            'render_fps': 60,
        }

        print(f"\nNumber of spatial orbitals: {num_spatial_orbitals}")
        print(f"Number of particles: {num_particles}")
        print(f"Number of qubits: {self.num_qubits}")
        print(f'FCI Energy: {self.fci_energy}')

    def get_qubit_op(self, 
                     symbols, 
                     geometry, 
                     multiplicity, 
                     charge, 
                     basis,
                     num_electrons_to_use, 
                     num_spatial_orbitals_to_use,
                     mapper):
        """
        Get the qubit-equivalent Hamiltonian of a particular molecule.

        Args:
            symbols (list): The chemical symbols of the atoms in the molecule.
            geometry (tuple): The coordinates of the atoms in the molecule.
            multiplicity (int): The multiplicity of the molecule (2*spin + 1).
            charge (int): The charge of the molecule.
            basis (str): The basis set to use for the molecular calculation.
            num_electrons_to_use (int): The number of electrons to use in the calculation.
            num_spatial_orbitals_to_use (int): The number of spatial orbitals to use in the calculation.
            mapper: The mapper to use for converting fermionic operators to qubit operators.

        Returns:
            qubit_op (SparsePauliOp): The qubit operator representing the Hamiltonian.
            num_particles (tuple): The number of particles in the molecule.
            num_spatial_orbitals (int): The number of spatial orbitals in the molecule.
            molecule (MoleculeInfo): The molecule information after freezing orbitals.
        """
        print(f'\nBuilding the qubit-equivalent Hamiltonian for the {self.molecule_name} molecule...')

        z2symmetry_reduction = None

        # Mol. info with coordinates in Angstrom:
        molecule_info = MoleculeInfo(
            symbols=symbols,
            coords=geometry,
            multiplicity=multiplicity,
            charge=charge,
        )

        # Driver:
        driver = PySCFDriver.from_molecule(molecule_info, basis=basis)

        # Get the electronic structure of the molecule:
        molecule = driver.run()

        #print('Num of alpha+beta spin electrons before reduction:', molecule.num_alpha + molecule.num_beta)
        #print('Num of spatial orbitals before reduction:', molecule.num_spatial_orbitals)

        # Reduce the molecule size using ACTIVE space approximation:
        transformer = ActiveSpaceTransformer(
            num_electrons=num_electrons_to_use,
            num_spatial_orbitals=num_spatial_orbitals_to_use,
            active_orbitals=None
        )

        # Reduced electronic structure of the molecule:
        molecule  = transformer.transform(molecule)

        #print('\nNum of alpha+beta spin electrons after reduction:', molecule.num_alpha + molecule.num_beta)
        #print('Num of spatial orbitals after reduction:', molecule.num_spatial_orbitals)

        # Properties:
        num_particles = molecule.num_particles
        num_spatial_orbitals = molecule.num_spatial_orbitals

        # Get the Operator in the 2nd quantization formalism:
        second_q_ops = molecule.second_q_ops()

        # Get the Hamiltonian:
        hamiltonian = second_q_ops[0]

        # Get the qubit operator:
        qubit_op = mapper.map(hamiltonian)

        # Apply symmetry:
        if z2symmetry_reduction != None:
            tapered_mapper = molecule.get_tapered_mapper(mapper)
            qubit_op = tapered_mapper.map(hamiltonian)

        return qubit_op, num_particles, num_spatial_orbitals, molecule

    def get_fci_energy(self, qubit_op, molecule):
        """
        Get the ground state energy of a particular molecule.

        Args:
            qubit_op (SparsePauliOp): The qubit operator representing the Hamiltonian.
            molecule (MoleculeInfo): The molecule.

        Returns:
            float: The ground state energy of the molecule.
        """

        sol = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qubit_op)
        result = molecule.interpret(sol)
        return result.total_energies[0].real

    def compute_expectation_value(self, ansatz, hamiltonian, params) -> float:
        """
        Compute the expectation value of the Hamiltonian given a custom ansatz and parameters.

        Args:
            ansatz (QuantumCircuit): The quantum circuit representing the ansatz.
            hamiltonian (SparsePauliOp): The Hamiltonian as a SparsePauliOp object.
            params (np.ndarray or torch.Tensor): Parameter values for the ansatz.

        Returns:
            float: The computed expectation value.
        """

        # Estimator:
        estimator = StatevectorEstimator()
        # Pubs:
        pub = (ansatz, hamiltonian, params)
        # Run:
        job = estimator.run([pub])
        result = job.result()[0]
        expectation_value = result.data.evs

        return expectation_value

    def compute_reward(self):
    def compute_reward(self, ansatz: QuantumCircuit, params: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Computes the reward for a given circuit.

        Args:
            ...

        Returns:
            reward (float): the reward value.
         計算給定 ansatz 和參數的哈密頓期望值，並轉成獎勵值。
        以負期望能量作為獎勵，越低能量獎勵越高。
        可加入對電路深度的簡單懲罰促使電路更簡潔。
        """
         hamiltonian = self.qubit_operator
        expectation_value = self.compute_expectation_value(ansatz, hamiltonian, params)
    
        # 範例：獎勵為負期望值減去電路深度懲罰（depth * 0.01）
        depth_penalty = 0.01 * ansatz.depth()
        reward = -expectation_value - depth_penalty
    
        return reward

        pass

    def reset(self, seed: int = 42, options: dict = {}) -> tuple:
        """
        Reset the environment to the initial state.

        - Args:
            - seed (int): random seed.
            - options (dict): dictionary of options with additional information of how to reset the environment.
        
        - Returns:
            - state (numpy.ndarray): the initial state of the environment.
            - info (dict): additional information if required.
        """
        
        print('\nReseting the environment...')
        # Seed for reproducibility, i.e., to generate the same initial state for each episode:
        np.random.seed(seed)
        # Terminal state reached:
        self.terminated = False
        # Max episode length (time steps) reached:
        self.truncated = False
        # Counter for the Maximum number of steps in the episode:
        self.counter = 0
        # Initialize the environment to a random state:
        self.state = self.observation_space.low.copy()
        self.info = {'ep_reward': [], 'ep_energy': []}
        return self.state, self.info
    
    def step(self, action: List) -> tuple:
        """
        Returns a single experience from the environment.

        - Args:
            - action (List): action taken by the agent.

        - Returns:
            - self.new_state (numpy.ndarray or tf.Tensor): the next state normalized.
            - self.reward (float): reward for the action taken.
            - self.terminated (bool): whether the episode is terminated.
            - self.truncated (bool): whether the episode is truncated.
            - self.info (dict): to ensure gym-compliance.
        """

        # Update counter
        self.counter += 1

        # 1. 解碼 action 成量子電路
        ansatz = decode_actions_into_circuit(action, num_qubits=self.num_qubits)
        
        # 2. 計算期望能量
        # 提示：若 ansatz 需要參數，可自訂參數或使用固定值（這裡假設無參數）
        params = []  # 空列表或預設參數，可依需求調整
        energy = self.compute_expectation_value(ansatz, self.qubit_operator, params)
        
        # 3. 計算 reward
        self.reward = self.compute_reward(ansatz, params)
        
        # 4. 編碼電路為 state
        self.new_state = encode_circuit_into_input_embedding(ansatz, max_gates=self.max_circuit_depth)
        
        # 5. 判斷是否終止或截斷
        self.terminated = (abs(energy - self.fci_energy) < self.conv_tol) or (self.counter >= self.max_steps_per_episode)
        self.truncated = (self.counter >= self.max_steps_per_episode)
        
        # 6. 更新 info
        self.info['ep_energy'].append(energy)
        self.info['ep_reward'].append(self.reward)
        
        return self.new_state, self.reward, self.terminated, self.truncated, self.info
        '''
        Write your code here.

        Tip:
        - Decode the action into a quantum circuit.
        - Compute the expectation value of the Hamiltonian.
        - Compute the reward based on the expectation value and circuit depth.
        - Encode the quantum circuit into a tensor representation.
        - Update the state with the new tensor representation.
        - Check if the episode is terminated or truncated.
        - Update the info dictionary with the current energy and reward.
        '''

        return self.new_state, self.reward, self.terminated, self.truncated, self.info

    def render(self, mode: list = ['circuit', 'energy', 'reward'], flag: str = 'inference'):
        '''
        Plot the optimized circuit, reward curve and energy curve in a .png file.
        '''
        if self.state is None:
            raise ValueError('Failed to render. The environment is not initialized yet. Call the reset() method first.')

        def _render_circuit():
            save_dir = f'./results/{self.mol_name}/{flag}/imgs/optimized_circuit/'
            os.makedirs(save_dir, exist_ok=True)

        def _render_energy():
            save_dir = f'./results/{self.mol_name}/{flag}/imgs/energy_curve/'
            os.makedirs(save_dir, exist_ok=True)
            self.info['ep_energy'].append(self.current_energy)

        def _render_reward():
            save_dir = f'./results/{self.mol_name}/{flag}/imgs/reward_trend/'
            os.makedirs(save_dir, exist_ok=True)
            self.info['ep_reward'].append(self.reward)

        for m in mode:
            print(f'Rendering {m}...')
            if m not in self.metadata['render_modes']:
                raise ValueError(f'Invalid mode. Choose one of the following: {self.metadata["render_modes"]}')
            elif m == 'circuit':
                _render_circuit()
            elif m == 'energy':
                _render_energy()
            elif m == 'reward':
                _render_reward()

        self.render_counter += 1
    
    def close(self):
        """
        (Optional) Close the environment.
        """

        '''
        Write your code here.
        '''

        pass
