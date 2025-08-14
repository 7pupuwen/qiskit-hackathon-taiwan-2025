import configparser # Config file.
import numpy as np # Numerical operations.
import ast # Convert string to list.
import sys # Command-line arguments.
import os # Directories.

# Helper functions:
#from src.helper_functions.save_qubit_op import save_qubit_op_to_file
from src.helper_functions.load_qubit_op import load_qubit_op_from_file

# Import the agent and environment classes:
from src.agent import PPOAgent
from src.env import VQEnv

##########################################
if __name__ == '__main__':
    # Parse command-line arguments:
    config_file = sys.argv[1]

    # Get the path to the config.cfg file:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(current_dir, config_file)

    # Load the configuration file:
    config = configparser.ConfigParser()
    config.read(config_file_path)

    # Molecule hyperparameters:
    mol_name = config['MOL'].get('mol_name', fallback='Unknown')

    # Atoms:
    atoms_str = config['MOL'].get('atoms', fallback=None)
    atoms = ast.literal_eval(atoms_str) if atoms_str else []

    # Coordinates:
    coordinates_str = config['MOL'].get('coordinates', fallback=None)
    coordinates = ast.literal_eval(coordinates_str) if coordinates_str else ()

    # Number of particles:
    num_particles_str = config['MOL'].get('num_particles', fallback = None)
    num_particles = ast.literal_eval(num_particles_str) if num_particles_str else (0, 0)

    # Multiplicity:
    multiplicity = config.getint('MOL', 'multiplicity', fallback=1)
    # Charge:
    charge = config.getint('MOL', 'charge', fallback=0)
    # Electrons:
    num_electrons = config.getint('MOL', 'num_electrons', fallback = None)
    # Spatial orbitals:
    num_spatial_orbitals = config.getint('MOL', 'num_spatial_orbitals', fallback = None) 
    # Number of qubits:
    num_qubits = config.getint('MOL', 'num_qubits', fallback = None)
    # FCI energy:
    fci_energy = config.getfloat('MOL', 'fci_energy', fallback = None)

    # Convergence tolerance:
    conv_tol = config.getfloat('TRAIN', 'conv_tol', fallback=1e-5)

    # Training hyperparameters:
    learning_rate = config.getfloat('TRAIN', 'learning_rate', fallback=0.0003)
    gamma = config.getfloat('TRAIN', 'gamma', fallback=0.99) 
    gae_lambda = config.getfloat('TRAIN', 'gae_lambda', fallback=0.95) 
    policy_clip = config.getfloat('TRAIN', 'policy_clip', fallback=0.2) 
    batch_size = config.getint('TRAIN', 'batch_size', fallback=64) 
    num_episodes = config.getint('TRAIN', 'num_episodes', fallback=1000) # This is the number of episodes to train the agent.
    num_steps = config.getint('TRAIN', 'num_steps', fallback=20) # This is the number of steps per episode.
    num_epochs = config.getint('TRAIN', 'num_epochs', fallback=10) # This is the number of passes over the same batch of collected data for policy update.
    max_circuit_depth = config.getint('TRAIN', 'max_circuit_depth', fallback=50) 
    conv_tol = config.getfloat('TRAIN', 'conv_tol', fallback=1e-5)
    optimizer_option = config['TRAIN'].get('optimizer_option', fallback='SGD')

    ##########################################

    '''
    # Create an instance of the VQEnv class:
    env = VQEnv(molecule_name = "LiH", 
                symbols = atoms, 
                geometry = coordinates, 
                multiplicity = multiplicity, 
                charge = charge,
                num_electrons = num_electrons,
                num_spatial_orbitals = num_spatial_orbitals)

    # Save the qubit operator to disk:
    save_qubit_op_to_file(qubit_op = env.qubit_operator, file_name = "qubit_op_LiH.qpy")
    '''
    
    # Load the qubit operator from disk:
    qubit_operator = load_qubit_op_from_file(file_path = "./src/operators/qubit_op_LiH.qpy")

    ##########################################

    # Create the environment with the loaded qubit operator:
    env = VQEnv(qubit_operator = qubit_operator, 
                num_spatial_orbitals = num_spatial_orbitals, 
                num_particles = num_particles,
                fci_energy = fci_energy)

    # Agent:
    agent = PPOAgent(
        state_dim = env.observation_space.shape[0],
        action_dim = env.action_space,
        learning_rate = learning_rate,
        gamma = gamma,
        gae_lambda = gae_lambda,
        policy_clip =policy_clip,
        batch_size = batch_size,
        num_epochs = num_epochs,
        optimizer_option = optimizer_option,
        chkpt_dir = 'model/ppo')

    # Training loop:
    for i in range(num_episodes):
    observation, info = env.reset()  # 重置環境，得到初始狀態(state)和附加資訊(info)
    done = False
    truncated = False
    ep_reward = 0.0

    while not (done or truncated):
        # 1. 將 observation 轉成 tensor (浮點數、batch size = 1)
        state_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(agent.device)

        # 2. 呼叫 agent 的 sample_action，取得動作、對應機率、state價值
        action, probs, value = agent.sample_action(state_tensor)

        # 3. 對環境執行該動作，得到新狀態、獎勵、是否終止、截斷、附加資訊
        new_observation, reward, done, truncated, info = env.step(action)

        # 4. 將當前 transition (state, action, reward, probs, value, done) 存入記憶體
        agent.store_transitions(observation, action, reward, probs, value, done)

        # 5. 更新狀態
        observation = new_observation

        # 6. 累積回合獎勵
        ep_reward += reward

        # 7. 如果記憶體達到 batch size，執行學習更新
        if agent.memory_buffer.ready():
            agent.learn()

    # 每回合結束後也可呼叫學習，確保全部數據都被用到
    if agent.memory_buffer.has_data():
        agent.learn()

    # 印出該回合結果，方便觀察訓練進度
    print(f"Episode {i+1}/{num_episodes}, Total Reward: {ep_reward:.4f}, Last Energy: {info['ep_energy'][-1]:.6f}")

    # 可視需要定期儲存模型
    if (i + 1) % 100 == 0:
        agent.save_models()
