import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import ast
import re
from tqdm import tqdm

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qcfd.projects.qem_vqa.adapt_zne import adaptzne_scale_noise
from qiskit_helpers.simulate_circuit import get_noise_model
from qiskit_helpers.simulate_circuit import htest_stvec_simulation
from qiskit_helpers.aer_config import AER_BASIC_STVEC, AER_BASIC_DM

class GlobalFoldingDataset(Dataset):
    def __init__(self, simulation_points, ideal_outputs):
        """
        simulation_points: tensor or array containing circuit sample points
        ideal_outputs: tensor or array with ideal circuit results
        """
        self.x = torch.tensor(simulation_points, dtype=torch.float32)
        self.y = torch.tensor(ideal_outputs, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        input_sample = self.x[idx]
        target = self.y[idx]
        return input_sample, target
    
    @classmethod
    def from_csv(cls, csv_file):
        """
        Load a previously generated dataset from CSV file.
        
        Args:
            csv_file: Path to the CSV file containing the data
            
        Returns:
            GlobalFoldingDataset instance
        """
        df = pd.read_csv(csv_file)

        def _parse_list_or_scalar(value):
            def _normalize_np_float_wrappers(text):
                return re.sub(r"np\.float64\(([^()]+)\)", r"\1", text)

            if isinstance(value, str):
                stripped = value.strip()
                normalized = _normalize_np_float_wrappers(stripped)

                if normalized.startswith('[') and normalized.endswith(']'):
                    parsed = ast.literal_eval(normalized)
                    return np.asarray(parsed, dtype=float)

                if normalized.startswith('(') and normalized.endswith(')'):
                    parsed = ast.literal_eval(normalized)
                    return np.asarray(parsed, dtype=float)

                if stripped.startswith('[') and stripped.endswith(']'):
                    parsed = ast.literal_eval(_normalize_np_float_wrappers(stripped))
                    return np.asarray(parsed, dtype=float)

                return float(normalized)
            if isinstance(value, (list, tuple, np.ndarray)):
                return np.asarray(value, dtype=float)
            return float(value)

        df['noisy_output'] = df['noisy_output'].map(_parse_list_or_scalar)
        df['ideal_output'] = df['ideal_output'].map(_parse_list_or_scalar)
        if 'c' in df.columns:
            df['c'] = df['c'].map(_parse_list_or_scalar)

        if 'scale' in df.columns:
            df['_scale_num'] = pd.to_numeric(df['scale'], errors='coerce')

        sample_features = []
        grouped = df.groupby('sample_id', sort=False)

        for _, sample_data in grouped:
            if len(sample_data) == 1:
                noisy_vals = np.asarray(sample_data['noisy_output'].iloc[0], dtype=np.float32).reshape(-1)
                c_vals = np.asarray(sample_data['c'].iloc[0], dtype=np.float32).reshape(-1) if 'c' in sample_data.columns else np.array([], dtype=np.float32)
                features = np.concatenate([noisy_vals, c_vals])
            else:
                ordered = sample_data
                if '_scale_num' in sample_data.columns and not sample_data['_scale_num'].isna().all():
                    ordered = sample_data.sort_values('_scale_num')

                noisy_series = ordered['noisy_output'].map(lambda v: np.asarray(v, dtype=np.float32).reshape(-1))
                noisy_vals = np.concatenate(noisy_series.to_numpy())

                if 'c' in ordered.columns:
                    c_series = ordered['c'].map(lambda v: np.asarray(v, dtype=np.float32).reshape(-1))
                    c_vals = np.concatenate(c_series.to_numpy())
                else:
                    c_vals = np.array([], dtype=np.float32)

                features = np.concatenate([noisy_vals, c_vals])

            sample_features.append(features)

        target_series = grouped['ideal_output'].first().map(
            lambda v: float(np.asarray(v, dtype=float).reshape(-1)[0])
        )

        print(f"Loaded dataset from {csv_file}")
        print(f"Total samples: {len(grouped)}")
        print(f"Features per sample: {len(sample_features[0])}")

        sample_features_np = np.stack(sample_features).astype(np.float32, copy=False)
        sample_targets_np = target_series.to_numpy(dtype=np.float32)

        return cls(sample_features_np, sample_targets_np)
    
    @staticmethod
    def generate_data(size, circuit_template: QuantumCircuit, noise_model_params, scales, seed, 
                     output_file='global_folding_data.csv', target_c=10, shots0=1000, debias=True, num_workers=1):
        """
        Generate dataset for global folding ZNE training.
        
        Args:
            size: Number of data samples to generate
            circuit_template: Parameterized quantum circuit
            noise_model_params: Parameters for noise model configuration
            scales: List of noise scaling factors
            seed: Random seed for reproducibility
            output_file: Path to save the generated data (CSV format)
            target_c: Target confidence level for adaptive ZNE
            shots0: Initial number of shots
            debias: Whether to debias the dataset for large (~1) ideal values
            num_workers: Number of parallel workers (not yet implemented)
            
        Returns:
            GlobalFoldingDataset instance with generated data
        """
        # Initialize simulators and random seed
        nm = get_noise_model(noise_model_params)[0]
        aer_stv = AerSimulator(**AER_BASIC_STVEC)
        aer_dm = AerSimulator(**AER_BASIC_DM, noise_model=nm)
        circuit_template = transpile(circuit_template, backend=aer_dm)
        torch.manual_seed(seed)
        np.random.seed(seed)

        def _meas_z(dm):
            """Measure expectation value of Z operator"""
            dm = dm.data
            return np.real(dm[0, 0] - dm[1, 1])

        shot_incrementer = lambda x: np.sqrt(x)
        
        # Generate random parameter rotations for all samples
        rotations = np.random.uniform(-np.pi, np.pi, (int(size*0.95) if debias else size, circuit_template.num_parameters))
        if debias:
            size1 = size - int(size*0.95)  # This ensures 972 + 52 = 1024
            rotatinos1 = np.random.normal(0, 0.1, (int(size1), circuit_template.num_parameters))
            rotations = np.concatenate([rotations, rotatinos1], axis=0)
            np.random.shuffle(rotations)


        # Storage for results
        all_data = []
        
        print(f"Generating {size} data samples...")
        for i in tqdm(range(size), desc="Generating samples"):
            
            # Get parameters for this sample
            param_values = [{p: [v] for p, v in zip(circuit_template.parameters, rotations[i])}]
            
            # Run adaptive ZNE to get noisy measurements at different scales
            xdata, ydata, (clist, _) = adaptzne_scale_noise(
                circuit_template, scales, param_values, aer_dm, 
                target_c, shot_incrementer, shots0
            )
            
            # Compute ideal (noise-free) output
            ideal_dm = htest_stvec_simulation(circuit_template, param_values, aer_stv)
            ideal_output = _meas_z(ideal_dm)
            
            all_data.append({
                'sample_id': i,
                'scale': [float(x) for x in xdata],
                'noisy_output': [float(y[0][0]) for y in ydata],
                'ideal_output': float(ideal_output),
                'c': [float(c) for c in clist],
            })
        
        # Create DataFrame
        df = pd.DataFrame.from_records(all_data)
        df.to_csv(output_file, index=False)
        print(f"\nData saved to {output_file}")
        
        describe_dataset(df)
    
def describe_dataset(df: pd.DataFrame):
        # Save to CSV file
    print(f"\nDataFrame summary:")
    print(df.describe())
    
    # Print statistics for list columns by position
    print(f"\nNoisy output statistics (by scale point):")
    max_len = max(len(x) for x in df['noisy_output'])
    for i in range(max_len):
        values = [df['noisy_output'].iloc[j][i] for j in range(len(df)) if i < len(df['noisy_output'].iloc[j])]
        print(f"  Position {i}: mean={np.mean(values):.6f}, std={np.std(values):.6f}, min={np.min(values):.6f}, max={np.max(values):.6f}")
    
    print(f"\nC values statistics (by scale point):")
    max_len = max(len(x) for x in df['c'])
    for i in range(max_len):
        values = [df['c'].iloc[j][i] for j in range(len(df)) if i < len(df['c'].iloc[j])]
        print(f"  Position {i}: mean={np.mean(values):.6f}, std={np.std(values):.6f}, min={np.min(values):.6f}, max={np.max(values):.6f}")
    
        
if __name__ == "__main__":
    # from qiskit import QuantumCircuit
    # from global_folding_dataset import GlobalFoldingDataset
    # from qiskit_helpers.simulate_circuit import DepolarizingErrorParams
    # from qiskit.circuit.library import n_local
    # from qiskit import transpile

    # qc = QuantumCircuit(5, 5)
    # qc.h(0)
    # pqc = n_local(4, 'ry', 'cx', reps=10, entanglement='linear')
    # qc.append(pqc.control(1).to_gate(), range(qc.num_qubits))
    # qc.h(0)

    # nm_params = DepolarizingErrorParams(single_qubit_error_rate=0.001, two_qubits_error_rate=0.01)
    # GlobalFoldingDataset.generate_data(1024, qc, nm_params, (1,3,5), 1821, target_c=10, output_file="4q_10l_ry_cxlin_depolar_sq1e-3_tq1e-2_debias.csv")

    from__csv_path = '/home/pconstantinidis/Dev/angelakis_research_group/qcfd/projects/qem_vqa/4q_10l_ry_cxlin_depolar_sq1e-3_tq1e-2_debias.csv'
    describe_dataset(pd.read_csv(from__csv_path))

