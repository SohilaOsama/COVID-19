import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import joblib
import tensorflow as tf
from keras.layers import TFSMLayer

class NeuralNetworkPredictor:
    def __init__(self):
        self.nn_model = TFSMLayer('multi_tasking_model_converted', call_endpoint='serving_default')
        self.scaler = joblib.load('scaler.pkl')
        self.selected_features = joblib.load('selected_features.pkl')

    def calculate_descriptors(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return {
                'MolWt': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'NumHDonors': Descriptors.NumHDonors(mol),
                'NumHAcceptors': Descriptors.NumHAcceptors(mol)
            }
        return None

    def smiles_to_morgan(self, smiles, radius=2, n_bits=1024):
        mol = Chem.MolFromSmiles(smiles)
        return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)) if mol else None

    def generate(self, smiles):
        accuracy = 88 / 100  # Fixed accuracy of 88%
        error_percentage = 30 / 100  # Fixed error percentage of 30%
        return accuracy, error_percentage

    def predict(self, smiles):
        try:
            # Calculate molecular descriptors
            descriptors = self.calculate_descriptors(smiles)
            descriptors_df = pd.DataFrame([descriptors])

            # Convert SMILES to Morgan fingerprints
            fingerprints = self.smiles_to_morgan(smiles)
            fingerprints_df = pd.DataFrame([fingerprints], columns=[str(i) for i in range(len(fingerprints))])

            # Combine descriptors and fingerprints
            combined_df = pd.concat([descriptors_df, fingerprints_df], axis=1)

            # Scale the features
            combined_scaled = self.scaler.transform(combined_df)

            # Select only the features used during training
            combined_selected = pd.DataFrame(combined_scaled, columns=combined_df.columns)[self.selected_features]

            # Convert to NumPy array for inference
            input_data = combined_selected.to_numpy()

            # Call the TFSMLayer model
            outputs = self.nn_model(input_data)

            # Extract the outputs
            regression_pred = outputs['output_0'].numpy()  # Regression prediction (pIC50)
            classification_pred = outputs['output_1'].numpy()  # Classification prediction (bioactivity)

            # Extract final predictions
            pIC50 = regression_pred[0][0]
            bioactivity = 'active' if classification_pred[0][0] > 0.5 else 'inactive'

            # Generate fixed accuracy and error percentage
            accuracy, error_percentage = self.generate(smiles)

            return pIC50, bioactivity, accuracy, error_percentage
        except Exception as e:
            return None, None, None, None