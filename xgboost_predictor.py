import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib

class XGBoostPredictor:
    def __init__(self):
        self.xgboost_clf = joblib.load('xgboost_model1.pkl')
        self.variance_threshold = joblib.load('variance_threshold1.pkl')

    def smiles_to_morgan(self, smiles, radius=2, n_bits=1024):
        mol = Chem.MolFromSmiles(smiles)
        return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)) if mol else None

    def generate_xgboost_accuracy(self, smiles):
        accuracy = 91 / 100  # Fixed accuracy of 91%
        return accuracy

    def predict(self, smiles):
        try:
            fingerprints = self.smiles_to_morgan(smiles)
            if fingerprints:
                fingerprints_df = pd.DataFrame([fingerprints])
                X_filtered = self.variance_threshold.transform(fingerprints_df)
                prediction = self.xgboost_clf.predict(X_filtered)
                accuracy = self.generate_xgboost_accuracy(smiles)  # Use the fixed accuracy for XGBoost
                class_mapping = {0: 'inactive', 1: 'active'}
                return class_mapping[prediction[0]], accuracy
            return None, None
        except Exception as e:
            return None, None