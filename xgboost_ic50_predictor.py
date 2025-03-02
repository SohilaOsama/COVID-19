import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib

class XGBoostIC50Predictor:
    def __init__(self):
        self.xgboost_clf_ic50 = joblib.load('xgboost_model1_IC50.pkl')
        self.variance_threshold_ic50 = joblib.load('variance_threshold1_IC50.pkl')

    def smiles_to_morgan(self, smiles, radius=2, n_bits=1024):
        mol = Chem.MolFromSmiles(smiles)
        return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)) if mol else None

    def generate_xgboost_IC50_accuracy(self, smiles):
        accuracy = 88 / 100  # Fixed accuracy of 88%
        return accuracy

    def predict(self, smiles):
        try:
            fingerprints = self.smiles_to_morgan(smiles)
            if fingerprints:
                fingerprints_df = pd.DataFrame([fingerprints])
                X_filtered = self.variance_threshold_ic50.transform(fingerprints_df)
                prediction = self.xgboost_clf_ic50.predict(X_filtered)
                accuracy = self.generate_xgboost_IC50_accuracy(smiles)  # Use the fixed accuracy for the new XGBoost
                class_mapping = {0: 'inactive', 1: 'active'}
                return class_mapping[prediction[0]], accuracy
            return None, None
        except Exception as e:
            return None, None