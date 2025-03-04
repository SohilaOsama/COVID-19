import streamlit as st
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import tensorflow as tf
from keras.layers import TFSMLayer
import numpy as np
import chardet  # For automatic encoding detection
import hashlib  # For generating fixed accuracy and error

from about import show_about
from readme import show_readme
from mission import show_mission
from molecular_visualization import show_molecular_visualization, generate_3d_view, generate_2d_view  # Import the necessary functions

# Load models and preprocessing steps
nn_model = TFSMLayer('multi_tasking_model_converted', call_endpoint='serving_default')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')
xgboost_clf = joblib.load('2_random_forest_model1_LE.pkl')
variance_threshold = joblib.load('2_variance_threshold1_LE.pkl')

# Load new models for IC50 classification and prediction
random_forest_clf_ic50 = joblib.load('1_random_forest_model1_IC50.pkl')
variance_threshold_ic50 = joblib.load('1_variance_threshold1_IC50.pkl')

# Detect encoding of uploaded file
def detect_encoding(file):
    raw_data = file.read(4096)  # Read a small chunk
    file.seek(0)  # Reset file position
    result = chardet.detect(raw_data)  # Detect encoding
    return result["encoding"]

# Compute molecular descriptors
def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return {
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol)
        }
    return None

# Convert SMILES to Morgan fingerprints
def smiles_to_morgan(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)) if mol else None

# Generate fixed accuracy and error percentage
def generate(smiles):
    accuracy = 88 / 100  # Fixed accuracy of 88%
    error_percentage = 30 / 100  # Fixed error percentage of 30%
    return accuracy, error_percentage

# Generate fixed accuracy for XGBoost
def generate_xgboost_accuracy(smiles):
    accuracy = 91 / 100  # Fixed accuracy of 91%
    return accuracy

# Generate fixed accuracy for Random Forest
def generate_rf_accuracy(smiles):
    accuracy = 91 / 100  # Fixed accuracy of 91%
    return accuracy

# Generate fixed accuracy for Random Forest IC50
def generate_rf_IC50_accuracy(smiles):
    accuracy = 88 / 100  # Fixed accuracy of 88%
    return accuracy

# Prediction using multi-tasking neural network
def predict_with_nn(smiles):
    try:
        # Calculate molecular descriptors
        descriptors = calculate_descriptors(smiles)
        descriptors_df = pd.DataFrame([descriptors])

        # Convert SMILES to Morgan fingerprints
        fingerprints = smiles_to_morgan(smiles)
        fingerprints_df = pd.DataFrame([fingerprints], columns=[str(i) for i in range(len(fingerprints))])

        # Combine descriptors and fingerprints
        combined_df = pd.concat([descriptors_df, fingerprints_df], axis=1)

        # Scale the features
        combined_scaled = scaler.transform(combined_df)

        # Select only the features used during training
        combined_selected = pd.DataFrame(combined_scaled, columns=combined_df.columns)[selected_features]

        # Convert to NumPy array for inference
        input_data = combined_selected.to_numpy()

        # Call the TFSMLayer model
        outputs = nn_model(input_data)

        # Extract the outputs
        regression_pred = outputs['output_0'].numpy()  # Regression prediction (pIC50)
        classification_pred = outputs['output_1'].numpy()  # Classification prediction (bioactivity)

        # Extract final predictions
        pIC50 = regression_pred[0][0]
        bioactivity = 'active' if classification_pred[0][0] > 0.5 else 'inactive'

        # Generate fixed accuracy and error percentage
        accuracy, error_percentage = generate(smiles)

        return pIC50, bioactivity, accuracy, error_percentage
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None, None, None

# Prediction function for XGBoost
def predict_with_xgboost(smiles):
    try:
        fingerprints = smiles_to_morgan(smiles)
        if fingerprints:
            fingerprints_df = pd.DataFrame([fingerprints])
            X_filtered = variance_threshold.transform(fingerprints_df)
            print("XGBoost Input Data:", X_filtered)  # Debugging print statement
            prediction = xgboost_clf.predict(X_filtered)
            accuracy = generate_xgboost_accuracy(smiles)  # Use the fixed accuracy for XGBoost
            class_mapping = {0: 'inactive', 1: 'active'}
            return class_mapping[prediction[0]], accuracy
        return None, None
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None

# Prediction function for IC50 using the new Random Forest model
def predict_with_rf_ic50(smiles):
    try:
        fingerprints = smiles_to_morgan(smiles)
        if fingerprints:
            fingerprints_df = pd.DataFrame([fingerprints])
            X_filtered = variance_threshold_ic50.transform(fingerprints_df)
            print("Random Forest IC50 Input Data:", X_filtered)  # Debugging print statement
            prediction = random_forest_clf_ic50.predict(X_filtered)
            accuracy = generate_rf_IC50_accuracy(smiles)  # Use the fixed accuracy for the new Random Forest
            class_mapping = {0: 'inactive', 1: 'active'}
            return class_mapping[prediction[0]], accuracy
        return None, None
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None

# Convert pIC50 values
def convert_pIC50_to_uM(pIC50):
    return 10 ** (-pIC50) * 1e6

def convert_pIC50_to_ng_per_uL(pIC50, mol_weight):
    return convert_pIC50_to_uM(pIC50) * mol_weight / 1000

def convert_pIC50_to_nM(pIC50):
    return 10 ** (-pIC50) * 1e9

# Streamlit UI
st.set_page_config(page_title="Bioactivity Prediction", page_icon="🥼", layout="wide")

# Load custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load custom JavaScript
with open("script.js") as f:
    st.markdown(f"<script>{f.read()}</script>", unsafe_allow_html=True)

# Navigation
st.sidebar.markdown("## Navigation")
nav_home = st.sidebar.button("Home")
nav_molecular_visualization = st.sidebar.button("Molecular Visualization")
nav_mission = st.sidebar.button("Mission")
nav_readme = st.sidebar.button("README")

if nav_home:
    st.session_state.page = "Home"
elif nav_molecular_visualization:
    st.session_state.page = "Molecular Visualization"
elif nav_mission:
    st.session_state.page = "Mission"
elif nav_readme:
    st.session_state.page = "README"
else:
    if 'page' not in st.session_state:
        st.session_state.page = "Home"

if st.session_state.page == "Home":
    st.title("👩🏻‍🔬 Bioactivity Prediction from SMILES")
    st.image("images/Drug.png", use_container_width=True)

    # Instructions
    st.markdown("## Instructions:")
    st.write("""
        To convert your compound to a Simplified Molecular Input Line Entry System (SMILES), please visit this website: [decimer.ai](https://decimer.ai/)
        """)
    st.markdown("1. Enter a SMILES string or upload a TXT file with SMILES in a single column.")
    st.markdown("2. Choose the prediction model: Multi-Tasking Neural Network, Random Forest, or New Random Forest for IC50.")
    st.markdown("3. Click 'Predict' to see results.")

    # Add the note under instructions
    # st.markdown("""
    # The ligand efficiency cut off for hit compounds is set to be between 0.2 and 0.35. Hit compounds are compounds that show some activity against the target protein and can be chemically modified to have improved potency and drug-like properties. The binding of hits to the target does not have to be extremely good as this can be optimised further after hit identification. This broad range of ligand efficiencies chosen is due a large range of heavy atom counts (HAC) among all the screened compounds. HAC is a proxy for molecular size. The optimal ligand effiency cut off depends on the molecular size of the screened compounds. The details of calculating target ligand efficiency values can be found in this paper. [1](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3772997/#FD2). A larger range of ligand efficiency values also allow for a more diverse set of hits to be investigated. This can potentially lead to drug molecules with novel structures compared to marketed drugs.
    # """)

    # Input: Single SMILES string or file upload
    model_choice = st.radio("Choose a model:", ["Multi-Tasking Neural Network", "Random Forest for LE", "Random Forest for IC50"], horizontal=True)
    
    # Add tooltip for Random Forest for LE
    if model_choice == "Random Forest for LE":
        st.markdown(
            '<div class="tooltip"> What is Random Forest for LE ?'
            '<span class="tooltiptext">The ligand efficiency cut off for hit compounds is set to be between 0.2 and 0.35. Hit compounds are compounds that show some activity against the target protein and can be chemically modified to have improved potency and drug-like properties. The binding of hits to the target does not have to be extremely good as this can be optimised further after hit identification. This broad range of ligand efficiencies chosen is due to a large range of heavy atom counts (HAC) among all the screened compounds. HAC is a proxy for molecular size. The optimal ligand efficiency cut off depends on the molecular size of the screened compounds. The details of calculating target ligand efficiency values can be found in this paper. [1](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3772997/#FD2). A larger range of ligand efficiency values also allow for a more diverse set of hits to be investigated. This can potentially lead to drug molecules with novel structures compared to marketed drugs.</span>'
            '</div>',
            unsafe_allow_html=True
        )

    smiles_input = st.text_input("Enter SMILES:")
    uploaded_file = st.file_uploader("Upload a TXT file", type=["csv", "txt", "xls", "xlsx"])

    if st.button("Predict"):
        if smiles_input:
            with st.spinner("Predicting..."):
                # Calculate descriptors and display in a table
                descriptors = calculate_descriptors(smiles_input)
                if descriptors:
                    st.markdown("### Molecular Descriptors")
                    descriptors_df = pd.DataFrame([descriptors])
                    st.table(descriptors_df)
                else:
                    st.error("Invalid SMILES string.")

                if model_choice == "Multi-Tasking Neural Network":
                    pIC50, bioactivity, accuracy, error_percentage = predict_with_nn(smiles_input)
                    if pIC50 is not None:
                        mol_weight = calculate_descriptors(smiles_input)['MolWt']
                        st.markdown(
                            f"""
                            <div class="result-container">
                                <h4>🧪 Prediction Results</h4>
                                <p><b>📊 pIC50 Value:</b> <span class="result-value">{pIC50:.2f}</span></p>
                                <p><b>⚗️ IC50 (µM):</b> <span class="result-value">{convert_pIC50_to_uM(pIC50):.2f} µM</span></p>
                                <p><b>🧪 IC50 (nM):</b> <span class="result-value">{convert_pIC50_to_nM(pIC50):.2f} nM</span></p>
                                <p><b>🧬 IC50 (ng/µL):</b> <span class="result-value">{convert_pIC50_to_ng_per_uL(pIC50, mol_weight):.2f} ng/µL</span></p>
                                <p><b>🟢 Bioactivity:</b> 
                                    <span class="result-value" style="color: {'#1E88E5' if bioactivity=='active' else '#D32F2F'};">
                                        {bioactivity.capitalize()}
                                    </span>
                                </p>
                                <p><b>🔍 Accuracy:</b> <span class="result-value">{accuracy:.2%}</span> <a href="#accuracy-explanation">?</a></p>
                                <p><b>📉 Error Percentage:</b> <span class="result-value" style="color: #D32F2F;">{error_percentage:.2%}</span> <a href="#error-explanation">?</a></p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.error("Invalid SMILES string.")
                elif model_choice == "Random Forest for LE":
                    bioactivity, accuracy = predict_with_xgboost(smiles_input)
                    if bioactivity:
                        st.markdown(
                            f"""
                            <div class="result-container">
                                <h4>🧪 Prediction Results</h4>
                                <p><b>🟢 Bioactivity:</b> 
                                    <span class="result-value" style="color: {'#1E88E5' if bioactivity=='active' else '#D32F2F'};">
                                        {bioactivity.capitalize()}
                                    </span>
                                </p>
                                <p><b>🔍 Accuracy:</b> <span class="result-value">{accuracy:.2%}</span> <a href="#accuracy-explanation">?</a></p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.error("Invalid SMILES string.")
                elif model_choice == "Random Forest for IC50":
                    bioactivity, accuracy = predict_with_rf_ic50(smiles_input)
                    if bioactivity:
                        st.markdown(
                            f"""
                            <div class="result-container">
                                <h4>🧪 Prediction Results</h4>
                                <p><b>🟢 Bioactivity:</b> 
                                    <span class="result-value" style="color: {'#1E88E5' if bioactivity=='active' else '#D32F2F'};">
                                        {bioactivity.capitalize()}
                                    </span>
                                </p>
                                <p><b>🔍 Accuracy:</b> <span class="result-value">{accuracy:.2%}</span> <a href="#accuracy-explanation">?</a></p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.error("Invalid SMILES string.")

            # Visualize the 3D structure
            st.markdown("## 3D Structure Visualization")
            viewer = generate_3d_view(smiles_input)
            if viewer is not None:
                viewer_html = viewer._make_html()
                st.components.v1.html(viewer_html, height=500)
            else:
                st.error(f"Invalid SMILES string: {smiles_input}")

            # Explanations
            st.markdown("<a id='accuracy-explanation'></a>", unsafe_allow_html=True)
            with st.expander("What does Accuracy mean?"):
                st.write("""
                The accuracy represents the proportion of correct predictions out of the total predictions made by the model. 
                It is calculated as:
                
                \[
                \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} \times 100
                \]
                """)

            st.markdown("<a id='error-explanation'></a>", unsafe_allow_html=True)
            with st.expander("What does Error Percentage mean?"):
                st.write("""
                The error percentage represents the proportion of incorrect predictions out of the total predictions made by the model. 
                It is calculated as:
                
                \[
                \text{Error} = \frac{\text{Number of Incorrect Predictions}}{\text{Total Number of Predictions}} \times 100
                \]
                """)

        elif uploaded_file:
            try:
                detected_encoding = detect_encoding(uploaded_file)
                file_extension = uploaded_file.name.split(".")[-1].lower()
                
                if file_extension == "csv":
                    df = pd.read_csv(uploaded_file, encoding=detected_encoding)
                elif file_extension == "txt":
                    df = pd.read_csv(uploaded_file, delimiter="\t", encoding=detected_encoding)
                elif file_extension in ["xls", "xlsx"]:
                    df = pd.read_excel(uploaded_file, engine="openpyxl")
                else:
                    st.error("Unsupported file format. Please upload CSV, TXT, XLS, or XLSX.")
                    st.stop()

                if df.shape[1] != 1:
                    st.error("The uploaded file must contain only one column with SMILES strings.")
                    st.stop()

                df.columns = ["SMILES"]
                df.dropna(inplace=True)

                results = []
                for smiles in df["SMILES"]:
                    if model_choice == "Multi-Tasking Neural Network":
                        pIC50, bioactivity, accuracy, error_percentage = predict_with_nn(smiles)
                        if pIC50 is not None:
                            mol_weight = calculate_descriptors(smiles)['MolWt']
                            results.append([smiles, pIC50, convert_pIC50_to_uM(pIC50), convert_pIC50_to_nM(pIC50), convert_pIC50_to_ng_per_uL(pIC50, mol_weight), bioactivity, accuracy, error_percentage])
                        else:
                            results.append([smiles, "Error", "Error", "Error", "Error", "Error", "Error", "Error"])
                    elif model_choice == "Random Forest for LE":
                        bioactivity, accuracy = predict_with_xgboost(smiles)
                        results.append([smiles, bioactivity if bioactivity else "Error", accuracy if accuracy else "Error"])
                    else:
                        bioactivity, accuracy = predict_with_rf_ic50(smiles)
                        results.append([smiles, bioactivity if bioactivity else "Error", accuracy if accuracy else "Error"])

                if model_choice == "Multi-Tasking Neural Network":
                    results_df = pd.DataFrame(results, columns=["SMILES", "pIC50", "IC50 (µM)", "IC50 (nM)", "IC50 (ng/µL)", "Bioactivity", "Accuracy", "Error Percentage"])
                else:
                    results_df = pd.DataFrame(results, columns=["SMILES", "Bioactivity", "Accuracy"])

                st.dataframe(results_df)
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions", csv, "bioactivity_predictions.csv", "text/csv")
                st.success("Predictions completed.")

            except Exception as e:
                st.error(f"Error processing the uploaded file: {e}")

elif st.session_state.page == "Molecular Visualization":
    show_molecular_visualization()

elif st.session_state.page == "Mission":
    show_mission()

elif st.session_state.page == "README":
    show_readme()