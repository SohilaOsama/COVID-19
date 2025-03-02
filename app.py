import streamlit as st
import pandas as pd
import chardet  # For automatic encoding detection
from neural_network_predictor import NeuralNetworkPredictor
from xgboost_predictor import XGBoostPredictor
from xgboost_ic50_predictor import XGBoostIC50Predictor
from about import show_about
from readme import show_readme
from mission import show_mission
from molecular_visualization import show_molecular_visualization, generate_3d_view, generate_2d_view
import concurrent.futures

# Initialize predictors
nn_predictor = NeuralNetworkPredictor()
xgboost_predictor = XGBoostPredictor()
xgboost_ic50_predictor = XGBoostIC50Predictor()

# Detect encoding of uploaded file
def detect_encoding(file):
    raw_data = file.read(4096)  # Read a small chunk
    file.seek(0)  # Reset file position
    result = chardet.detect(raw_data)  # Detect encoding
    return result["encoding"]

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
    st.markdown("""
    The ligand efficiency cut off for hit compounds is set to be between 0.2 and 0.35. Hit compounds are compounds that show some activity against the target protein and can be chemically modified to have improved potency and drug-like properties. The binding of hits to the target does not have to be extremely good as this can be optimised further after hit identification. This broad range of ligand efficiencies chosen is due a large range of heavy atom counts (HAC) among all the screened compounds. HAC is a proxy for molecular size. The optimal ligand effiency cut off depends on the molecular size of the screened compounds. The details of calculating target ligand efficiency values can be found in this paper. [1](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3772997/#FD2). A larger range of ligand efficiency values also allow for a more diverse set of hits to be investigated. This can potentially lead to drug molecules with novel structures compared to marketed drugs.
    """)

    # Input: Single SMILES string or file upload
    model_choice = st.radio("Choose a model:", ["Multi-Tasking Neural Network", "Random Forest", "New Random Forest for IC50"], horizontal=True)
    smiles_input = st.text_input("Enter SMILES:")
    uploaded_file = st.file_uploader("Upload a TXT file", type=["csv", "txt", "xls", "xlsx"])

    if st.button("Predict"):
        if smiles_input:
            with st.spinner("Predicting..."):
                # Calculate descriptors and display in a table
                descriptors = nn_predictor.calculate_descriptors(smiles_input)
                if descriptors:
                    st.markdown("### Molecular Descriptors")
                    descriptors_df = pd.DataFrame([descriptors])
                    st.table(descriptors_df)
                else:
                    st.error("Invalid SMILES string.")
                    st.stop()

                def predict():
                    if model_choice == "Multi-Tasking Neural Network":
                        return nn_predictor.predict(smiles_input)
                    elif model_choice == "Random Forest":
                        return xgboost_predictor.predict(smiles_input)
                    else:
                        return xgboost_ic50_predictor.predict(smiles_input)

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(predict)
                    result = future.result()

                if result:
                    if model_choice == "Multi-Tasking Neural Network":
                        pIC50, bioactivity, accuracy, error_percentage = result
                        if pIC50 is not None:
                            mol_weight = descriptors['MolWt']
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
                    else:
                        bioactivity, accuracy = result
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
                    def predict(smiles):
                        if model_choice == "Multi-Tasking Neural Network":
                            return nn_predictor.predict(smiles)
                        elif model_choice == "Random Forest":
                            return xgboost_predictor.predict(smiles)
                        else:
                            return xgboost_ic50_predictor.predict(smiles)

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(predict, smiles)
                        result = future.result()

                    if result:
                        if model_choice == "Multi-Tasking Neural Network":
                            pIC50, bioactivity, accuracy, error_percentage = result
                            if pIC50 is not None:
                                mol_weight = nn_predictor.calculate_descriptors(smiles)['MolWt']
                                results.append([smiles, pIC50, convert_pIC50_to_uM(pIC50), convert_pIC50_to_nM(pIC50), convert_pIC50_to_ng_per_uL(pIC50, mol_weight), bioactivity, accuracy, error_percentage])
                            else:
                                results.append([smiles, "Error", "Error", "Error", "Error", "Error", "Error", "Error"])
                        else:
                            bioactivity, accuracy = result
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