from flask import Flask, request, render_template
import random
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Expanded dataset with more SMILES strings
try:
    data = pd.read_csv('molecules.csv')
except FileNotFoundError:
    print("Error: molecules.csv not found! Using default data.")
    data = pd.DataFrame({
        'SMILES': [
            'CCC', 'CCO', 'CCN', 'CCCC', 'CCOC', 'CNC', 'CCCO', 'CCNC',
            'c1ccccc1', 'CC(=O)O', 'CN(C)C', 'CC(C)O', 'c1ccncc1', 'CC(=O)NC',
            'CCS', 'c1ccc(cc1)O', 'CC(C)C', 'COC(=O)C'
        ],
        'affinity': [
            8.5, 6.7, 7.9, 8.0, 7.2, 7.5, 6.9, 8.2, 7.8, 6.5, 7.0, 7.4, 8.1,
            6.8, 7.3, 7.7, 8.3, 7.6
        ],
        'toxicity': [
            1.2, 3.1, 0.9, 1.5, 2.0, 1.8, 2.5, 1.3, 1.0, 2.8, 1.7, 2.2, 0.8,
            2.4, 1.6, 1.1, 1.9, 1.4
        ]
    })


# Simple generative function to create "novel" SMILES
def generate_novel_smiles(base_smiles, num_variants=3):
    """Generate simple variations of a base SMILES string."""
    variants = set()
    building_blocks = ['C', 'O', 'N', 'c1ccccc1', 'C(=O)', 'CC']

    for _ in range(num_variants):
        # Randomly modify the base SMILES
        mod = random.choice(['add', 'replace'])
        if mod == 'add':
            new_smiles = base_smiles + random.choice(building_blocks)
        else:
            new_smiles = random.choice(building_blocks) + base_smiles
        # Validate SMILES
        if Chem.MolFromSmiles(new_smiles):
            variants.add(new_smiles)
    return list(variants)


# Extract features with error handling
def get_mol_weight(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Descriptors.MolWt(mol) if mol else 0
    except:
        return 0


data['mol_weight'] = data['SMILES'].apply(get_mol_weight)
data = data[data['mol_weight'] > 0]  # Filter invalid entries

# Train Random Forest models
X = data[['mol_weight']]
affinity_model = RandomForestRegressor(random_state=42).fit(
    X, data['affinity'])
toxicity_model = RandomForestRegressor(random_state=42).fit(
    X, data['toxicity'])


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        if request.method == 'POST':
            # Generate 7 candidates (mix of existing and novel)
            num_candidates = 7
            candidates = []

            # Pick some base molecules from dataset
            base_smiles = random.sample(data['SMILES'].tolist(),
                                        min(3, len(data['SMILES'])))

            # Generate novel SMILES from bases
            novel_smiles = []
            for base in base_smiles:
                novel_smiles.extend(generate_novel_smiles(base,
                                                          num_variants=2))

            # Combine with some existing SMILES
            existing_smiles = random.sample(data['SMILES'].tolist(),
                                            min(4, len(data['SMILES'])))
            all_smiles = list(set(novel_smiles +
                                  existing_smiles))[:num_candidates]

            for smiles in all_smiles:
                mol = Chem.MolFromSmiles(smiles)
                if not mol:
                    continue
                features = [Descriptors.MolWt(mol)]

                affinity = affinity_model.predict([features])[0]
                toxicity = toxicity_model.predict([features])[0]

                if affinity > 7 and toxicity < 2:
                    category = 'Potential Lead'
                elif affinity <= 7:
                    category = 'Ineffective'
                else:
                    category = 'Toxic'

                candidates.append({
                    'smiles': smiles,
                    'affinity': round(affinity, 2),
                    'toxicity': round(toxicity, 2),
                    'category': category
                })

            # Create scatter plot
            plt.figure(figsize=(10, 6))
            for cand in candidates:
                color = 'green' if cand[
                    'category'] == 'Potential Lead' else 'red' if cand[
                        'category'] == 'Toxic' else 'orange'
                plt.scatter(cand['affinity'],
                            cand['toxicity'],
                            c=color,
                            label=cand['category'],
                            s=100)
            plt.xlabel('Affinity', fontsize=12)
            plt.ylabel('Toxicity', fontsize=12)
            plt.title('Drug Candidates: Affinity vs Toxicity', fontsize=14)
            plt.legend()
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()

            return render_template('result.html',
                                   candidates=candidates,
                                   plot_url=plot_url)
        return render_template('index.html')
    except Exception as e:
        print(f"Error: {str(e)}")
        return "An error occurred. Please try again later.", 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
