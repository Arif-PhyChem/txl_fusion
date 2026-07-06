import json
import math
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from pymatgen.core import Composition, Element
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

try:
    from num2words import num2words
except ModuleNotFoundError:
    sys.path.insert(0, '/path/to/3_classes_classification/weighted_version/class_imbalance_sensitivity_test')
    from num2words import num2words

ROOT = Path('/path/to/3_classes_classification')
OUT = Path(__file__).resolve().parent
SPLIT_MANIFEST_PATH = ROOT / 'label_noise_analysis' / 'shared_split_manifest.json'
TRAINING_DATA_PATH = ROOT / 'label_noise_analysis' / 'clean_training_data.json'
CLEAN_TOPOGIVITY_DIR = ROOT / 'weighted_version' / 'gm_three_class_baseline' / 'weighted_topogivity_artifacts'
SEED = 42
LABEL_ORDER = ['semimetal', 'topological', 'trivial']


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def dump_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def normalize_label(raw: str) -> str:
    label = raw.strip().lower()
    return {'sm': 'semimetal', 'tsm': 'semimetal', 'ti': 'topological'}.get(label, label)


def normalize_element_symbol(symbol: str) -> str:
    return 'H' if symbol == 'D' else symbol


def normalize_formula(formula: str) -> str:
    comp = Composition(formula)
    normalized = defaultdict(float)
    for el, amt in comp.get_el_amt_dict().items():
        normalized[normalize_element_symbol(el)] += amt
    return Composition(normalized).formula.replace(' ', '')



def compute_topogivity_scores(formula: str, trivial_topogivities, sm_topogivities):
    comp = Composition(normalize_formula(formula))
    el_amt = comp.get_el_amt_dict()
    total_atoms = sum(el_amt.values())
    trivial_g_m = 0.0
    sm_g_m = 0.0
    for el, amt in el_amt.items():
        normalized_el = normalize_element_symbol(el)
        fraction = amt / total_atoms
        trivial_tau_e = trivial_topogivities.get(normalized_el)
        sm_tau_e = sm_topogivities.get(normalized_el)
        if trivial_tau_e is not None:
            trivial_g_m += fraction * trivial_tau_e
        if sm_tau_e is not None:
            sm_g_m += fraction * sm_tau_e
    return trivial_g_m, sm_g_m

def record_key(entry):
    return {
        'compound': normalize_formula(entry['compoundName']),
        'space_group': int(entry['symmetryGroupNumber']),
        'label': normalize_label(entry['topologicalClassificationShortDescription']),
    }


def evaluate(y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    report['accuracy'] = float(accuracy_score(y_true, y_pred))
    report['macro_f1'] = float(f1_score(y_true, y_pred, labels=[0, 1, 2], average='macro', zero_division=0))
    report['weighted_f1'] = float(f1_score(y_true, y_pred, labels=[0, 1, 2], average='weighted', zero_division=0))
    report['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    return report


def class_weights(y):
    counts = Counter(map(int, y))
    total = len(y)
    return {cls: total / (len(counts) * count) for cls, count in counts.items()}


category_map = {
    'Alkali_metal': ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr'],
    'Alkaline_earth_metal': ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra'],
    'Nonmetal': ['H', 'C', 'N', 'O', 'P', 'S', 'Se'],
    'Halogen': ['F', 'Cl', 'Br', 'I', 'At', 'Ts'],
    'Noble_gas': ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn', 'Og'],
    'Transition_metal': [
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn'
    ],
    'Metal': ['Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi', 'Nh', 'Fl', 'Mc', 'Lv'],
    'Lanthanide': ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'],
    'Actinide': ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'],
    'Metalloid': ['B', 'Si', 'Ge', 'As', 'Sb', 'Te', 'Po'],
}
all_categories = list(category_map.keys())


def symbol_to_name(symbol):
    elements = {
        'H': 'Hydrogen', 'He': 'Helium', 'Li': 'Lithium', 'Be': 'Beryllium', 'B': 'Boron',
        'C': 'Carbon', 'N': 'Nitrogen', 'O': 'Oxygen', 'F': 'Fluorine', 'Ne': 'Neon',
        'Na': 'Sodium', 'Mg': 'Magnesium', 'Al': 'Aluminum', 'Si': 'Silicon', 'P': 'Phosphorus',
        'S': 'Sulfur', 'Cl': 'Chlorine', 'Ar': 'Argon', 'K': 'Potassium', 'Ca': 'Calcium',
        'Sc': 'Scandium', 'Ti': 'Titanium', 'V': 'Vanadium', 'Cr': 'Chromium', 'Mn': 'Manganese',
        'Fe': 'Iron', 'Co': 'Cobalt', 'Ni': 'Nickel', 'Cu': 'Copper', 'Zn': 'Zinc',
        'Ga': 'Gallium', 'Ge': 'Germanium', 'As': 'Arsenic', 'Se': 'Selenium', 'Br': 'Bromine',
        'Kr': 'Krypton', 'Rb': 'Rubidium', 'Sr': 'Strontium', 'Y': 'Yttrium', 'Zr': 'Zirconium',
        'Nb': 'Niobium', 'Mo': 'Molybdenum', 'Tc': 'Technetium', 'Ru': 'Ruthenium', 'Rh': 'Rhodium',
        'Pd': 'Palladium', 'Ag': 'Silver', 'Cd': 'Cadmium', 'In': 'Indium', 'Sn': 'Tin',
        'Sb': 'Antimony', 'Te': 'Tellurium', 'I': 'Iodine', 'Xe': 'Xenon', 'Cs': 'Cesium',
        'Ba': 'Barium', 'La': 'Lanthanum', 'Ce': 'Cerium', 'Pr': 'Praseodymium', 'Nd': 'Neodymium',
        'Pm': 'Promethium', 'Sm': 'Samarium', 'Eu': 'Europium', 'Gd': 'Gadolinium', 'Tb': 'Terbium',
        'Dy': 'Dysprosium', 'Ho': 'Holmium', 'Er': 'Erbium', 'Tm': 'Thulium', 'Yb': 'Ytterbium',
        'Lu': 'Lutetium', 'Hf': 'Hafnium', 'Ta': 'Tantalum', 'W': 'Tungsten', 'Re': 'Rhenium',
        'Os': 'Osmium', 'Ir': 'Iridium', 'Pt': 'Platinum', 'Au': 'Gold', 'Hg': 'Mercury',
        'Tl': 'Thallium', 'Pb': 'Lead', 'Bi': 'Bismuth', 'Po': 'Polonium', 'At': 'Astatine',
        'Rn': 'Radon', 'Fr': 'Francium', 'Ra': 'Radium', 'Ac': 'Actinium', 'Th': 'Thorium',
        'Pa': 'Protactinium', 'U': 'Uranium', 'Np': 'Neptunium', 'Pu': 'Plutonium', 'Am': 'Americium',
        'Cm': 'Curium', 'Bk': 'Berkelium', 'Cf': 'Californium', 'Es': 'Einsteinium', 'Fm': 'Fermium',
        'Md': 'Mendelevium', 'No': 'Nobelium', 'Lr': 'Lawrencium', 'Rf': 'Rutherfordium', 'Db': 'Dubnium',
        'Sg': 'Seaborgium', 'Bh': 'Bohrium', 'Hs': 'Hassium', 'Mt': 'Meitnerium', 'Ds': 'Darmstadtium',
        'Rg': 'Roentgenium', 'Cn': 'Copernicium', 'Nh': 'Nihonium', 'Fl': 'Flerovium', 'Mc': 'Moscovium',
        'Lv': 'Livermorium', 'Ts': 'Tennessine', 'Og': 'Oganesson'
    }
    return elements.get(symbol.title(), 'Unknown')


def compute_ionic_character(el_amt):
    elements = list(el_amt.keys())
    if len(elements) <= 1:
        return None
    for el in elements:
        normalized_el = normalize_element_symbol(el)
        category = next((cat for cat, els in category_map.items() if normalized_el in els), 'unknown')
        if category == 'Noble_gas':
            return None
    differences = []
    for i, a in enumerate(elements):
        for b in elements[i + 1:]:
            try:
                en_diff = abs(Element(normalize_element_symbol(a)).X - Element(normalize_element_symbol(b)).X)
                differences.append(en_diff)
            except Exception:
                continue
    return sum(differences) / len(differences) if differences else 0.0



def describe_ionic_character(value):
    if value is None:
        return 'undefined'
    if value < 0.4:
        return 'very_covalent'
    if value < 1.0:
        return 'mostly_covalent'
    if value < 2.0:
        return 'moderately_ionic'
    return 'highly_ionic'

def get_element_category(element):
    normalized = normalize_element_symbol(element)
    for category, elements in category_map.items():
        if normalized in elements:
            return category
    return 'unknown'


def extract_valence_orbitals(electronic_structure, element_symbol):
    import re
    orbitals = re.findall(r'(\d+)([spdf])(\d+)', electronic_structure)
    if not orbitals:
        return defaultdict(int)
    orbital_info = [(int(n), l, int(e)) for n, l, e in orbitals]
    max_n = max(n for n, _, _ in orbital_info)
    category = get_element_category(element_symbol)
    valence = defaultdict(int)
    for n, l, e in orbital_info:
        if n == max_n:
            valence[l] += e
        elif category == 'Transition_metal' and l == 'd' and n == max_n - 1:
            valence[l] += e
        elif category in ['Lanthanide', 'Actinide']:
            if l == 'f' and n == max_n - 2:
                valence[l] += e
            if l == 'd' and n == max_n - 1 and e > 0:
                valence[l] += e
    return valence


def prepare_category_wise_el_list(category_list):
    total_atoms = 0.0
    category_dict = {cat: 0.0 for cat in all_categories}
    parsed_list = []
    for item in category_list.split(','):
        item = item.strip()
        parts = item.split(' ', 1)
        value = float(parts[0])
        category = parts[1].strip()
        parsed_list.append((value, category))
        total_atoms += value
    for val, cat in parsed_list:
        ratio = val / total_atoms if total_atoms > 0 else 0
        category_dict[cat] = ratio
    return category_dict



def build_space_group_priors(entries):
    sg_class_counts = defaultdict(lambda: defaultdict(int))
    sg_total_counts = defaultdict(int)
    for entry in entries:
        sg_key = str(int(round(entry.get('symmetryGroupNumber', 0))))
        label = normalize_label(entry['topologicalClassificationShortDescription'])
        sg_class_counts[sg_key][label] += 1
        sg_total_counts[sg_key] += 1
    return sg_class_counts, sg_total_counts


def get_sg_probabilities(sym_group, sg_class_counts, sg_total_counts):
    sg_key = str(int(round(sym_group)))
    total_sg = sg_total_counts.get(sg_key, 0)
    if total_sg <= 0:
        return 0.0, 0.0, 0.0, 0
    counts = sg_class_counts[sg_key]
    return (
        counts.get('trivial', 0) / total_sg,
        counts.get('semimetal', 0) / total_sg,
        counts.get('topological', 0) / total_sg,
        int(total_sg),
    )

def build_feature_rows(compounds, sg_class_counts, sg_total_counts, trivial_topogivities, sm_topogivities):
    features = []
    for idx, entry in enumerate(compounds):
        try:
            formula = normalize_formula(entry['compoundName'])
            sym_group = round(entry.get('symmetryGroupNumber', 0))
            label = normalize_label(entry['topologicalClassificationShortDescription'])
            prob_trivial, prob_sm, prob_ti, sg_prior_support = get_sg_probabilities(sym_group, sg_class_counts, sg_total_counts)

            comp = Composition(formula)
            el_amt = comp.get_el_amt_dict()
            total_atoms = sum(el_amt.values())
            total_electrons = sum(Element(normalize_element_symbol(el)).Z * amt for el, amt in el_amt.items())
            ionic_value = compute_ionic_character(el_amt)
            trivial_g_m, sm_g_m = compute_topogivity_scores(formula, trivial_topogivities, sm_topogivities)

            valence_totals = defaultdict(float)
            categories = Counter()
            for el, amt in el_amt.items():
                normalized_el = normalize_element_symbol(el)
                elem = Element(normalized_el)
                category = next((cat for cat, els in category_map.items() if normalized_el in els), 'unknown')
                _ = symbol_to_name(normalized_el)
                _ = num2words(int(amt)).replace('-', ' ')
                valence_orbitals = extract_valence_orbitals(elem.electronic_structure, normalized_el)
                for orb, count in valence_orbitals.items():
                    valence_totals[orb] += count * amt
                categories[category] += amt

            orbital_order = ['s', 'p', 'd', 'f']
            mean_valence = {
                orb: round(valence_totals.get(orb, 0.0) / total_atoms, 2) if orb in valence_totals else 0.0
                for orb in orbital_order
            }
            d_present = 1 if mean_valence['d'] > 0 else 0
            f_present = 1 if mean_valence['f'] > 0 else 0
            category_summary = ', '.join(f"{count} {cat}" for cat, count in categories.items())
            category_dict = prepare_category_wise_el_list(category_summary)

            bonding_nature = describe_ionic_character(ionic_value)

            feat = {
                'SG': sym_group,
                'Total_electrons': total_electrons,
                'Mean_s_valence_electrons': mean_valence['s'],
                'Mean_p_valence_electrons': mean_valence['p'],
                'Mean_d_valence_electrons': mean_valence['d'],
                'Mean_f_valence_electrons': mean_valence['f'],
                'Is_d_val_electrons_present?': d_present,
                'Is_f_val_electrons_present?': f_present,
                'Is_d_and_f_val_electrons_present?': int(d_present == 1 and f_present == 1),
                'Is_total_electrons_even?': int(total_electrons % 2 == 0),
                'Ionic_value': ionic_value,
                'Bonding_is_undefined': int(bonding_nature == 'undefined'),
                'Bonding_is_very_covalent': int(bonding_nature == 'very_covalent'),
                'Bonding_is_mostly_covalent': int(bonding_nature == 'mostly_covalent'),
                'Bonding_is_moderately_ionic': int(bonding_nature == 'moderately_ionic'),
                'Bonding_is_highly_ionic': int(bonding_nature == 'highly_ionic'),
                'Trivial_SG_prob': prob_trivial,
                'SM_SG_prob': prob_sm,
                'TI_SG_prob': prob_ti,
                'SG_prior_support': sg_prior_support,
                'Is_trivial_SG_prob_zero': int(prob_trivial == 0.0),
                'Is_SM_SG_prob_zero': int(prob_sm == 0.0),
                'Is_TI_SG_prob_zero': int(prob_ti == 0.0),
                'Trivial_g': trivial_g_m,
                'SM_g': sm_g_m,
                'Is_trivial_g_positive': int(trivial_g_m > 0.0),
                'Is_sm_g_positive': int(sm_g_m > 0.0),
                'Are_both_g_negative': int(trivial_g_m < 0.0 and sm_g_m < 0.0),
                'Is_trivial_positive_sm_negative': int(trivial_g_m > 0.0 and sm_g_m < 0.0),
                'Is_sm_positive_trivial_negative': int(sm_g_m > 0.0 and trivial_g_m < 0.0),
                'label': label,
            }
            for k, v in category_dict.items():
                feat[k] = round(v * 100, 2)
            features.append(feat)
        except Exception as exc:
            print(f'[ERROR] Entry index {idx}: {exc}')
            traceback.print_exc()
            raise
    return features


def load_shared_split(compounds):
    manifest = load_json(SPLIT_MANIFEST_PATH)
    canonical_manifest = manifest
    if manifest.get('shared_split_path'):
        canonical_manifest = load_json(Path(manifest['shared_split_path']))
    current_keys = [record_key(entry) for entry in compounds]
    if current_keys != canonical_manifest.get('record_keys', current_keys):
        raise ValueError(f'Training data does not match split manifest: {canonical_manifest}')
    return np.asarray(manifest['train_indices']), np.asarray(manifest['validation_indices'])


def build_model(device):
    return xgb.XGBClassifier(
        device=device,
        tree_method='hist',
        max_depth=4,
        learning_rate=0.01,
        n_estimators=1000,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=0.2,
        reg_alpha=0.2,
        reg_lambda=2.0,
        eval_metric='mlogloss',
        early_stopping_rounds=20,
        random_state=SEED,
    )


def plot_feature_importance(model, feature_names, output_json, output_pdf):
    import matplotlib.pyplot as plt
    import seaborn as sns
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    feature_importance_data = [
        {'feature': feat, 'importance': float(imp)}
        for feat, imp in zip(top_features, top_importances)
    ]
    dump_json(output_json, feature_importance_data)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=top_importances, y=top_features, palette='viridis')
    plt.title('Weighted XGB Feature Importance', fontsize=14)
    plt.xlabel('Score', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    plt.close()


def main():
    compounds = load_json(TRAINING_DATA_PATH)
    trivial_topogivities = load_json(CLEAN_TOPOGIVITY_DIR / 'trivial_topogivities.json')
    sm_topogivities = load_json(CLEAN_TOPOGIVITY_DIR / 'sm_topogivities.json')
    train_idx, val_idx = load_shared_split(compounds)
    subtrain_entries = [compounds[i] for i in train_idx]
    sg_class_counts, sg_total_counts = build_space_group_priors(subtrain_entries)
    features = build_feature_rows(compounds, sg_class_counts, sg_total_counts, trivial_topogivities, sm_topogivities)
    dump_json(OUT / 'xgb_data_weighted_shared_split.json', features)
    dump_json(
        OUT / 'sg_priors_from_shared_subtrain.json',
        {
            'space_group_class_counts': {sg: dict(counts) for sg, counts in sg_class_counts.items()},
            'space_group_total_counts': dict(sg_total_counts),
            'source': str(TRAINING_DATA_PATH),
            'split_manifest': str(SPLIT_MANIFEST_PATH),
            'n_subtraining_records': int(len(train_idx)),
        },
    )

    df = pd.DataFrame(features)
    X = df.drop(columns='label')
    y = df['label']

    le = LabelEncoder()
    le.fit(['topological', 'semimetal', 'trivial'])
    y_encoded = le.transform(y)
    label_to_code = {label: code for label, code in zip(le.classes_, range(len(le.classes_)))}
    dump_json(OUT / 'label_to_code.json', label_to_code)

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_train = y_encoded[train_idx]
    y_val = y_encoded[val_idx]

    feature_names = X.columns.tolist()
    dump_json(OUT / 'xgb_feature_names.json', {'xgb_feature_names': feature_names})

    weights = class_weights(y_train)
    sample_weights = np.asarray([weights[int(label)] for label in y_train], dtype=float)
    dump_json(OUT / 'class_weights.json', {str(k): v for k, v in weights.items()})

    model = build_model('cuda')
    try:
        model.fit(X_train, y_train, sample_weight=sample_weights, eval_set=[(X_val, y_val)], verbose=False)
    except Exception:
        print('Falling back to CPU...')
        model = build_model('cpu')
        model.fit(X_train, y_train, sample_weight=sample_weights, eval_set=[(X_val, y_val)], verbose=False)

    # Switch the fitted booster to CPU for validation-time prediction so
    # pandas/numpy inputs do not trigger XGBoost's device-mismatch fallback warning.
    model.get_booster().set_param({'device': 'cpu'})
    model.set_params(device='cpu')

    preds = model.predict(X_val)
    report = evaluate(y_val, preds)
    dump_json(OUT / 'validation_metrics.json', report)
    print('Validation accuracy:', report['accuracy'])
    print('Validation macro-F1:', report['macro_f1'])
    print(classification_report(y_val, preds, target_names=LABEL_ORDER, zero_division=0))

    model.get_booster().save_model(OUT / 'xgb_model.json')
    plot_feature_importance(model, feature_names, OUT / 'xgb_features_data.json', OUT / 'xgb_features_importance.pdf')


if __name__ == '__main__':
    main()
