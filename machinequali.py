import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut, ShuffleSplit, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, cohen_kappa_score, recall_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="App ML Qualitative",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Dark and Technological Theme from First Code ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400;500&display=swap');

    /* Main app styling */
    body {
        font-family: 'Roboto', sans-serif;
        color: #e0e6ed !important;
    }
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%) !important;
        border-right: 2px solid #333333;
        box-shadow: 5px 0 15px rgba(0, 255, 255, 0.1);
    }
    [data-testid="stSidebar"] .stFileUploader label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stButton button,
    [data-testid="stSidebar"] .stCheckbox label {
        color: #b0c4de !important;
        font-family: 'Roboto', sans-serif;
    }
    [data-testid="stSidebar"] .stButton button {
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border-radius: 5px;
    }

    /* Main title */
    h1 {
        color: #00ffff;
        text-align: center;
        font-family: 'Arial', monospace;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
        padding-top: 1.5rem;
    }

    /* Sub-headers */
    h2, h3, h4 {
        color: #00ffff;
        font-family: 'Arial', sans-serif;
        text-shadow: 0 0 8px rgba(0, 255, 255, 0.4);
    }

    /* Selectbox and multiselect styling */
    .stSelectbox, .stMultiSelect {
        background-color: rgba(15, 15, 35, 0.8) !important;
        border: 5px solid #333333;
        border-radius: 5px;
        padding: 10px;
    }

    /* Metric styling */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(15, 15, 35, 0.9) 0%, rgba(26, 26, 46, 0.9) 100%);
        border: 1px solid #333333;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }

    /* DataFrame styling */
    .stDataFrame {
        background: rgba(15, 15, 35, 0.8) !important;
        border: 1px solid rgba(0, 255, 255, 0.3);
    }

    /* Uploader styling in sidebar */
    [data-testid="stFileUploader"] {
        border: 2px dashed #333333;
        background-color: #1a1a2e;
        padding: 20px;
        border-radius: 10px;
    }

    /* Alert styling */
    .stAlert {
        border-radius: 0.5rem;
        background: rgba(15, 15, 35, 0.8);
        border: 1px solid #333333;
        color: #e0e6ed !important;
    }

    /* Uploaded file styling */
    .uploaded-file {
        color: #00ff88 !important;
        font-weight: bold;
    }

    /* Footer styling */
    .footer {
        font-size: 0.8rem;
        color: #b0c4de;
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
        background: linear-gradient(135deg, rgba(15, 15, 35, 0.9) 0%, rgba(26, 26, 46, 0.9) 100%);
        border: 1px solid #333333;
        border-radius: 5px;
    }

    /* Author info box */
    .author-info {
        background: linear-gradient(135deg, rgba(15, 15, 35, 0.9) 0%, rgba(26, 26, 46, 0.9) 100%);
        border: 2px solid #333333;
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 10px 30px rgba(0, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- Plotting Theme Configuration from First Code ---
plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': '#0c0c0c',
    'axes.facecolor': '#1a1a2e',
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#e0e6ed',
    'xtick.color': '#e0e6ed',
    'ytick.color': '#e0e6ed',
    'grid.color': '#4a5568',
    'text.color': '#e0e6ed',
    'legend.facecolor': '#0f0f23',
    'legend.edgecolor': '#333333'
})

# Titre principal
st.title("🔍 Application Machine Learning pour Variables Qualitatives")
st.markdown("<p style='text-align: center; color: #b0c4de; font-family: Roboto, sans-serif;'>Une plateforme interactive pour l'entraînement et l'évaluation de modèles de classification binaire.</p>", unsafe_allow_html=True)

# --- Sidebar for File Upload and Configuration ---
with st.sidebar:
    st.header("📁 Chargement du fichier")
    uploaded_file = st.file_uploader(
        "Téléchargez votre fichier CSV ou Excel",
        type=["csv", "xlsx"],
        help="Formats supportés : CSV, Excel (.xlsx)"
    )

    if uploaded_file is not None:
        st.subheader("⚙️ Options CSV")
        header = st.checkbox("Première ligne comme en-tête", value=True)
        sep = st.selectbox("Séparateur de colonnes", [",", ";", "\t"], index=1)
        dec = st.selectbox("Séparateur décimal", [".", ","], index=0)

    # Author Information from First Code
    st.markdown("---")
    st.markdown("""
    <div class="author-info">
        <h4>🧾 À propos de l'auteur</h4>
        <p><b>Nom:</b> N'dri</p>
        <p><b>Prénom:</b> Abo Onesime</p>
        <p><b>Rôle:</b> Data Analyst / Scientist</p>
        <p><b>Téléphone:</b> 07-68-05-98-87 / 01-01-75-11-81</p>
        <p><b>Email:</b> <a href="mailto:ndriablatie123@gmail.com" style="color:#00ff88;">ndriablatie123@gmail.com</a></p>
        <p><b>LinkedIn:</b> <a href="https://www.linkedin.com/in/abo-onesime-n-dri-54a537200/" target="_blank" style="color:#00ff88;">Profil LinkedIn</a></p>
        <p><b>GitHub:</b> <a href="https://github.com/Aboonesime" target="_blank" style="color:#00ff88;">Mon GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)

# --- Main Content ---
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(
                uploaded_file,
                sep=sep,
                decimal=dec,
                header=0 if header else None,
                engine='python'
            )
        else:
            df = pd.read_excel(uploaded_file)
        st.success(f"Fichier chargé: **{uploaded_file.name}**")
        st.write(f"🔍 **{len(df)}** observations, **{len(df.columns)}** variables")
        st.markdown(f"<p class='uploaded-file'>Fichier chargé: {uploaded_file.name}</p>", unsafe_allow_html=True)
        st.write("Aperçu des données :", df.head())

        # --- 2. Sélection de la variable cible ---
        st.header("🎯 Sélection de la variable cible")
        target_col = st.selectbox("Sélectionnez la variable cible (binaire 0/1)", df.columns)
        if df[target_col].nunique() != 2:
            st.error("⚠️ La variable cible doit être binaire (valeurs 0 et 1 uniquement).")
            st.stop()
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # --- 3. Première Partie : Transformation des Données ---
        st.header("⚙️ 1. Transformation des données")
        transformations = st.multiselect(
            "Sélectionnez une ou deux transformations",
            ["BoxCox", "Centrer", "Réduire", "MinMaxScaler", "Standardisation", "Binarisation", "Transfo ACP"],
            max_selections=2
        )
        show_code_transfo = st.checkbox("Afficher le code de transformation", key="transfo_code")

        def apply_transformations(X, transformations):
            steps = []
            for transfo in transformations:
                if transfo == "BoxCox":
                    pt = PowerTransformer(method='box-cox')
                    steps.append(('boxcox', pt))
                elif transfo == "Centrer":
                    steps.append(('center', StandardScaler(with_std=False)))
                elif transfo == "Réduire":
                    steps.append(('scale', StandardScaler(with_mean=False)))
                elif transfo == "MinMaxScaler":
                    steps.append(('minmax', MinMaxScaler()))
                elif transfo == "Standardisation":
                    steps.append(('standardize', StandardScaler()))
                elif transfo == "Binarisation":
                    steps.append(('binarize', Binarizer()))
                elif transfo == "Transfo ACP":
                    steps.append(('pca', PCA(n_components=0.95)))
            pipe = Pipeline(steps)
            X_trans = pipe.fit_transform(X)
            return X_trans

        if transformations:
            X_trans = apply_transformations(X, transformations)
            st.write("📊 Résumé statistique après transformation :")
            st.dataframe(pd.DataFrame(X_trans).describe())
            if show_code_transfo:
                imports_needed = []
                steps_code = []
                for t in transformations:
                    if t == "BoxCox":
                        imports_needed.append("PowerTransformer")
                        steps_code.append("('boxcox', PowerTransformer(method='box-cox'))")
                    elif t == "Centrer":
                        imports_needed.append("StandardScaler")
                        steps_code.append("('center', StandardScaler(with_std=False))")
                    elif t == "Réduire":
                        imports_needed.append("StandardScaler")
                        steps_code.append("('scale', StandardScaler(with_mean=False))")
                    elif t == "MinMaxScaler":
                        imports_needed.append("MinMaxScaler")
                        steps_code.append("('minmax', MinMaxScaler())")
                    elif t == "Standardisation":
                        imports_needed.append("StandardScaler")
                        steps_code.append("('standardize', StandardScaler())")
                    elif t == "Binarisation":
                        imports_needed.append("Binarizer")
                        steps_code.append("('binarize', Binarizer())")
                    elif t == "Transfo ACP":
                        imports_needed.append("PCA")
                        steps_code.append("('pca', PCA(n_components=0.95))")
                unique_imports = list(set(imports_needed))
                st.code(f"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import {', '.join(unique_imports)}

pipe = Pipeline([
    {', '.join(steps_code)}
])

X_trans = pipe.fit_transform(X)
                """)

        # --- 4. Deuxième Partie : Méthode de Rééchantillonnage ---
        st.header("🔄 2. Méthode de Rééchantillonnage")
        resampling_method = st.selectbox(
            "Choisissez une méthode de rééchantillonnage",
            [
                "Échantillonnage d'apprentissage et échantillonnage de test",
                "Échantillonnage par bootstrap",
                "Validation croisée K-fold",
                "Validation croisée K-fold répétée",
                "Validation croisée Leave one out"
            ]
        )
        show_code_resamp = st.checkbox("Afficher le code de rééchantillonnage", key="resamp_code")

        def apply_resampling(X_trans, y, method):
            if method == "Échantillonnage d'apprentissage et échantillonnage de test":
                return train_test_split(X_trans, y, test_size=0.2, stratify=y, random_state=42)
            elif method == "Échantillonnage par bootstrap":
                rs = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                for train_index, test_index in rs.split(X_trans):
                    return X_trans[train_index], X_trans[test_index], y.iloc[train_index], y.iloc[test_index]
            elif method == "Validation croisée K-fold":
                skf = StratifiedKFold(n_splits=5)
                return [(i, (train_idx, test_idx)) for i, (train_idx, test_idx) in enumerate(skf.split(X_trans, y))]
            elif method == "Validation croisée K-fold répétée":
                rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)
                return [(i, (train_idx, test_idx)) for i, (train_idx, test_idx) in enumerate(rskf.split(X_trans, y))]
            elif method == "Validation croisée Leave one out":
                loo = LeaveOneOut()
                return [(i, (train_idx, test_idx)) for i, (train_idx, test_idx) in enumerate(loo.split(X_trans, y))]

        if transformations:
            result = apply_resampling(X_trans, y, resampling_method)
            st.success("✅ Rééchantillonnage appliqué.")
            if show_code_resamp:
                st.code(f"""
from sklearn.model_selection import { 
                    'train_test_split' if resampling_method == "Échantillonnage d'apprentissage et échantillonnage de test" else
                    'ShuffleSplit' if resampling_method == "Échantillonnage par bootstrap" else
                    'StratifiedKFold' if resampling_method == "Validation croisée K-fold" else
                    'RepeatedStratifiedKFold' if resampling_method == "Validation croisée K-fold répétée" else
                    'LeaveOneOut'
                }
# Exemple de code pour la méthode sélectionnée...
                """)

        # --- 5. Troisième Partie : Comparaison des algorithmes ---
        st.header("📊 3. Comparaison des algorithmes")
        if st.button("🚀 Lancer l'évaluation des modèles"):
            models = {
                "Analyse Discrimination linéaire": LinearDiscriminantAnalysis(),
                "Régression logistique": LogisticRegression(max_iter=1000),
                "Régression logistique pénalisée Ridge": LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000),
                "Régression logistique pénalisée Lasso": LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000),
                "Régression logistique pénalisée Elastic-Net": LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000),
                "KNN": KNeighborsClassifier(),
                "SVM": SVC(probability=True),
                "Arbre de décision": DecisionTreeClassifier(),
                "Naïve Bayes": GaussianNB(),
                "Bagging": BaggingClassifier(n_estimators=10),
                "Random Forest": RandomForestClassifier(n_estimators=100),
                "Boosting": GradientBoostingClassifier(n_estimators=100)
            }

            results = []
            for name, model in models.items():
                scores = cross_val_score(model, X_trans, y, cv=5, scoring='accuracy')
                model.fit(X_trans, y)
                y_pred = model.predict(X_trans)
                y_proba = model.predict_proba(X_trans)[:, 1] if hasattr(model, "predict_proba") else [0]*len(y_pred)
                cm = confusion_matrix(y, y_pred)
                acc = accuracy_score(y, y_pred)
                auc = roc_auc_score(y, y_proba) if len(np.unique(y_proba)) > 1 else 0
                kappa = cohen_kappa_score(y, y_pred)
                sens = recall_score(y, y_pred)
                spec = precision_score(y, y_pred)
                results.append({
                    "Modèle": name,
                    "Accuracy": acc,
                    "AUC-ROC": auc,
                    "Kappa": kappa,
                    "Sensibilité": sens,
                    "Spécificité": spec,
                    "Confusion Matrix": cm,
                    "Predictions": y_pred
                })

            # Stocker les résultats dans session_state
            st.session_state['results'] = results
            st.session_state['X_trans'] = X_trans
            st.session_state['y'] = y
            st.session_state['target_col'] = target_col

            results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
            st.dataframe(results_df[["Modèle", "Accuracy", "AUC-ROC", "Kappa", "Sensibilité", "Spécificité"]])
            best_model = results_df.iloc[0]['Modèle']
            st.success(f"🏆 Meilleur modèle : **{best_model}** avec une précision de **{results_df.iloc[0]['Accuracy']:.2f}**")

            # Affichage de la matrice de confusion du meilleur modèle
            best_cm = results_df.iloc[0]["Confusion Matrix"]
            fig, ax = plt.subplots(figsize=(5, 5))
            sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Prédiction')
            ax.set_ylabel('Réalité')
            st.pyplot(fig)

            # SHAP values pour expliquer le meilleur modèle
            st.subheader("Explication SHAP du meilleur modèle")
            try:
                if best_model == "SVM":
                    # Use KernelExplainer for SVC
                    explainer = shap.KernelExplainer(models[best_model].predict_proba, X_trans)
                    shap_values = explainer.shap_values(X_trans, nsamples=100)
                    shap.summary_plot(shap_values[1], X_trans, plot_type="bar", show=False)
                else:
                    explainer = shap.Explainer(models[best_model], X_trans)
                    shap_values = explainer(X_trans)
                    shap.summary_plot(shap_values, X_trans, plot_type="bar", show=False)
                st.pyplot(plt.gcf())
            except Exception as e:
                st.warning(f"⚠️ Impossible de générer l'explication SHAP : {e}")

        # --- 6. Quatrième Partie : Affichage des valeurs réelles et prédites ---
        st.header("📊 4. Valeurs réelles et prédites par algorithme")
        if st.button("🚀 Afficher les valeurs réelles et prédites"):
            if 'results' in st.session_state:
                results = st.session_state['results']
                y = st.session_state['y']
                target_col = st.session_state['target_col']

                # Create a DataFrame starting with the target variable
                comparison_df = pd.DataFrame({target_col: y})

                # Add predictions from each model
                for result in results:
                    model_name = result["Modèle"]
                    y_pred = result["Predictions"]
                    comparison_df[model_name] = y_pred

                st.write("Tableau des valeurs réelles et prédites :")
                st.dataframe(comparison_df)
            else:
                st.warning("⚠️ Vous devez d'abord lancer l'évaluation des modèles.")

    except Exception as e:
        st.error(f"❌ Erreur lors du traitement : {e}")

else:
    st.info("📥 Veuillez charger un fichier CSV ou Excel pour commencer.")

# --- Instructions ---
st.sidebar.markdown("---")
st.sidebar.info("""
**Instructions:**
1. Uploader un fichier CSV ou Excel.
2. Sélectionner la variable cible (binaire, 0/1).
3. Choisir jusqu'à deux transformations des données.
4. Sélectionner une méthode de rééchantillonnage.
5. Lancer l'évaluation des modèles pour comparer les performances.
6. Afficher les valeurs réelles et prédites.
""")

# --- Footer ---
st.markdown("""
<div class="footer">
    © 2025 Abo Onesime N'dri | Développé avec ❤️ en Python/Streamlit
</div>
""", unsafe_allow_html=True)
