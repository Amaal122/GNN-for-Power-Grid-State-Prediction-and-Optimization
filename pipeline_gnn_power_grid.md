# GNN for Power Grid State Prediction and Optimization — Pipeline du projet

## 1. Objectif

Construire un **Graph Neural Network (GNN)** qui exploite la topologie d'un réseau électrique (bus = nœuds, lignes = arêtes) pour **prédire rapidement l'état électrique** du réseau (tensions, angles, flux de puissance), puis utiliser cette prédiction dans une étape d'**optimisation (OPF)**.

Question scientifique visée :
> *Un GNN entraîné peut-il approximer l'état d'un système électrique avec une précision acceptable, en étant beaucoup plus rapide qu'un solveur classique ?*

## 2. Vue d'ensemble du pipeline

```
IEEE 14-bus (MATPOWER)
        │
        ▼
Paramètres réseau (loads, generators, lines, impédances, limites)
        │
        ▼
Génération de scénarios (variation aléatoire des charges)
        │
        ▼
Power Flow Solver (MATPOWER / pandapower)  →  Ground Truth (Vm, Va, Pg, Qg, flux)
        │
        ▼
Construction du graphe (nodes = bus, edges = lignes)
        │
        ▼
Feature engineering (node features + edge features)
        │
        ▼
GNN (GCN puis GAT) — PyTorch Geometric
        │
        ▼
Prédiction de l'état (Vm, Va)
        │
        ▼
Évaluation (MAE, RMSE, R², comparaison au solveur)
        │
        ▼
Optimisation (OPF) — bonus
```

## 3. Stack technique

| Domaine | Outils |
|---|---|
| Langage | Python 3.10+ |
| Systèmes électriques | MATPOWER, pandapower, cas IEEE 14-bus / 30-bus |
| Machine Learning | PyTorch |
| Graph Learning | PyTorch Geometric (GCNConv, GATConv, SAGEConv), NetworkX |
| Données | NumPy, Pandas |
| Visualisation | Matplotlib, Seaborn |
| Environnement | VS Code, Jupyter, Git/GitHub |

## 4. Étapes détaillées

### Étape 1 — Choisir le réseau
Commencer par **IEEE 14-bus** (14 bus, 20+ branches, générateurs, charges, contraintes de tension) : assez petit pour être compris en détail, assez riche pour un vrai GNN. Passer ensuite à IEEE 30-bus pour tester le passage à l'échelle.

### Étape 2 — Comprendre les données
- **Bus data** (→ nodes) : Pd, Qd, Vm, Va, type (PQ/PV/Slack)
- **Branch data** (→ edges) : r, x, b (résistance, réactance, susceptance)
- **Generator data** : Pg, Qg, Vg, Pmax, Pmin, Qmax, Qmin

### Étape 3 — Générer le dataset
Ne pas entraîner sur un seul état. Créer des centaines/milliers de scénarios en faisant varier aléatoirement les charges :
```python
load_factor = np.random.uniform(0.8, 1.2)
```
appliqué à Pd et Qd.

### Étape 4 — Power Flow = Ground Truth
Pour chaque scénario, lancer un AC Power Flow (MATPOWER/pandapower) et récupérer Vm, Va, Pg, Qg, Pbranch, Qbranch. Ces sorties constituent la vérité terrain (ground truth).

### Étape 5 — Transformer le réseau en graphe
G = (V, E) avec V = bus, E = lignes de transmission. Prototypage possible avec NetworkX, implémentation finale avec PyTorch Geometric.

### Étape 6 — Features des nœuds
Vecteur de features par bus, par exemple :
```
X_i = [Pd, Qd, Pg, Qg, bus_type, Vmax, Vmin]
```
⚠️ Ne jamais inclure Vm/Va réels en entrée si ce sont les cibles à prédire (fuite de données / triche).

### Étape 7 — Features des arêtes
Par ligne : `[resistance, reactance, susceptance, capacity]` — donne au GNN la physique du réseau, pas seulement les données des bus.

### Étape 8 — Construire l'objet Data (PyG)
```python
Data(
    x=node_features,
    edge_index=edge_index,
    edge_attr=edge_features,
    y=target
)
```

### Étape 9 — Construire le GNN
Architecture de départ (GCN) :
```
Input (7 features/node)
   → GCNConv(7 → 64) → ReLU
   → GCNConv(64 → 64) → ReLU
   → Linear(64 → 2)
   → Output [Vm, Va]
```

### Étape 10 — Cible à prédire
`Y = [Vm, Va]` pour chaque bus. Commencer par Vm seul, puis ajouter Va.

### Étape 11 — Fonction de perte
MSE Loss :
```python
loss = torch.nn.functional.mse_loss(prediction, target)
```

### Étape 12 — Évaluation
Métriques : MAE, RMSE, R². Comparer visuellement prédiction vs vérité terrain.

### Étape 13 — Graphiques à produire (minimum 5)
1. Topologie du réseau IEEE 14-bus (NetworkX)
2. Vraie tension vs tension prédite
3. Erreur par bus
4. Courbe d'apprentissage (loss / epoch)
5. GNN vs solveur classique (précision + temps de calcul — **mesuré**, pas inventé)

### Étape 14 — Comparaison avec le solveur classique
Newton-Raphson/OPF = précis mais itératif ; GNN = approximation apprise, inférence rapide. L'objectif n'est pas de remplacer le solveur mais de montrer la faisabilité d'une approximation rapide.

### Étape 15 — Ajouter l'optimisation (OPF)
Une fois la prédiction validée, ajouter la vérification de contraintes (tension, flux de ligne) et un solveur OPF (MATPOWER) pour minimiser le coût de génération tout en respectant les contraintes.
```
Load → Graphe → GNN → État prédit → Vérification contraintes → OPF → Coût minimal
```

### Étape 16 — Test de généralisation
Au lieu d'un simple split train/test 80/20, entraîner sur une plage de charge (ex. 0.8–1.1) et tester sur une plage non vue (ex. 1.1–1.2), pour vérifier la capacité de généralisation.

### Étape 17 — Bonus : GCN vs GAT
Comparer GCN (agrégation uniforme des voisins) et GAT (pondération par attention) sur MAE, RMSE, R², temps de calcul.

## 5. Organisation GitHub recommandée

```
gnn-power-grid-optimization/
├── README.md
├── data/
│   ├── raw/
│   ├── processed/
│   └── scenarios/
├── notebooks/
│   ├── 01_power_grid_analysis.ipynb
│   ├── 02_dataset_generation.ipynb
│   ├── 03_graph_construction.ipynb
│   ├── 04_gnn_training.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── data_generation.py
│   ├── graph_builder.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── optimization.py
├── results/
│   ├── figures/
│   ├── metrics/
│   └── models/
├── requirements.txt
└── LICENSE
```
