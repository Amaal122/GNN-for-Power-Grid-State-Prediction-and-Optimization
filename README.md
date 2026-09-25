# Graph Neural Network for Power Grid State Prediction and Optimization

Prédiction rapide de l'état électrique d'un réseau (tensions, angles) à partir de sa structure topologique via un Graph Neural Network, couplée à une optimisation OPF conditionnelle.

## Objectif

Un réseau électrique est naturellement un graphe : les bus sont les nœuds, les lignes et transformateurs sont les arêtes. Ce projet entraîne un GNN à approximer le résultat d'un power flow (solveur physique itératif) à partir de la charge du réseau, puis évalue dans quelle mesure cette approximation peut accélérer une chaîne de décision réaliste (détection de violation → optimisation OPF).

**Question scientifique posée** : un GNN entraîné peut-il approximer l'état d'un système électrique avec une précision acceptable, en interpolation comme en extrapolation, tout en offrant un gain de temps significatif par rapport aux solveurs classiques ?

## Stack technique

| Domaine | Outils |
|---|---|
| Langage | Python 3.13 |
| Systèmes électriques | pandapower, cas IEEE 14-bus |
| Machine Learning | PyTorch |
| Graph Learning | PyTorch Geometric (GCNConv, GATConv), NetworkX |
| Données / Visualisation | NumPy, Pandas, Matplotlib, scikit-learn |
| Environnement | Jupyter, VS Code |

## Réseau étudié : IEEE 14-bus

- 14 bus (nœuds), 15 lignes + 5 transformateurs (20 arêtes au total)
- 11 bus de charge (PQ), 4 générateurs (PV), 1 bus de référence (slack)
- Contraintes de tension : [0.94, 1.06] pu sur tous les bus

## Méthodologie

1. **Génération du dataset** : 650 scénarios (500 pour l'entraînement/test in-distribution avec load factor ∈ [0.8, 1.1], 150 pour le test hors distribution avec load factor ∈ [1.1, 1.2]), chacun résolu par power flow AC (pandapower) pour obtenir Vm/Va (ground truth).
2. **Construction du graphe** : nodes = bus (features : Pd, Qd, type de bus, Vmin, Vmax), edges = lignes **et transformateurs** (voir section Debugging), pas de fuite de données (Vm/Va jamais en entrée).
3. **Normalisation** : standardisation (z-score) des features et cibles, calculée sur le train uniquement.
4. **Modèles** : GCN (2 couches GCNConv) et GAT (2 couches GATConv, 4 têtes d'attention), entraînés avec Adam (lr=0.01), MSE loss, 100 epochs.
5. **Évaluation** : MAE, RMSE, R² séparés pour Vm et Va, sur test in-distribution et out-of-distribution.
6. **Optimisation** : intégration avec l'OPF de pandapower, déclenché conditionnellement selon la prédiction du GNN.

## Résultats

### GCN — Performance de base (après correction du graphe)

| | Vm MAE | Vm R² | Va MAE (°) | Va R² |
|---|---|---|---|---|
| Test in-distribution | 0.00046 | 0.9992 | 0.132 | 0.9987 |
| Test out-of-distribution | 0.00172 | 0.9897 | 0.868 | 0.9574 |

### GCN vs GAT

| | Vm MAE | Vm R² | Va MAE (°) | Va R² |
|---|---|---|---|---|
| GCN in-dist | 0.00046 | 0.9992 | 0.132 | 0.9987 |
| **GAT in-dist** | **0.00036** | **0.9995** | **0.073** | **0.9996** |
| GCN OOD | 0.00172 | 0.9897 | 0.868 | 0.9574 |
| **GAT OOD** | **0.00128** | **0.9913** | **0.181** | **0.9981** |

**Conclusion** : GAT surpasse GCN à la fois en interpolation et en extrapolation, avec un gain particulièrement marqué en généralisation (erreur Va réduite de plus de 4x en régime hors distribution). Le mécanisme d'attention, en pondérant dynamiquement l'influence des bus voisins selon leur état réel plutôt que selon la seule topologie, s'adapte mieux à des configurations de charge inhabituelles. Le coût computationnel supplémentaire de GAT reste négligeable sur un réseau de cette taille (temps d'inférence comparable, voire légèrement meilleur, que GCN).

### Pipeline hybride GNN + OPF conditionnel

Sur un échantillon de 30 scénarios (load factor ∈ [0.8, 1.3]) :

| Approche | Temps total | Notes |
|---|---|---|
| OPF systématique | ~10 827 ms | référence, chaque scénario passe par l'OPF |
| **Pipeline hybride (GNN + OPF si violation)** | **~3 118 ms** | OPF déclenché seulement 8/30 fois (26.7%) |
| GNN seul (théorique) | ~227 ms | 47x plus rapide que l'OPF, mais sans garantie d'exactitude |

**Gain de temps du pipeline hybride vs OPF systématique : ~71%**, en réservant le solveur exact aux cas où le GNN détecte une violation de contrainte de tension sur les bus de charge (marge de tolérance de 0.005 pu pour absorber l'erreur normale de prédiction).

L'OPF lui-même réduit le coût de génération de **1.1%** par rapport à une répartition non optimisée sur le cas de référence.

## Points méthodologiques importants (debugging)

- **Normalisation indispensable** : sans elle, la MSE globale est dominée par Va (amplitude ~16°) au détriment de Vm (amplitude ~0.16 pu) — R² de Vm passait de 0.37 à 0.997 après normalisation.
- **Graphe initialement incomplet** : les 5 transformateurs du réseau (liaisons HV/LV, ex. bus 6→7) n'étaient pas inclus dans `edge_index`, isolant le bus 7 du reste du graphe et causant une erreur Va 7x supérieure à la moyenne sur ce bus. Identifié via le graphique d'erreur par bus, corrigé en fusionnant `net.line` et `net.trafo`.
- **Violations "structurelles" vs réelles** : les générateurs aux bus 5 et 7 ont une consigne de tension (1.07, 1.09 pu) supérieure à la limite du réseau (1.06 pu) par construction du cas IEEE 14-bus — seul l'OPF corrige ça, pas le power flow classique. La détection de violation pour le pipeline hybride est donc restreinte aux bus de charge (PQ) uniquement.

## Structure du projet

```
gnn-power-grid-optimization/
├── README.md
├── notebooks/
│   └── 01_power_grid_analysis.ipynb
├── requirements.txt
└── LICENSE
```

## Limites et pistes d'amélioration

- Le test de généralisation se limite à une extrapolation de charge (facteur > 1.1) ; une généralisation à d'autres topologies (IEEE 30-bus) reste à tester.
- Le pipeline OPF conditionnel utilise un seuil de marge fixe (0.005 pu) ; une calibration plus rigoureuse (ex. basée sur l'intervalle de confiance du modèle) serait plus robuste.
- Non testé : robustesse face à des scénarios de perte de ligne (contingency analysis), qui changeraient la topologie du graphe.
