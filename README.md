# GDA-Project
Charles &amp; Elouan, 
- The Details Matter: Preventing Class Collapse in Supervised Contrastive Learning [paper link](https://mdpi-res.com/d_attachment/csmf/csmf-03-00004/article_deploy/csmf-03-00004.pdf?version=1650444797)
- [consignes projet](https://notes.inria.fr/kRDAr2fmSI6Bw5tbBz0y1g#)
- [report overleaf](https://www.overleaf.com/4548983824dskwwxgsthcy#58a21c)

Idées :

- à quel point est-ce que les résultats sont applicables suivant les types de données (données médicales ? données + complexes type vidéo ?) (ça se rapproche de ce que tu disais sur le dataset)
- exploration autour de hiérarchies à l’intérieur des classes ou entre les classes ?
- exploration autour de strata ambiguës ou mal définies ?
- utilisation d’un classifier plus complexe qu’un linear classifier pour utiliser les embeddings ?
- évaluer sur d’autres métriques ?
- autres datasets
- learnable alpha pour différents moments du training --> original figure pour l'évaluation
- analysis of alpha's effect. ils mentionnent sweep et garder les meilleures values found selon les dataset mais n'analysent / hypothèsent pas sur les
origines possibles des différents alpha optimaux selon les données

Critiques papier:
- critique --> InfoNCE is not used anymore in SOTA SSL methods, even contrastive-derived (ByOL, Dino, etc)
- repose sur des hypothèses fortes (class balance, labels de haute qualité)
- résultats théoriques reposent sur des hypothèses très fortes (infinite encoder) donc en réalité résultats plutôt empiriques que théoriques 



