# Traitement d’images — version 1.5.0, couleurs du sujet protégées

## Correctif colorimétrique 1.5.0

Les TIFF source fournis sont en RVB et ne contiennent aucun profil ICC. La
version précédente convertissait l'ensemble du sujet en CMJN sans profil de
destination ; selon le logiciel d'affichage, cela produisait une image plus
claire et plus jaune.

Cette conversion globale est supprimée. Le sujet reste désormais en RVB sRGB,
sans correction de couleur, luminosité, contraste ou saturation. Le fond et la
marge utilisent le bleu-noir de référence `RVB 0, 0, 12` (`#00000c`). Un profil
sRGB est intégré au TIFF final afin que les logiciels l'affichent de façon
cohérente.

Le contrôle qualité relit chaque TIFF sauvegardé et compare tous les pixels du
cœur du sujet avec ceux attendus. Au moindre écart de canal, le fichier reçoit
un avertissement de contrôle qualité.

Cette version fonctionne exclusivement avec une détection déterministe du fond
gris extérieur. Elle ne télécharge, ne charge et n’exécute aucun modèle
d’intelligence artificielle.

## Principe de traitement

Pour chaque fichier TIFF placé dans `input`, le programme suit cet ordre :

1. lecture de chaque trame du fichier source ;
2. utilisation prioritaire d’un canal alpha déjà présent et valide ;
3. sinon, mesure du fond gris sur le cadre extérieur de l’image ;
4. retrait des seuls pixels gris reliés au cadre extérieur ;
5. séparation des éléments réellement disjoints ;
6. protection des pixels intérieurs de chaque élément ;
7. calcul de l’orientation à partir du plus long bord ;
8. rotation rigide, sans étirement ni écrasement ;
9. recadrage de l’élément après rotation ;
10. ajout d’une bordure physique de 2 mm ;
11. remplacement du fond par RVB 0, 0, 12, sans convertir le sujet ;
12. contrôle qualité, nommage incrémenté et sauvegarde dans `output`.

Le fond gris n’est supprimé que s’il ressemble au fond mesuré et s’il est
connecté à l’extérieur de l’image. Une zone grise enfermée à l’intérieur d’un
timbre ou d’une photographie reste donc conservée.

Une petite languette ou bande du scanner collée au cadre est ignorée lorsqu’elle
reste superficielle et de faible surface. Cette correction empêche les résidus
du support d’être exportés comme des éléments supplémentaires.

Si le fond extérieur n’est pas suffisamment gris, uniforme ou visible autour
des éléments, le programme signale une erreur claire. Il n’essaie pas une
méthode de secours susceptible d’abîmer les dentelures ou les bordures.

## Dossiers

```text
traitement_images_cmjn/
├── input/                  fichiers originaux à traiter
├── output/                 fichiers générés
├── image_pipeline/         modules du programme
├── requirements.txt        dépendances
└── run_pipeline.py         lancement
```

Les originaux placés dans `input` ne sont jamais écrasés.

## Installation

Depuis l’invite de commandes Windows :

```bat
cd /d C:\Users\arman\OneDrive\Bureau\traitement_images_cmjn_v2\traitement_images_cmjn
python -m pip install -r requirements.txt
```

Les dépendances utilisées sont NumPy, Pillow, OpenCV et SciPy. Aucun composant
GPU n’est nécessaire.

## Utilisation normale

1. Copier les TIFF dans le dossier `input`.
2. Ouvrir l’invite de commandes.
3. Exécuter :

```bat
cd /d C:\Users\arman\OneDrive\Bureau\traitement_images_cmjn_v2\traitement_images_cmjn
python run_pipeline.py
```

4. Récupérer les résultats dans `output`.

Vérification de la version :

```bat
python run_pipeline.py --version
```

La réponse attendue est `1.5.0`.

## Nommage et séparation

Chaque élément détecté produit un fichier indépendant. Pour une source nommée
`document_A.tif`, les sorties sont :

```text
document_A_1.tif
document_A_2.tif
document_A_3.tif
```

Si certains numéros existent déjà, le programme continue après le plus grand
numéro afin de ne rien écraser.

## Bordure foncée de 2 mm

La bordure est ajoutée après la rotation et le recadrage. Sa taille en pixels
est calculée avec la résolution du TIFF :

```text
pixels = plafond(2 × résolution_dpi / 25,4)
```

Si le fichier ne contient pas de résolution exploitable, 300 dpi sont utilisés.
Le TIFF final est enregistré en RVB sRGB pour conserver strictement les
couleurs intrinsèques du sujet. La bordure utilise :

```text
R = 0   V = 0   B = 12   (#00000c)
```

## Paramètres principaux

Ils se trouvent à la fin de `image_pipeline/config.py` :

- `split_subjects=True` : sépare les éléments disjoints ;
- `straighten=True` : active le redressement ;
- `output_margin_mm=2.0` : largeur de la bordure ;
- `preserve_subject_rgb=True` : interdit la conversion globale en CMJN ;
- `rgb_background=(0, 0, 12)` : couleur du fond et de la marge ;
- `output_format="TIFF"` : format sans perte recommandé.

Exemple avec des dossiers personnalisés :

```bat
python run_pipeline.py "C:\MesTIFF" -o "C:\MesResultats" --margin-mm 2
```

## Journaux et erreurs

Le dossier `output` contient :

- `traitements.log` : progression et erreurs lisibles ;
- `controle_qualite.jsonl` : résultats détaillés des contrôles.

Une erreur « fond extérieur non exploitable » signifie que le programme ne
peut pas distinguer le fond avec assez de sécurité. Il faut alors fournir un
scan avec un fond gris plus uniforme et une bande de fond visible tout autour
des éléments, ou fournir une image déjà détourée avec transparence.

## Limites

- Deux éléments qui se touchent peuvent être considérés comme un seul élément.
- Un fond très texturé, coloré, ombré ou interrompu n’est pas traité.
- Une dentelure ayant exactement la même couleur que le fond et directement
  reliée à celui-ci peut rester ambiguë.
- L’orientation d’un sujet presque carré peut être indéterminable ; le journal
  l’indique alors et évite une rotation hasardeuse.

Ces limites sont volontaires : la priorité est de ne pas altérer inutilement le
sujet original.
