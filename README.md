# Traitement d’images — version 1.6.0, contrôle manuel et couleurs protégées

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
Traitement-Images-LTC/
├── input/                  fichiers originaux à traiter
├── output/                 fichiers générés
├── image_pipeline/         modules du programme
├── requirements.txt        dépendances
└── run_pipeline.py         lancement
```

Les originaux placés dans `input` ne sont jamais écrasés.

## Installation

Dans l'explorateur de fichiers, ouvrir le dossier extrait du zip, puis
Maj + clic droit dans le dossier → « Ouvrir dans le terminal » (ou « Ouvrir
la fenêtre PowerShell ici »). Ensuite, exécuter :

```bat
python -m pip install -r requirements.txt
```

Les dépendances utilisées sont NumPy, Pillow, OpenCV et SciPy. Aucun composant
GPU n’est nécessaire.

## Utilisation normale

1. Copier les TIFF dans le dossier `input`.
2. Ouvrir un terminal dans le dossier du projet (Maj + clic droit → « Ouvrir
   dans le terminal »).
3. Exécuter :

```bat
python run_pipeline.py
```

4. Récupérer les résultats dans `output`.

Vérification de la version :

```bat
python run_pipeline.py --version
```

La réponse attendue est `1.6.0`.

## Nommage, lots et séparation

Plusieurs dossiers peuvent être déposés dans `input` lors d'un même lancement.
Chaque dossier de premier niveau produit un dossier portant le même nom suivi de
`-ps` dans `output`. Par exemple :

```text
input/Lot_001/001.tif  ->  output/Lot_001-ps/001.tif
input/Lot_002/010.tif  ->  output/Lot_002-ps/010.tif
```

Une source qui ne produit qu'une seule image conserve son nom. Si une source
produit plusieurs images indépendantes, les sorties sont numérotées avec un
tiret :

```text
document_A-1.tif
document_A-2.tif
document_A-3.tif
```

Le programme n'écrase jamais silencieusement une sortie existante.

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


## Contrôle manuel après traitement

Le traitement crée dans chaque dossier `*-ps` un manifeste technique
`controle_manuel_manifest.json`. Ce manifeste mémorise la source de chaque
sortie, les sorties effectivement produites, les contrôles qualité et les
éventuels refus/erreurs.

Après avoir contrôlé visuellement les résultats, il est possible de supprimer
les sorties jugées insatisfaisantes puis de lancer :

```bat
python run_pipeline.py --bilan-manuel
```

Le programme compare alors les sorties présentes au manifeste et crée, si
nécessaire :

```text
output/Lot_001-ps/
├── A_TRAITER_MANUELLEMENT/
└── CONTROLE_MANUEL.txt
```

Pour chaque sortie supprimée, `A_TRAITER_MANUELLEMENT` reçoit une copie binaire
intégrale du TIFF source. Si deux sorties provenant du même TIFF ont été
supprimées, deux copies du TIFF original sont préparées, avec les noms des
sorties supprimées. Les échecs/refus automatiques sont également préparés pour
reprise manuelle et leur raison est inscrite dans `CONTROLE_MANUEL.txt`.

Les fichiers du dossier `input` ne sont jamais déplacés, renommés, écrasés ou
modifiés par cette fonction. Une copie déjà présente dans
`A_TRAITER_MANUELLEMENT` n'est jamais écrasée, afin de protéger un éventuel
travail manuel déjà commencé.

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
