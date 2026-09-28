# Traitement d’images — version 1.9.1

## Nouveauté 1.9.0

La bordure foncée autour de chaque sujet passe de **2 mm à 1 mm**
(12 px à 300 dpi, 24 px à 600 dpi). Elle reste modifiable dans
`image_pipeline/config.py` (`output_margin_mm`) ou avec `--margin-mm`.

## Nouveautés 1.8.0

- **Éléments posés presque bord à bord** : deux planches séparées par un
  couloir de fond de quelques pixels (souvent plus sombre que le fond, à cause
  de l'ombre des bords du papier) sont désormais séparées en deux fichiers.
  Seul un couloir *étroit* (8 px au plus) peut séparer : une zone d'encre
  sombre plus large touchant le bord d'un timbre ne le coupe jamais.
- **Bande parasite le long d'un bord du scan** (carton noir plus court que la
  vitre, marge blanche du scanner) : elle est ignorée au lieu de faire échouer
  le scan (« fond extérieur non exploitable »), si elle ne dépasse pas 5 % de
  la dimension.
- **Numérotation** : des éléments côte à côte sont numérotés de gauche à
  droite, même si l'un est posé quelques pixels plus haut.
- **Poussières du fond** (1.7.1) : elles ne sont plus rattachées au sujet et
  ne faussent plus le cadrage ni la mesure de l'inclinaison.

## Rotation sans lignes de décalage (1.7.0)

Jusqu'à la 1.6.0, la rotation déplaçait des pixels entiers (plus proche
voisin). Sur un sujet incliné de quelques degrés, cela dupliquait ou sautait
une ligne de pixels à intervalles réguliers : les traits fins et le texte
présentaient des **lignes de décalage** (effet d'escalier), et le bord du
sujet devenait **pointillé**.

La rotation utilise désormais :

- un filtre **Lanczos** pour les couleurs (famille du « Bicubique » de
  Photoshop, légèrement plus net) : plus aucun pixel dupliqué ou sauté ;
- un **anti-halo** : chaque pixel tourné reste compris entre le minimum et le
  maximum des 4 pixels d'origine qui l'entourent, ce qui supprime les liserés
  clairs ou sombres qu'un filtre bicubique ou Lanczos crée le long des traits
  fins ;
- un **contour sous-pixel** : les pixels du bord, mi-sujet mi-fond gris dans
  le scan, reçoivent une transparence partielle proportionnelle à leur
  mélange, et le gris en est retiré. Le bord reste ainsi droit et lisse après
  redressement (paramètre `subpixel_edges=True`).

Une rotation d'un angle quelconque recalcule forcément les pixels : ils ne sont
plus identiques au bit près à l'original, mais aucune couleur plus claire ou
plus foncée que celles du voisinage d'origine n'est créée. **Sans rotation**
(sujet déjà droit, inclinaison inférieure à `min_rotation_deg`), les pixels
sont copiés sans aucun recalcul.


## Correctif colorimétrique 1.5.0

Les TIFF source fournis sont en RVB et ne contiennent aucun profil ICC. La
version précédente convertissait l'ensemble du sujet en CMJN sans profil de
destination ; selon le logiciel d'affichage, cela produisait une image plus
claire et plus jaune.

Cette conversion globale est supprimée. Le sujet reste désormais en RVB sRGB,
sans correction de couleur, luminosité, contraste ou saturation. Le fond et la
marge utilisent le noir de référence `RVB 5, 0, 2` (`#050002`). Un profil
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
3. sinon, mesure du fond gris sur le cadre extérieur de l’image (une bande
   parasite étroite le long d'un bord est d'abord écartée) ;
4. retrait des seuls pixels gris reliés au cadre extérieur ;
5. séparation des éléments disjoints, y compris posés presque bord à bord ;
6. protection des pixels intérieurs de chaque élément ;
7. calcul de l’orientation à partir du plus long bord ;
8. rotation rigide (Lanczos + anti-halo), sans étirement ni écrasement ;
9. recadrage de l’élément après rotation ;
10. ajout d’une bordure physique de 1 mm ;
11. remplacement du fond par RVB 5, 0, 2, sans convertir le sujet ;
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
├── tests/                  tests automatiques
├── INSTALLER.bat           installation (une seule fois)
├── LANCER_TRAITEMENT.bat   traitement du dossier input
├── BILAN_MANUEL.bat        bilan du contrôle manuel
├── requirements.txt        dépendances
└── run_pipeline.py         lancement en ligne de commande
```

Les originaux placés dans `input` ne sont jamais écrasés.

## Installation

Une seule fois : double-cliquer sur **`INSTALLER.bat`**.

Il crée dans le dossier du projet un environnement Python 3.12 (`.venv`) et y
installe NumPy, Pillow, OpenCV et SciPy. Python 3.12 doit être installé
(https://www.python.org/downloads/). Python 3.14 n'est volontairement pas
utilisé : le contrôle intelligent des applications de Windows y bloque une
bibliothèque de SciPy.

`LANCER_TRAITEMENT.bat` et `BILAN_MANUEL.bat` utilisent automatiquement cet
environnement `.venv`. S'il n'existe pas, ils essaient Python 3.12, puis
`python`. Aucun composant GPU n'est nécessaire.

## Utilisation normale

Le plus simple : copier les TIFF dans le dossier `input`, puis double-cliquer
sur `LANCER_TRAITEMENT.bat`. La fenêtre reste ouverte à la fin (ou en cas
d'erreur) tant qu'une touche n'a pas été pressée, pour pouvoir lire le
résultat. Les fichiers générés apparaissent dans `output`.

En ligne de commande :

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

La réponse attendue est `1.9.1`.

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

## Bordure foncée de 1 mm

La bordure est ajoutée après la rotation et le recadrage. Sa taille en pixels
est calculée avec la résolution du TIFF :

```text
pixels = plafond(1 × résolution_dpi / 25,4)
```

Si le fichier ne contient pas de résolution exploitable, 300 dpi sont utilisés.
Le TIFF final est enregistré en RVB sRGB pour conserver strictement les
couleurs intrinsèques du sujet. La bordure utilise :

```text
R = 5   V = 0   B = 2   (#050002)
```

## Paramètres principaux

Ils se trouvent à la fin de `image_pipeline/config.py` :

- `split_subjects=True` : sépare les éléments disjoints ;
- `straighten=True` : active le redressement ;
- `subpixel_edges=True` : contour anticrénelé, sans escalier après rotation ;
- `output_margin_mm=1.0` : largeur de la bordure ;
- `preserve_subject_rgb=True` : interdit la conversion globale en CMJN ;
- `rgb_background=(5, 0, 2)` : couleur du fond et de la marge ;
- `output_format="TIFF"` : format sans perte recommandé.

Exemple avec des dossiers personnalisés :

```bat
python run_pipeline.py "C:\MesTIFF" -o "C:\MesResultats" --margin-mm 1
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

- Deux éléments qui se chevauchent, ou qui se touchent sur une longueur
  importante, sont considérés comme un seul élément.
- Un élément dont un bord déborde de travers (pellure, languette) peut fausser
  la mesure de l'inclinaison ; le contrôle qualité le signale alors.
- Un fond très texturé, coloré, ombré ou interrompu n’est pas traité.
- Une dentelure ayant exactement la même couleur que le fond et directement
  reliée à celui-ci peut rester ambiguë.
- L’orientation d’un sujet presque carré peut être indéterminable ; le journal
  l’indique alors et évite une rotation hasardeuse.

Ces limites sont volontaires : la priorité est de ne pas altérer inutilement le
sujet original.
