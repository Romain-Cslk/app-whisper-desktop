# Diarisation 1.2.1 — fonctionnement, limites et recette

## 1. Ce que l'utilisateur peut corriger

La revue présente les groupes de voix, leur durée, un nom éditable, plusieurs extraits, les variantes disponibles et une case de mémorisation. Elle propose jusqu'à trois extraits distincts par groupe d'origine, de six secondes au maximum. Après une fusion, les extraits des différents groupes restent accessibles pour vérifier que toutes les voix appartiennent bien à la même personne.

Sélectionner au moins deux groupes et choisir **Fusionner**, puis saisir leur nom commun. Les groupes disparaissent au profit d'un groupe unique, et l'aperçu de tous leurs passages est mis à jour. Les mots et horodatages sont conservés. L'identifiant d'origine de chaque segment est conservé pour pouvoir séparer une fusion plus tard. Deux personnes portant le même nom ne sont pas fusionnées implicitement.

**Annuler la fusion** restaure l'état précédent des groupes ; **Séparer** restaure les groupes d'origine de la fusion sélectionnée. Une simple modification du nom met aussi l'aperçu à jour. Fermer ou annuler le dialogue ne sauvegarde ni noms ni profils. **Appliquer** met à jour le TXT et le JSON structuré. Cela ne remplace pas les fichiers exportés auparavant ni les comptes rendus/résumés déjà générés ; le dialogue le signale lorsqu'un document dérivé est détecté.

La lecture utilise le fichier local original ou les pistes isolées encore disponibles. Un seul extrait joue à la fois ; sa lecture est arrêtée à la fin de l'intervalle ou à la fermeture. La précision effective du positionnement dépend du format et du moteur multimédia Qt. Les erreurs de chargement, les fichiers manquants et les bornes incompatibles sont affichés, pas masqués.

## 2. Pistes microphone/système et fichiers importés

La diarisation reste désactivée par défaut. Désactivée, elle ne charge pas le moteur de diarisation et ne calcule pas d'empreintes vocales.

Activée avant un enregistrement natif, elle conserve les pistes microphone et système et leurs décalages temporels sous le répertoire de données `results/recording_sources/<id>`. En mode de transcription **local**, des pistes natives valides sont transcrites séparément puis fusionnées chronologiquement, mots compris. La diarisation conserve leur provenance. Une piste annoncée mais manquante fait revenir à l'audio mixé complet plutôt que d'ignorer la moitié de la réunion.

**Le microphone personnel n'est pas une preuve d'identité** : son attribution au nom personnel configuré suppose qu'une seule personne utilise ce microphone. Un micro de salle, le retour de haut-parleurs ou de l'écho peuvent violer cette hypothèse. Le dispositif ne sépare pas automatiquement toutes les voix d'un micro partagé.

Pour les fichiers importés sans pistes séparées, aucune provenance microphone n'est inventée. Quand plusieurs voix peuvent correspondre au même passage avec des scores d'alignement proches, le passage reste non attribué plutôt que d'être affecté arbitrairement au microphone. Les paroles simultanées restent une limite acoustique importante.

Le chemin existant de transcription OpenAI n'a pas été transformé en transcription locale : s'il est sélectionné, son comportement réseau existant subsiste. Le correctif n'ajoute aucun envoi de profil vocal ni de diarisation vers un service externe, et aucun secours cloud automatique.

## 3. Réduction des faux interlocuteurs

L'option **Nombre d'intervenants** vaut `Auto` ou un entier de 1 à 32. Elle désigne le nombre total de personnes qui parlent réellement, microphone personnel compris ; pas le nombre d'invités à la réunion. Sur des pistes séparées, le nombre distant n'est diminué de un que si une parole est détectée au microphone. Une configuration contradictoire est refusée.

En mode Auto, le seuil de regroupement passe de 0,50 à 0,60 et devient réglable. Dans le moteur retenu, augmenter ce seuil permet davantage de regroupements ; un nombre imposé rend le seuil sans effet. **0,60 est un réglage d'ingénierie, pas un optimum mesuré sur les réunions de l'utilisateur.** Trop regrouper risque de confondre deux personnes ; trop peu risque de fragmenter une personne. L'option de nombre et la revue humaine restent nécessaires.

Les extraits utilisés pour les empreintes sont répartis dans le temps, préférentiellement sans autre voix superposée, entre 1,5 et 8 secondes, jusqu'à six par groupe. Le même extrait n'est pas répété artificiellement pour faire croire que plusieurs preuves indépendantes existent. Silence, valeurs non finies et bornes invalides sont contrôlés.

Référence du paramètre de regroupement : documentation officielle Sherpa-ONNX, exemple `speaker_diarization.html` :
https://k2-fsa.github.io/sherpa/onnx/javascript-api/examples/speaker_diarization.html

## 4. Mémorisation des voix : explicite, multivariantes, locale

Fusionner n'enregistre **pas** automatiquement une voix pour les prochaines réunions. La case de mémorisation est décochée à chaque ouverture. L'utilisateur doit la cocher puis confirmer la mémorisation. Un nom existant du même espace de modèle est enrichi ; le dialogue le précise. Des groupes distincts ayant le même nom ne sont pas enrôlés ensemble sans fusion explicite.

Le profil conserve les vecteurs des différents groupes fusionnés, et non une moyenne unique. Jusqu'à **64 variantes** sont conservées par profil. Au-delà, un sous-ensemble diversifié est retenu plutôt qu'une simple file qui jette les plus anciennes variantes. Ce n'est donc ni un stockage infini ni un réentraînement du réseau de neurones.

La reconnaissance compare les extraits aux variantes connues. Elle exige notamment un score de similarité minimal de 0,82, une marge minimale de 0,08 face au deuxième profil, cinq secondes de parole, au moins deux extraits de requête et deux références dans le profil. Tous les extraits de requête doivent soutenir l'attribution. Ces valeurs ne sont **pas des pourcentages de certitude** ni des seuils statistiquement calibrés sur ce corpus. En cas d'ambiguïté, d'incompatibilité du modèle ou de preuves insuffisantes, le nom n'est pas proposé automatiquement.

Une reconnaissance automatique n'enrichit jamais le profil. Un faux nom confirmé ou une fusion de personnes différentes peut cependant contaminer un profil : écouter les variantes avant de mémoriser. Une fusion préserve les variantes **encore disponibles** ; elle ne peut reconstituer celles déjà perdues dans l'ancien patch.

## 5. Données, migration et sauvegarde

Les profils et les revues contenant des empreintes utilisent Windows DPAPI pour le compte Windows courant, sous `%LOCALAPPDATA%\TranscripteurWhisper\speakers` par défaut. Les emplacements configurés via l'application restent respectés. Aucun mode production ne remplace DPAPI par un stockage en clair. Les codecs non chiffrants des tests sont explicitement réservés aux données synthétiques.

Il faut distinguer les **profils permanents** créés après consentement des **empreintes temporaires de revue**, nécessaires pour permettre ce choix après la transcription. Les secondes sont chiffrées mais existent aussi sans consentement à un profil permanent. Elles sont supprimées après application de la revue ou, passé sept jours, **au prochain accès/nettoyage**, pas par un service autonome à heure fixe. Les métadonnées utiles à la correction des noms restent disponibles. L'action de suppression de tous les profils efface aussi les empreintes temporaires accessibles. Un fichier temporaire chiffré illisible peut devoir être supprimé entièrement.

Les fichiers audio sources restent dans leur emplacement existant. Le correctif ne leur applique pas DPAPI. Les TXT et JSON de transcription restent des documents en clair comme auparavant. Les chemins utiles à l'écoute sont conservés dans la revue chiffrée. Les dossiers temporaires du moteur sont supprimés à la fin de son exécution normale ; un arrêt brutal peut retarder le nettoyage. Une suppression de fichier n'est pas une garantie d'effacement physique sur SSD ou de disparition d'une sauvegarde externe.

Les profils de schéma 1 et les anciennes revues sont lisibles. Une ancienne revue sans tours de parole explicites utilise les segments du JSON ; une ancienne revue sans chemin audio permet de choisir **le fichier original complet sur la même chronologie**. Un fichier tronqué ou recadré n'est pas automatiquement réaligné. Si l'audio a disparu, l'application ne peut pas l'inventer ; le renommage reste possible. Si les anciennes empreintes ont expiré, retranscrire l'audio est nécessaire pour constituer de nouvelles références.

L'application des noms, du JSON, du TXT, des profils demandés et de la revue passe par une sauvegarde journalisée. Les validations et le chiffrement sont préparés avant publication. Des verrous de thread et de processus sérialisent les écritures coopératives ; une erreur provoque un retour arrière. Après interruption, une transaction inachevée est récupérée au prochain accès. Des modifications externes plus récentes ou une revue devenue obsolète font refuser l'écrasement. Cela n'est pas une promesse d'atomicité multi-fichiers fournie par le système de fichiers ni une garantie contre tout sinistre matériel.

## 6. Validation Windows obligatoire avant diffusion

Depuis le dépôt corrigé :

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\validate_diarization_patch.ps1
```

Le script exige Windows x64/Python 3.11, un `uv.lock` cohérent, Ruff, la compilation syntaxique, la suite pytest du dépôt et un rapport JUnit. Il **refuse** que les neuf tests obligatoires Qt/lecteur/DPAPI soient absents, ignorés ou en échec. Il prépare ensuite les modèles et lance les contrôles de chargement Qt et d'inférence du processus isolé. Le mode `-BuildInstaller` ajoute le build PyInstaller, les essais du véritable EXE et Inno Setup. Les rapports source sont placés dans `build/diarization-validation` ; le build d'installateur recrée son dossier `build` et produit ses propres rapports.

Les modèles sont téléchargés depuis les releases configurées, avec contrôles du manifest de hash utilisé par le projet. L'archive du patch ne contient pas ces poids. Le processus principal garde la diarisation native isolée dans un processus enfant Sherpa-ONNX.

## 7. Recette acoustique ciblée

Effectuer ces contrôles sur une copie d'une réunion réelle avant d'utiliser les noms pour attribuer des actions automatiquement.

| Cas | Résultat attendu |
|---|---|
| Réunion de quatre voix connues, microphone personnel et trois distantes | Les extraits correspondent aux bons passages ; les décalages microphone/système restent cohérents. Examiner les erreurs réelles de segmentation avant de choisir Auto ou 4. |
| Une personne apparaît sous plusieurs groupes | Chaque variante reste écoutable après fusion ; tous ses passages portent le même nom dans l'aperçu. |
| Annuler une fusion, séparer, fermer sans appliquer | Les groupes d'origine reviennent ; aucun nom ou profil permanent n'est sauvegardé par une fermeture. |
| Appliquer sans mémoriser | TXT et JSON cohérents, export neuf corrigé ; aucun profil permanent ajouté. Les documents déjà générés restent à revoir. |
| Nouvelle transcription, fusion puis mémorisation confirmée | Plusieurs variantes sont conservées dans un profil ; aucune mémorisation si la confirmation est refusée. |
| Redémarrage puis autre réunion avec les mêmes personnes | Les profils sont lisibles ; un nom n'est proposé que si les garde-fous passent. Mesurer les faux noms, pas seulement les noms reconnus. |
| Deux personnes parlent simultanément, micro partagé ou fort écho | Examiner les passages ambigus ; ne pas attendre une identification fiable garantie. Corriger ou laisser non attribué. |
| Ancienne revue, fichier supprimé ou fichier audio déplacé | Message explicite, possibilité de choisir l'original, noms encore modifiables ; aucune fausse lecture d'un fichier inventé. |
| Diarisation désactivée | Transcription habituelle, aucune attribution ni reconnaissance vocale nouvelle. |

Ces essais complètent les tests de code ; un test réussi sur une réunion ne garantit pas l'absence d'erreur sur toutes les autres.
