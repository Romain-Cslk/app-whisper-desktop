# Validation Windows V1

## Preuves automatisées

État initial : 38 tests unittest réussis avant migration, dépôt propre au commit `5d64581`.
La suite finale couvre les services extraits, de vrais encodages PyAV, la segmentation/recomposition,
les documents map/reduce, l'annulation, les résultats partiels, le cache, les exports, les identités
audio, les pertes de sources, le mixage, les WAV après crash et l'interface Qt avec backends factices.

Le test Qt de la vraie fenêtre interdit explicitement `socket.socket.bind` : aucune écoute TCP
n'est requise pour afficher l'application. Le smoke source/bundle crée QApplication/MainWindow,
charge logos/icône, Qt Multimedia, Whisper, CTranslate2, PyAV, SoundCard, sounddevice, soundfile,
NumPy, ONNX Runtime et OpenAI ; il instancie le VAD et l'encodeur MP3 puis ferme proprement.

La validation locale réelle est reproductible :

```powershell
.\.venv\Scripts\python.exe scripts\validate_local_model.py --report build\local-model-smoke.json
```

Elle nécessite le modèle `models/small` déjà présent dans ce workspace. Windows System.Speech
génère une phrase française dans un WAV ; JobService effectue la transcription réelle avec
faster-whisper Small, CPU int8, beam size 5 et VAD. Aucun microphone n'est ouvert, aucun téléchargement
réseau n'est autorisé par le script. La transcription est correcte, puis cache/historique/export
sont revérifiés après création d'une nouvelle instance de service. Durée mesurée : 16,66 s.
Preuve conservée dans [validation/local-model-smoke.json](validation/local-model-smoke.json).

L'énumération des vrais backends sur la machine a trouvé quatre entrées et cinq sorties ; les
quatre microphones ont été associés à leur index WASAPI. Cela ne prouve pas les essais physiques
de déconnexion ni la qualité d'un enregistrement matériel.

## Procédure manuelle restante

Suivre cette procédure sur **Windows 10 x64 puis Windows 11 x64**, idéalement dans un compte
ne disposant pas de Python. Préparer un micro USB, un casque USB/Bluetooth, un MP3/WAV et un MP4
contenant de la parole ; disposer de votre propre clé uniquement pour les étapes OpenAI.

1. Exécuter l'installateur, vérifier le raccourci Démarrer et l'option Bureau, puis lancer sans terminal.
   Vérifier l'affichage clair/sombre. Installer une nouvelle fois la même version et vérifier que
   les résultats/modèles restent accessibles. La désinstallation doit laisser les données utilisateur.
2. Sélectionner Small en Local, accepter son téléchargement si absent et vérifier la progression.
   Annuler puis relancer ; interrompre la connexion, la rétablir et relancer. Transcrire MP3/WAV/MP4,
   vérifier TXT, copie, ouverture et ZIP du lot. Relancer l'application hors ligne et réutiliser le modèle.
3. Sélectionner OpenAI, saisir votre clé et un fichier de plus de dix minutes. Vérifier plusieurs segments,
   leur ordre, le contexte, puis un résumé/compte rendu. Annuler après un segment : vérifier que la partie
   disponible peut être exportée. Vérifier qu'aucune clé ne figure dans les journaux/préférences.
4. Laisser l'onglet Enregistrer ouvert, sans micro USB. Brancher le micro **sans toucher à l'application** :
   il doit apparaître en quelques secondes. Le sélectionner. Brancher un second appareil ; le premier
   doit rester sélectionné. Débrancher le second, puis commencer une capture avec le premier : son nouvel
   index éventuel doit être résolu correctement. Répéter avec un casque/micro Bluetooth.
5. Hors capture, changer les périphériques Windows par défaut, activer/désactiver une entrée et une sortie.
   Les listes doivent changer automatiquement. Une sélection toujours présente reste inchangée ; une
   sélection supprimée est remplacée par le défaut Windows puis le premier compatible, avec message.
6. Choisir micro + sortie PC, jouer un son sur cette sortie et parler. Démarrer : les deux vumètres et
   le timer doivent bouger. Arrêter : écouter le WAV et vérifier les deux sources, puis le transcrire.
   Recommencer en micro seul puis en son PC seul.
7. Recommencer avec les deux sources. Débrancher le micro pendant que le son PC continue : pas de crash,
   indication dégradée, aucune substitution du micro, WAV final contenant la partie micro déjà capturée
   et la suite PC. Répéter en perdant la sortie, puis les deux sources. Si le pilote reste bloqué, l'arrêt
   doit être retentable et les WAV temporaires conservés.
8. Lors d'une capture de test, terminer le processus via le Gestionnaire des tâches. Relancer et utiliser
   la récupération proposée. Vérifier les pistes récupérées, puis seulement supprimer les copies inutiles.

Le hot-plug Bluetooth dépend aussi du pilote et des profils mains libres/stéréo Windows. Deux microphones
homonymes impossibles à associer sans ambiguïté sont refusés ; les renommer dans Windows permet de lever
l'ambiguïté. Un pilote qui recrée un endpoint avec un nouvel identifiant peut nécessiter une nouvelle sélection.

## Limites de validation

Aucun appel OpenAI réel n'a été effectué : aucune clé personnelle n'a été utilisée. Le téléchargement
de modèle distant est testé avec backends factices ; l'intégration réelle vérifie le modèle déjà local.
La capture microphone/loopback physique et USB/Bluetooth doivent suivre la procédure ci-dessus.
Un build et son smoke locaux ne remplacent pas l'essai dans un Windows propre sans Python ni outils de build.
Un appel HTTP sans réponse peut retarder l'annulation jusqu'au délai réseau ; les résultats partiels
et les WAV restent conservés. Le chargement initial d'un modèle natif est également coopératif à son retour.
