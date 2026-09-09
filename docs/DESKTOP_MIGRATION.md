# Migration desktop Windows

## État initial audité

Le dépôt local au commit `5d64581` était propre avant intervention. Python 3.11.9 x64 dans `venv`.
Les **38 tests unittest existants réussissent (0 échec, 3,091 s)** avant toute migration.

`server.py` (environ 2 200 lignes) mélange HTTP FastAPI, protection CSRF, stockage global des jobs,
files de workers, téléchargements Hugging Face, PyAV, Whisper CPU et OpenAI. `main_gui.py` et
`desktop.py` sont deux lanceurs Uvicorn/pywebview. `templates/index.html`, le panneau audio séparé,
`static/app.js` et le CSS portent toute l'interface. Les fichiers `.bak` sont des versions antérieures
des mêmes composants ; les différences ont été inspectées. `pretest.py` et `diagnostic_audio.py`
sont des utilitaires historiques. Aucun script de build ni configuration de package n'était présent.
Les logos/icône existants et les ressources ONNX ont été identifiés. `models/small` contient déjà
un modèle local d'environ 484 Mo, à conserver sur disque mais à exclure de la distribution.

`native_recorder.py` combine sounddevice/PortAudio pour le microphone et SoundCard/WASAPI pour
le loopback. Il conserve des WAV par source, mixe à la fin et dispose de reprises de stop et salvage.
Les tests de concurrence et de session morte/vivante sont pertinents et doivent être conservés.

## Problèmes identifiés

- UI dépendante d'un port HTTP local et d'un navigateur embarqué ; lanceurs dupliqués.
- `sounddevice` effectivement importé mais absent de `requirements.txt`.
- Identité microphone basée sur `sd:<index>` ; aucune gestion automatique des connexions.
- États jobs/modèle/enregistreur globaux et logique métier couplée aux routes HTTP.
- Données écrites dans le dépôt en développement ; résultats soumis à expiration.
- Purge historique des enregistrements pouvant supprimer un WAV récupérable.
- Dépendances runtime/build mélangées ; aucun installateur ni smoke test de bundle.

## Architecture cible et décisions

Qt 6 Widgets/PySide6, sans WebEngine ni serveur. Services métier indépendants de Qt,
jobs en threads et communication avec la fenêtre par signaux. AppPaths place les données dans
`%LOCALAPPDATA%/TranscripteurWhisper` en source comme dans le bundle. Clé API uniquement en mémoire.
Conservation de PyAV, faster-whisper et des deux backends audio existants. PyInstaller onedir
puis Inno Setup par utilisateur. Version unique dans `transcripteur_whisper.__version__`.

À préserver : les 11 extensions, langues et modèles, lots, MP3 mono 16 kHz/64 kb/s,
segments API sous 600 secondes et 24 Mo, contexte ordonné entre segments, sorties partielles,
annulation, les sept documents hiérarchiques, TXT/ZIP, vumètres, mixage et récupération WAV.

## Étapes effectivement réalisées

1. Audit local Git, sources, imports, anciens fichiers, modèles et ressources.
2. Exécution de la suite initiale : 38 réussites.
3. Mise en place du package `src`, des dossiers utilisateur, des préférences sans clé et du logging rotatif.

4. Extraction des médias, jobs, documents, téléchargements et intégrations dans des services sans Qt.
5. Migration du moteur audio et ajout des endpoints stables, du monitor Qt, du debounce/polling et du salvage.
6. Création de MainWindow et de widgets fichiers/enregistrement/résultats ; historique et opérations longues asynchrones.
7. Migration des assertions métier et de coordination ; remplacement des tests HTTP par des tests de services et Qt.
8. Création du pyproject/lock, du build PyInstaller onedir, de l'installateur Inno Setup et du smoke exécutable.
9. Suppression de l'ancien serveur, des deux lanceurs webview, des templates/JS/CSS et des sauvegardes `.bak`.
   Les ressources graphiques/ONNX ont été copiées dans le package et leur identité binaire vérifiée avant suppression des doublons.
10. Transcription locale réelle hors ligne réussie avec le modèle Small du workspace et une voix synthétique Windows.
11. README et documentation d'architecture/validation réécrits ; aucune action commit/push/PR.

## Inventaire

Créés : `pyproject.toml`, `uv.lock`, `src/transcripteur_whisper` (core/models/services/integrations/audio/ui/assets),
`packaging/TranscripteurWhisper.spec`, `packaging/installer.iss`, `packaging/launcher.py`,
`scripts/build_windows.ps1`, `scripts/validate_local_model.py`, les documents `docs/` et les suites
`test_core`, `test_job_service`, `test_audio_devices`, `test_audio_monitor`, `test_audio_recovery`,
`test_recording_service`, `test_desktop_ui`.

Modifiés : `.gitignore`, `requirements.txt`, `README.md` et les tests existants médias/documents/enregistreur.
Supprimés : `server.py`, `server.py.bak`, `main_gui.py`, `desktop.py`, `native_recorder.py` à la racine,
`pretest.py`, `diagnostic_audio.py`, anciens contenus de `static/`, `templates/`, doublons `assets/` et
`tests/test_http_api.py`. Le moteur `native_recorder` et ses tests sont conservés dans le package.
Le modèle local `models/small` et les anciennes données utilisateur du dépôt n'ont pas été supprimés.

## Différences assumées

Les téléchargements HTTP de résultats deviennent des actions desktop et un historique persistant.
Les limites de requêtes et jetons CSRF disparaissent avec le serveur ; limites de fichiers/jobs et
validation d'extensions restent testées côté services. La clé vient du champ mémoire uniquement.
Les résultats ne sont plus purgés après 24 h. L'import d'un ancien modèle est effectué dans le cache
utilisateur depuis les sources ; aucun modèle de transcription n'est ajouté à l'installateur.
Le défaut API devient `gpt-4o-transcribe` ; le modèle historique reste proposé.
Les téléchargements Hub sont séquentiels, révisés par commit et reprenables ; Xet est désactivé par défaut
pour conserver les callbacks d'annulation HTTP et éviter un pool de téléchargement bloquant la fermeture.
