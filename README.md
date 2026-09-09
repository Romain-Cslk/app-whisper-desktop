# Transcripteur Whisper

Application desktop **Windows 10/11 x64**, interface native **PySide6 / Qt Widgets**.
Transcription locale CPU avec faster-whisper ou API OpenAI, fichiers audio/vidéo, lots,
microphone et son du PC. Aucun navigateur ni serveur HTTP local.

## Utilisateur final

### Installer et lancer

Recevez `TranscripteurWhisper-Setup-1.0.0.exe`, exécutez-le puis ouvrez **Transcripteur Whisper**
depuis le menu Démarrer. L'installation se fait pour votre compte, sans Python, pip ni terminal.
Un raccourci Bureau est proposé. Désinstallation depuis les paramètres Applications de Windows.
Le bundle `dist/TranscripteurWhisper` peut également être transmis en entier ; son EXE ne doit
jamais être séparé du dossier `_internal`.

L'installateur n'est pas signé : Windows peut afficher un avertissement SmartScreen. Une signature
de code nécessite un certificat de l'éditeur et n'est pas incluse dans cette version.

### Transcrire des fichiers

Ajoutez ou déposez plusieurs fichiers, retirez les éléments inutiles, choisissez le mode,
le modèle, la langue et éventuellement le nom de sortie. Formats : AAC, FLAC, M4A, MP3,
MPGA, OGG, WAV, MKV, MOV, MP4, WEBM. Cliquez sur **Démarrer**.

- **Local** : aucune clé requise, calcul sur votre CPU en int8. Le premier usage d'un modèle
  demande son téléchargement ; la progression est affichée et le modèle reste en cache.
  Une fois présent, ce mode fonctionne hors ligne. Les modèles volumineux demandent davantage
  de mémoire et de temps. Le modèle Small présent dans le dépôt est importé lors du développement.
- **OpenAI API** : fournissez votre propre clé dans le champ masqué. Elle reste uniquement
  en mémoire pendant la session. Ce mode transmet l'audio à OpenAI et peut entraîner une facturation.
  Les fichiers sont convertis en MP3 mono 16 kHz, 64 kb/s, découpés sous dix minutes/24 Mo,
  envoyés dans l'ordre puis recomposés avec contexte entre segments.
- Les sorties documentaires API comprennent résumé, compte rendu, note de cadrage, cahier
  des charges, procédure technique, rapport d'analyse et support de formation. Les documents
  longs sont synthétisés par blocs puis recomposés. La transcription reste disponible séparément.

La progression, les étapes par fichier et les messages apparaissent dans la fenêtre. **Annuler**
demande un arrêt coopératif : un appel réseau ou natif déjà engagé peut prendre du temps à rendre
la main. Les résultats déjà écrits restent disponibles, même après une erreur ultérieure.
Les résultats s'ouvrent depuis l'application ; utilisez **Enregistrer sous**, **Ouvrir le dossier**,
la copie de texte ou l'export TXT/ZIP pour les partager.

### Enregistrer et changer de périphérique

Sélectionnez le microphone et la sortie PC à capturer. Les sources peuvent être activées
séparément. Démarrez, vérifiez les vumètres et le compteur, puis arrêtez : le WAV peut être
ajouté aux fichiers à transcrire. Le son PC est capturé par loopback WASAPI ; vérifiez que
Windows joue effectivement l'audio sur la sortie sélectionnée.

Les changements USB/Bluetooth et les périphériques par défaut sont détectés automatiquement,
sans bouton d'actualisation. La sélection reste conservée lorsque le périphérique existe encore.
Hors enregistrement, une sélection disparue est remplacée par le périphérique par défaut puis
par le premier compatible. Pendant la capture, aucun basculement de source : la source perdue
est signalée et l'autre continue si elle fonctionne. Les données déjà capturées sont conservées.

### Données et dépannage

Tout est placé dans `%LOCALAPPDATA%\TranscripteurWhisper` :

```text
config/    préférences sans clé API
models/    modèles persistants, exclus de l'installateur
logs/      app.log et rotations
temp/      conversions et journaux WAV récupérables
results/   transcriptions, documents, enregistrements et historique
```

Les mises à jour et la désinstallation conservent ces données. Après un incident d'enregistrement,
utilisez la récupération proposée dans l'application ; ne supprimez pas les WAV temporaires avant
vérification. Les résultats ne sont pas purgés automatiquement.

En cas d'erreur, ouvrez le dossier des journaux depuis l'application. Pour un microphone absent,
contrôlez les autorisations Microphone de Windows et l'état du périphérique dans les paramètres Son.
Deux endpoints de même nom impossibles à distinguer sont refusés au démarrage de la capture :
donnez-leur des noms distincts dans Windows. Pour un modèle interrompu, relancez le téléchargement ;
contrôlez la connexion et l'espace disque.

## Développeur

### Prérequis et environnement VS Code

Windows x64, Python **3.11 x64**, `uv` et, pour l'installateur, **Inno Setup 6**.
Toutes les commandes suivantes sont lancées à la racine du workspace dans PowerShell :

```powershell
python -m pip install uv
uv sync --locked --group dev --group build --python 3.11
.\.venv\Scripts\python.exe -m transcripteur_whisper
```

Choisissez `.venv\Scripts\python.exe` comme interpréteur VS Code. L'ancien `venv` n'est pas
nécessaire au nouvel environnement. `requirements.txt` reste un point d'installation source via
pip ; `uv.lock` fait foi pour un build reproductible. Les dépendances runtime, tests/lint et build
sont séparées dans `pyproject.toml`.

### Tests, lint et smoke

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m compileall -q src tests packaging
.\.venv\Scripts\python.exe -m transcripteur_whisper --smoke-test --smoke-report build\source-smoke.json
Remove-Item Env:QT_QPA_PLATFORM
```

Les tests utilisent des backends audio et clients API factices. Ils ne nécessitent ni microphone,
ni clé, ni téléchargement de modèle. Le smoke vérifie la fenêtre, les ressources, Qt Multimedia,
les imports natifs, le VAD ONNX et l'encodeur MP3 ; il ne simule pas une transcription réelle.

### Build Windows et installateur

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1
```

Le script synchronise le lock, lance lint/tests/compilation, nettoie uniquement `build` et `dist`,
construit un bundle PyInstaller **onedir sans console**, vérifie les DLL/assets, lance son EXE,
puis compile Inno Setup s'il est présent et écrit les SHA-256 de tous les fichiers distribués.
Pour rendre l'installateur obligatoire ou préciser son compilateur :

```powershell
.\scripts\build_windows.ps1 -RequireInstaller -IsccPath 'C:\Program Files (x86)\Inno Setup 6\ISCC.exe'
```

Sorties : `dist/TranscripteurWhisper/TranscripteurWhisper.exe`,
`dist/installer/TranscripteurWhisper-Setup-1.0.0.exe` et `dist/SHA256SUMS.txt`.
Les rapports smoke et warnings PyInstaller sont dans `build`.
La seule version applicative se trouve dans `src/transcripteur_whisper/__init__.py`.

### Structure

```text
src/transcripteur_whisper/
  app.py, __main__.py       démarrage Qt, smoke, exceptions
  core/                    chemins, configuration, préférences, logs
  models/                  options, jobs et identités audio
  services/                orchestration sans Qt, médias, documents, cache
  integrations/            OpenAI et Whisper local
  audio/                   enregistreur, COM, monitor, récupération
  ui/                      fenêtre, workers, widgets
  assets/                  logos, icône et petites ressources ONNX
packaging/                 PyInstaller et Inno Setup
scripts/                   build local
tests/                     services, audio simulé et Qt
docs/                      architecture, migration et validation
```

Consultez [l'architecture](docs/ARCHITECTURE.md), [la migration](docs/DESKTOP_MIGRATION.md)
et [la validation Windows](docs/VALIDATION.md) pour les choix techniques et essais matériels restants.
Variables avancées conservées : `WHISPER_DATA_DIR`, `WHISPER_MODELS_DIR`, limites de fichiers/jobs,
modèle documentaire, délais et retries OpenAI (voir `core/config.py`). Ne définissez jamais de clé
dans un fichier du dépôt ou un script de build.
