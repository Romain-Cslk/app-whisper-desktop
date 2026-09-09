# Architecture

```text
QApplication / MainWindow (thread GUI)
    ↓ signaux/slots et workers
JobService (état par instance, snapshots)
    ↓
TranscriptionService
   ↙                  ↘
WhisperLocal          OpenAIService
   ↓                  ↓
ModelService          DocumentService (map/reduce)
           ↘        ↙
      MediaService / PyAV
           ↓
      results / TXT / ZIP
```

```text
QMediaDevices persistant + QTimer secours
    ↓ debounce / scan hors thread GUI
AudioDeviceMonitor
    ↓ snapshot des endpoints SoundCard + indices sounddevice
AudioDeviceService
    ↓ choix stable et résolution au démarrage
RecordingService
    ↓
NativeRecorder / _RecordingSession
    ├── microphone : sounddevice / PortAudio / WASAPI
    └── sortie PC  : SoundCard / WASAPI loopback
    ↓ journaux WAV séparés, mixage ou salvage
résultat WAV persistant
```

Les services métier ne dépendent pas de Qt. La fenêtre assemble des widgets dédiés aux fichiers,
résultats et enregistrement ; les opérations longues passent par des workers qui émettent des signaux.
L'UI ne manipule aucun widget depuis les threads audio, les threads de jobs ou le scan de périphériques.

## Propriété des données

`AppPaths` crée les dossiers utilisateur, identiques en source et après installation. Le package et
le dossier de l'EXE ne contiennent que des ressources en lecture. Les préférences sont autorisées
par liste explicite, sans champ de clé. Chaque job possède options, événements d'annulation, métadonnées
et sorties. Les services maintiennent leur état par instance avec verrous et snapshots copiés.
Les médias source restent à leur emplacement original ; les conversions sont isolées par job.

Les transcriptions sont écrites atomiquement après chaque segment. Un document est une sortie distincte,
ce qui laisse la transcription exploitable si la génération échoue. Les jobs interrompus sont identifiés
au prochain démarrage. Les résultats et les WAV récupérables ne sont pas purgés par leur âge.

## Audio et identité

Qt indique un changement, puis les vrais backends réénumèrent les endpoints : les objets Qt ne
remplacent pas les handles de capture. Les identifiants Windows IMMDevice exposés par SoundCard
servent de clés persistantes. Les microphones sont associés au backend WASAPI sounddevice par
nom normalisé et métadonnées ; un choix ambigu doit échouer explicitement.

PortAudio conserve sa propre liste. La réinitialisation nécessaire pour mettre ses indices à jour
est sérialisée avec le démarrage et n'est permise qu'à l'arrêt des flux. Pendant une capture,
SoundCard continue à voir les endpoints ajoutés/supprimés sans réinitialiser PortAudio. La résolution
finale de l'index se fait au démarrage suivant. Un index ancien ne constitue jamais une identité durable.

Le monitor combine un debounce de 400 ms et un polling de 2,5 s. Un seul scan peut tourner à la fois.
Une liste inchangée ne reconstruit pas les widgets. Une nouvelle source ne modifie pas une capture
en cours. La disparition d'une source marque l'état dégradé, conserve ses données et laisse l'autre
source fonctionner. Les journaux par source permettent le mixage/salvage final et la récupération
après crash.

## Distribution

PyInstaller embarque Python, Qt Widgets/Multimedia et les DLL des moteurs. Aucun WebEngine,
serveur, navigateur ou modèle Whisper n'est inclus. Inno Setup installe par utilisateur, avec
raccourcis et désinstallation. La version provient du package et alimente les métadonnées EXE
et l'installateur. Le build exige les tests puis un smoke de l'EXE produit : un simple exit 0
de PyInstaller est insuffisant.
