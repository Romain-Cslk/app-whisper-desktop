#ifndef AppVersion
  #error AppVersion must be supplied by build_windows.ps1
#endif

[Setup]
AppId={{74B07C4C-F611-4F6A-B875-BC644AC9DD86}
AppName=Transcripteur Whisper
AppVersion={#AppVersion}
AppVerName=Transcripteur Whisper {#AppVersion}
DefaultDirName={localappdata}\Programs\TranscripteurWhisper
DefaultGroupName=Transcripteur Whisper
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir=..\dist\installer
OutputBaseFilename=TranscripteurWhisper-Setup-{#AppVersion}
SetupIconFile=..\src\transcripteur_whisper\assets\icon.ico
UninstallDisplayIcon={app}\TranscripteurWhisper.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "french"; MessagesFile: "compiler:Languages\French.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Créer un raccourci sur le Bureau"; GroupDescription: "Raccourcis :"; Flags: unchecked

[Files]
Source: "..\dist\TranscripteurWhisper\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Transcripteur Whisper"; Filename: "{app}\TranscripteurWhisper.exe"
Name: "{autodesktop}\Transcripteur Whisper"; Filename: "{app}\TranscripteurWhisper.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\TranscripteurWhisper.exe"; Description: "Lancer Transcripteur Whisper"; Flags: nowait postinstall skipifsilent

; User models, results and recoverable recordings are deliberately not uninstall files.
