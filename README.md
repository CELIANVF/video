# Serveur multi-flux vidéo

Réception de plusieurs flux JPEG (et audio optionnel) sur TCP, tampon circulaire, enregistrement MP4/AVI (via **ffmpeg** quand disponible), affichage OpenCV ou **PyQt6**, et interface web **MJPEG** optionnelle.

## Prérequis

- Python 3.11+ (module standard `tomllib` pour les fichiers de configuration)
- Dépendances Python : `pip install -r requirements.txt`
- **ffmpeg** dans le `PATH` ou via `imageio-ffmpeg` (déjà listé) pour MP4 H.264 et mux audio
- **PyQt6** : interface `--gui`
- **sounddevice** + PortAudio (ex. Debian : `libportaudio2`) : audio côté `camera.py`
- **PyTurboJPEG** + `libturbojpeg` : encodage/décodage JPEG plus rapide (recommandé)

## Configuration (`video.toml`)

Copier [`video.toml.example`](video.toml.example) vers `video.toml` et adapter. Lancer avec :

```bash
python main.py --config video.toml
```

Les options en ligne de commande **remplacent** les valeurs du fichier pour les paramètres fournis.

Sections utiles :

- `[server]` : `host`, `port`, `frame_rate`, `buffer_duration`
- `[paths]` : `export_dir` (vidéos continues, snapshots PNG, clips, stack)
- `[gui]` : `enabled` (`true` = PyQt6 avec boutons ; `false` = fenêtre OpenCV et raccourcis clavier), `stream_labels`, `stream_order`
- `[logging]` : `level`, `json`
- `[metrics]` : Prometheus sur `http://<host>:<port>/metrics`
- `[web]` : serveur MJPEG + page HTML
- `[client]` : `host`, `port` du serveur pour `camera.py --config` ; optionnel `width` / `height` (résolution d’envoi JPEG, 0 = natif). La faute `hight` est acceptée comme alias de `height`.

## Exemples

Serveur (toutes interfaces, port par défaut 8765) :

```bash
python main.py --host 0.0.0.0 --port 8765
```

Avec interface graphique et exports dans `./sortie` :

```bash
python main.py --gui --export-dir ./sortie
```

Sans TOML, l’interface OpenCV (raccourcis) est utilisée par défaut ; avec `[gui] enabled = true` dans le TOML, PyQt6 est prise par défaut. Forcer l’un ou l’autre : `--gui` / `--no-gui`.

Métriques et aperçu navigateur (MJPEG) :

```bash
python main.py --metrics --web --web-host 127.0.0.1 --web-port 8080
```

Puis ouvrir `http://127.0.0.1:8080/` (flux listés dynamiquement ; chaque flux : `/mjpeg/<id>`).

Client caméra :

```bash
python camera.py --host 192.168.1.10 --name entree --device 0
```

Avec reconnexion automatique, horodatage sur l’image et config :

```bash
python camera.py --config video.toml --name cam1 --overlay-time
```

Logs JSON :

```bash
python main.py --log-json --log-level INFO
```

## Protocole TCP (aperçu)

1. Ligne texte UTF-8 terminée par `\n` : `CAMERA <nom_du_flux>\n` (`<nom_du_flux>` identifie le flux côté serveur).

2. Ensuite, soit **legacy** (sans audio) :
   - chaque image : 4 octets big-endian (longueur JPEG) + données JPEG ;

   soit **V2** (ligne `V2\n`) :
   - optionnel : `AUDIO <sample_rate_hz> <channels>\n` (PCM 16-bit little-endian par paquet)
   - paquets : 1 octet type (`1` = vidéo JPEG, `2` = audio) + 4 octets longueur + charge utile.

Le serveur détecte automatiquement legacy / V2 après l’en-tête `CAMERA` (voir [`video_app/protocol.py`](video_app/protocol.py)).

## Structure du dépôt

- `main.py` — point d’entrée serveur
- `camera.py` — client envoi vidéo (+ audio optionnel)
- `video_app/server.py` — acceptation TCP, décodage, registre de flux
- `video_app/buffer.py` — tampon par flux, REC continu, snapshots / clips
- `video_app/qt_gui.py` — interface PyQt6
- `video_app/web_mjpeg.py` — HTTP MJPEG + `/api/streams`
- `video_app/config.py` — chargement TOML

## Dépannage (Linux / Wayland)

- **`Could not find the Qt platform plugin "wayland"`** (OpenCV + fenêtre `cv2.imshow`) : le binaire OpenCV embarque Qt et cherche des plugins au mauvais endroit. Le projet force **`QT_QPA_PLATFORM=xcb`** au démarrage si la variable n’est pas déjà définie (session XWayland). Pour du Wayland natif, installez les plugins Qt adaptés ou définissez vous-même `QT_QPA_PLATFORM`.
- **Avertissements `QFontDatabase: Cannot find font directory …/cv2/qt/fonts`** : le serveur tente de définir **`QT_QPA_FONTDIR`** vers des répertoires système courants (`/usr/share/fonts/...`) avant de charger OpenCV. S’ils manquent, installez des polices (ex. Debian/Ubuntu : `sudo apt install fonts-dejavu-core`) ou définissez vous-même `QT_QPA_FONTDIR` vers un dossier contenant des `.ttf`.
- **`Aborted` / crash à la fermetre ou après REC** : une cause fréquente était une **course** entre le flux MJPEG et le tampon ; corrigée en copiant l’image avant encodage. Mettez à jour le code ; en dernier recours essayez `opencv-python-headless` **uniquement** si vous n’utilisez pas `cv2.imshow` (sinon il faut la variante GUI).

## Observabilité

- Journalisation : module `logging` (texte ou JSON avec `--log-json`).
- **Prometheus** : compteurs / jauge (flux actifs, images appendues, erreurs décodage JPEG) si `--metrics` et paquet `prometheus_client` installé.
