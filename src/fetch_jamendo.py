"""
Descarga tracks de muestra desde la API de Jamendo (licencias Creative Commons).

Requiere un client_id gratuito: https://developer.jamendo.com/v3.0
(crear cuenta -> crear app -> copiar el client_id, no hace falta tarjeta)

Uso:
    python src/fetch_jamendo.py --client-id TU_CLIENT_ID --n 20 --tags rock,jazz

El client_id también puede venir de la variable de entorno JAMENDO_CLIENT_ID
o de un archivo .env en la raíz del repo (línea `JAMENDO_CLIENT_ID=...`).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
META_PATH = RAW_DIR / "metadata.json"

API_URL = "https://api.jamendo.com/v3.0/tracks/"


def resolve_client_id(from_arg: str | None) -> str:
    """Prioridad: --client-id > variable de entorno > archivo .env en la raíz."""
    if from_arg:
        return from_arg
    if os.environ.get("JAMENDO_CLIENT_ID"):
        return os.environ["JAMENDO_CLIENT_ID"]
    env_file = ROOT_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "JAMENDO_CLIENT_ID" and value.strip():
                return value.strip().strip('"').strip("'")
    raise SystemExit(
        "Falta el client_id de Jamendo. Pasalo con --client-id, exportá "
        "JAMENDO_CLIENT_ID, o creá un .env con JAMENDO_CLIENT_ID=... "
        "(se consigue gratis en https://developer.jamendo.com/v3.0)"
    )


def fetch_tracks(client_id: str, n: int, tags: str | None = None) -> list[dict]:
    params = {
        "client_id": client_id,
        "format": "json",
        "limit": n,
        "include": "musicinfo",
        "audioformat": "mp32",  # mp3 320kbps si está disponible
        "boost": "popularity_total",
    }
    if tags:
        # `fuzzytags`, no `tags`: el filtro exacto de Jamendo devuelve resultados
        # erráticos (a veces 0 para tags que existen). Fuzzy matchea sobre el
        # conjunto de tags del track y es estable.
        params["fuzzytags"] = tags

    resp = requests.get(API_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if data["headers"]["status"] != "success":
        raise RuntimeError(f"Jamendo API error: {data['headers']}")

    return data["results"]


def download_track(track: dict, out_dir: Path) -> Path | None:
    audio_url = track.get("audio")
    if not audio_url:
        return None

    track_id = track["id"]
    filename = f"{track_id}_{track['artist_name']}_{track['name']}.mp3"
    # sanitize filename
    filename = "".join(c if c.isalnum() or c in " -_." else "_" for c in filename)
    out_path = out_dir / filename

    if out_path.exists():
        return out_path

    r = requests.get(audio_url, timeout=60)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    return out_path


def save_metadata(metadata: list[dict]) -> None:
    META_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    # La consola de Windows usa cp1252 por defecto y explota con títulos en
    # japonés/cirílico/etc. Forzar UTF-8 con reemplazo de lo que no se pueda mostrar.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", default=None, help="Jamendo client_id (o env JAMENDO_CLIENT_ID / .env)")
    parser.add_argument("--n", type=int, default=20, help="Cantidad de tracks a bajar")
    parser.add_argument("--tags", default=None, help="Tags separados por coma, ej: rock,jazz,electronic")
    args = parser.parse_args()
    client_id = resolve_client_id(args.client_id)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Buscando {args.n} tracks en Jamendo (tags={args.tags})...")
    tracks = fetch_tracks(client_id, args.n, args.tags)
    print(f"Encontrados {len(tracks)} tracks. Descargando...")

    # Acumular sobre corridas anteriores (ej: una corrida por género) en vez
    # de pisar el metadata.json. Se deduplica por id de Jamendo.
    metadata = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else []
    known_ids = {m["id"] for m in metadata}
    added = 0

    for i, track in enumerate(tracks, 1):
        if track["id"] in known_ids:
            print(f"  [{i}/{len(tracks)}] YA EXISTE: {track['artist_name']} - {track['name']}")
            continue

        path = download_track(track, RAW_DIR)
        if path is None:
            print(f"  [{i}/{len(tracks)}] SIN AUDIO: {track['name']}")
            continue

        metadata.append({
            "id": track["id"],
            "artist": track["artist_name"],
            "title": track["name"],
            "album": track.get("album_name", ""),
            "genre": track.get("musicinfo", {}).get("tags", {}).get("genres", []),
            # con qué tag se buscó: etiqueta "declarada" para validar después,
            # NUNCA entra como feature al modelo
            "query_tags": args.tags,
            "duration_sec": track.get("duration"),
            "file": path.name,
        })
        known_ids.add(track["id"])
        added += 1
        save_metadata(metadata)  # incremental: un crash a mitad no pierde lo ya bajado
        print(f"  [{i}/{len(tracks)}] OK: {track['artist_name']} - {track['name']}")
        time.sleep(0.2)  # ser amable con la API

    save_metadata(metadata)
    print(f"\nListo. {added} tracks nuevos (catálogo total: {len(metadata)}) en {RAW_DIR}")
    print(f"Metadata guardada en {META_PATH}")


if __name__ == "__main__":
    main()
