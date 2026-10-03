"""Client wrapper around the Mapbox Directions API."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

import requests

from ..utils.config import Config
from ..utils.logger import setup_logger


class MapboxClient:
    def __init__(self, access_token: str | None = None, timeout: int = 20):
        self.access_token = access_token or Config.MAPBOX_ACCESS_TOKEN
        self.openrouteservice_api_key = Config.OPENROUTESERVICE_API_KEY
        self.timeout = timeout
        self.base_url = "https://api.mapbox.com/directions/v5"
        self.osrm_base_url = "https://router.project-osrm.org/route/v1"
        self.ors_base_url = "https://api.openrouteservice.org/v2/directions"
        self.local_osrm_url = Config.LOCAL_OSRM_URL.rstrip("/")
        self.logger = setup_logger("mapbox_client")
        self.session = requests.Session()
        self.session.trust_env = False

    def _parse_ors_feature(self, feature: dict) -> dict:
        properties = feature.get("properties", {}) or {}
        summary = properties.get("summary", {}) or {}
        steps = []
        for segment in properties.get("segments", []) or []:
            for step in segment.get("steps", []) or []:
                steps.append(
                    {
                        "instruction": step.get("instruction", "Continue"),
                        "distance_m": round(float(step.get("distance", 0.0)), 1),
                        "duration_s": round(float(step.get("duration", 0.0)), 1),
                    }
                )
        return {
            "duration": float(summary.get("duration", 0.0)),
            "distance": float(summary.get("distance", 0.0)),
            "geometry": feature.get("geometry", {}) or {},
            "steps": steps,
        }

    def _request(self, start_coords, end_coords, profile: str):
        coordinates = f"{start_coords[0]},{start_coords[1]};{end_coords[0]},{end_coords[1]}"
        osrm_profile = "driving" if "driving" in profile else profile.split("/")[-1]
        provider_errors: list[str] = []

        if self.local_osrm_url:
            try:
                local_url = f"{self.local_osrm_url}/{osrm_profile}/{coordinates}"
                response = self.session.get(
                    local_url,
                    params={
                        "geometries": "geojson",
                        "overview": "full",
                        "alternatives": "false",
                        "steps": "true",
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json()
                if payload.get("routes"):
                    return payload["routes"][0]
                raise RuntimeError("Local OSRM returned no routes.")
            except Exception as exc:
                provider_errors.append(f"local_osrm={exc}")
                self.logger.warning("Local OSRM request failed: %s", exc)

        if self.openrouteservice_api_key:
            try:
                ors_profile = "driving-car" if "driving" in profile else profile.split("/")[-1]
                response = self.session.post(
                    f"{self.ors_base_url}/{ors_profile}/geojson",
                    json={"coordinates": [[start_coords[0], start_coords[1]], [end_coords[0], end_coords[1]]]},
                    headers={
                        "Authorization": self.openrouteservice_api_key,
                        "Content-Type": "application/json",
                        "Accept": "application/json, application/geo+json",
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json()
                if payload.get("features"):
                    return self._parse_ors_feature(payload["features"][0])
                raise RuntimeError("OpenRouteService returned no routes.")
            except Exception as exc:
                provider_errors.append(f"openrouteservice={exc}")
                self.logger.warning("OpenRouteService request failed: %s", exc)

        try:
            url = f"{self.osrm_base_url}/{osrm_profile}/{coordinates}"
            params = {
                "geometries": "geojson",
                "overview": "full",
                "alternatives": "false",
                "steps": "true",
            }
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            if payload.get("routes"):
                return payload["routes"][0]
            raise RuntimeError("OSRM returned no routes.")
        except Exception as exc:
            provider_errors.append(f"osrm={exc}")
            self.logger.warning("Public OSRM request failed: %s", exc)

        if self.access_token:
            try:
                url = f"{self.base_url}/{profile}/{coordinates}"
                params = {
                    "access_token": self.access_token,
                    "geometries": "geojson",
                    "overview": "full",
                    "alternatives": "false",
                    "steps": "true",
                }
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
                if payload.get("routes"):
                    return payload["routes"][0]
                raise RuntimeError("Mapbox returned no routes.")
            except Exception as exc:
                provider_errors.append(f"mapbox={exc}")
                self.logger.warning("Mapbox fallback failed: %s", exc)

        raise RuntimeError(f"Routing request failed across providers: {'; '.join(provider_errors)}")

    def get_directions(self, start_coords, end_coords, profile: str = Config.DEFAULT_MAPBOX_PROFILE) -> dict:
        route = self._request(start_coords, end_coords, profile)
        if "steps" in route and "geometry" in route and "duration" in route:
            return {
                "duration": route.get("duration", 0.0),
                "distance": route.get("distance", 0.0),
                "geometry": route.get("geometry", {}),
                "steps": route.get("steps", []),
            }
        steps = []
        for leg in route.get("legs", []):
            for step in leg.get("steps", []):
                maneuver = step.get("maneuver", {}) or {}
                instruction = maneuver.get("instruction") or step.get("name") or "Continue"
                modifier = maneuver.get("modifier")
                if modifier and modifier.lower() not in instruction.lower():
                    instruction = f"{instruction} ({modifier})"
                steps.append(
                    {
                        "instruction": instruction,
                        "distance_m": round(float(step.get("distance", 0.0)), 1),
                        "duration_s": round(float(step.get("duration", 0.0)), 1),
                    }
                )
        return {
            "duration": route.get("duration", 0.0),
            "distance": route.get("distance", 0.0),
            "geometry": route.get("geometry", {}),
            "steps": steps,
        }

    def get_route_geometry(self, start_coords, end_coords):
        return self.get_directions(start_coords, end_coords)["geometry"]

    def estimate_eta_minutes(self, start_coords, end_coords) -> float:
        return round(self.get_directions(start_coords, end_coords)["duration"] / 60.0, 2)

    def get_multiple_routes(self, start_coords, destinations_list: Iterable[tuple[float, float]]) -> list[dict]:
        destinations = list(destinations_list)
        results: list[dict | None] = [None] * len(destinations)
        with ThreadPoolExecutor(max_workers=min(6, max(len(destinations), 1))) as executor:
            future_map = {
                executor.submit(self.get_directions, start_coords, dest): idx
                for idx, dest in enumerate(destinations)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    self.logger.warning("Failed to fetch route %s: %s", idx, exc)
                    results[idx] = {"duration": None, "distance": None, "geometry": None, "error": str(exc)}
        return results
