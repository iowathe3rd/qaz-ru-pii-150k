from __future__ import annotations

from .config import OntologyConfig
from .constants import OPF_LABELS


class Ontology:
    def __init__(self, config: OntologyConfig) -> None:
        self.config = config
        for label, entry in config.labels.items():
            if entry.opf_label not in OPF_LABELS:
                raise ValueError(f"{label} maps to unknown OPF label {entry.opf_label!r}")

    @property
    def fine_labels(self) -> set[str]:
        return set(self.config.labels)

    def opf_label_for(self, fine_label: str) -> str:
        try:
            return self.config.labels[fine_label].opf_label
        except KeyError as exc:
            raise ValueError(f"Unknown fine label {fine_label!r}") from exc

    def label_for_marker(self, marker: str) -> str | None:
        normalized = marker.upper()
        for label, entry in self.config.labels.items():
            for prefix in entry.marker_prefixes:
                if normalized == prefix.upper() or normalized.startswith(f"{prefix.upper()}_"):
                    return label
        return None

    def validate_marker_label(self, marker: str, fine_label: str) -> None:
        expected = self.label_for_marker(marker)
        if expected is None:
            raise ValueError(f"Marker {marker!r} does not match any ontology marker prefix")
        if expected != fine_label:
            raise ValueError(
                f"Marker {marker!r} implies label {expected!r}, got value label {fine_label!r}"
            )
