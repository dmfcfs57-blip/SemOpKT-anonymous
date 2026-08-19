# Anonymous release checklist

Before uploading the repository:

1. Run `python scripts/audit_repository.py --root .`.
2. Confirm raw data, model caches, checkpoints, logs, and generated predictions
   are ignored.
3. Confirm no personal name, affiliation, e-mail, ORCID, private hostname,
   username, home path, Windows drive path, or submission identifier appears.
4. Remove repository hosting history that exposes identity, or publish from a
   venue-approved anonymous service.
5. Keep method and dataset citations in the manuscript; do not add author
   identity to the code archive until double-blind review ends.

The audit writes only a local JSON report under `checksums/generated/`; that
directory is ignored by version control.

