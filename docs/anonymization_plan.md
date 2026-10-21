# Data Anonymization Plan (Module 3)

Builds on the Module 2 privacy plan and Ethiopia's Personal Data Protection Proclamation No. 1321/2024.

| Field | Classification | Treatment | Where enforced |
|---|---|---|---|
| PatientId | Direct identifier | Salted SHA-256 -> 16-char `patient_key`; original dropped | `anonymize.py` |
| AppointmentID | Direct identifier | Salted SHA-256 -> `appointment_key`; original dropped | `anonymize.py` |
| Name, phone, MRN, kebele, national ID (future local extract) | Direct identifiers | Dropped on sight; never written to interim/processed | `anonymize.py`, tested with Faker data |
| Neighbourhood | Quasi-identifier | Coded `AREA_nnn`; areas with < 20 patients merged into `AREA_OTHER` (k=20) | `anonymize.py` |
| Age | Quasi-identifier | Top-coded at 90; `age_band` used in reports | `anonymize.py`, `build_features.py` |
| ScheduledDay time | Quasi-identifier | Clock time removed; calendar date kept | `clean.py` |
| Handcap, Alcoholism | Sensitive | Kept for bias audit; necessity review before Module 4 model use | governance committee |

Salt handling: the salt is read from `NOSHOW_HASH_SALT` (secret store / env), never committed. Rotating the
salt breaks linkage to earlier outputs by design. Re-identification is only possible by the HIS/data officer
with the salt and the restricted raw snapshot.

Pseudonymization is not full anonymization under the Proclamation: processed data is still treated as
personal data (access-controlled, retention-limited). Only the aggregate dashboard feed is treated as non-personal.
