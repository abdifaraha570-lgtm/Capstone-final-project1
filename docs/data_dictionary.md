# Data Dictionary - v0.3 (updated in Module 3)

## Source: Kaggle proxy (Hoppen, 2016) - CSV, UTF-8, static snapshot, 110,527 rows x 14 columns
See Module 2 Appendix D for the original field list. Module 3 changes:

## Processed model table (`appointments_model_table`)
| Column | Type | Derived from | Notes |
|---|---|---|---|
| appointment_key | str(16) | AppointmentID | salted hash |
| patient_key | str(16) | PatientId | salted hash, grouping key for split |
| scheduled_date, appointment_date | date | ScheduledDay, AppointmentDay | time removed |
| gender | F/M/U | Gender | U = invalid/unknown |
| age | int 0-90 | Age | invalid dropped, top-coded |
| age_band | cat | age | 0-5, 6-17, 18-34, 35-54, 55-69, 70+ |
| area_code | cat | Neighbourhood | k=20 generalization |
| scholarship, hypertension, diabetes, alcoholism, sms_received | 0/1 | source flags | |
| has_disability | 0/1 | Handcap > 0 | levels 1-4 collapsed |
| chronic_count | int | hypertension+diabetes+alcoholism | |
| lead_days, lead_band, same_day | int/cat/0-1 | appointment_date - scheduled_date | RCA scheduling-lag hypothesis |
| appt_weekday, appt_month | cat/int | appointment_date | |
| prior_visits, prior_no_shows, prior_no_show_rate, first_visit | num | patient history self-join | uses only earlier visits (no leakage) |
| no_show | 0/1 | No-show ("Yes" = missed) | target |

## Future local JJUSHYCSH contract
Unchanged from Module 2 Appendix D. Outreach log joins on `appointment_key` (LEFT JOIN, one-to-one).
Expected refresh: daily extract from HMIS/DHIS2 or OPD system, 18:00, for 24-72 h scoring.
