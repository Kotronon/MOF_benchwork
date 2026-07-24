# Isotherm Evaluation

Source: `outputs/module_A_co2_isotherm/runs/tao2022_pilot_003/isotherm_summary.json`

## Summary

- Pressure points: 8
- Mean absolute error: 0.755553 mol/kg
- Mean absolute relative error: 55.2707 %

## Isotherm Table

| pressure / bar | CO2 / cell | loading / mol kg^-1 | reference / mol kg^-1 | rel. error / % |
|---:|---:|---:|---:|---:|
| 0.1 | 0.129574 | 0.0841536 | 0.286475 | -70.6245 |
| 0.5 | 0.683863 | 0.444144 | 1.22421 | -63.7199 |
| 1 | 1.42352 | 0.92452 | 2.04195 | -54.7236 |
| 2 | 3.016 | 1.95878 | 2.88118 | -32.0147 |
| 5 | 8.87263 | 5.76244 |  |  |
| 10 | 23.4743 | 15.2457 |  |  |
| 20 | 35.3799 | 22.9779 |  |  |
| 35 | 39.817 | 25.8597 |  |  |

## Notes

- Loading is absolute adsorption from the GCMC molecule count.
- mol/kg and mmol/g have the same numeric value.
- Reference values are matched exactly where possible and otherwise linearly interpolated.
- NIST references may have unknown absolute/excess basis; this is reported in the CSV.
