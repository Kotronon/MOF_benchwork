# Isotherm Evaluation

Source: `outputs/module_A_co2_isotherm/runs/tao2022_pilot_003/isotherm_summary.json`

## Summary

- Pressure points: 8
- Mean absolute error: 32.2861 mol/kg
- Mean absolute relative error: 84.0403 %

## Isotherm Table

| pressure / bar | CO2 / cell | loading / mol kg^-1 | reference / mol kg^-1 | rel. error / % |
|---:|---:|---:|---:|---:|
| 0.1 | 0.129574 | 0.0841536 | 0.65935 | -87.2369 |
| 0.5 | 0.683863 | 0.444144 | 2.92703 | -84.8261 |
| 1 | 1.42352 | 0.92452 | 5.76163 | -83.9538 |
| 2 | 3.016 | 1.95878 | 12.5666 | -84.4128 |
| 5 | 8.87263 | 5.76244 | 34.4956 | -83.2952 |
| 10 | 23.4743 | 15.2457 | 83.8255 | -81.8126 |
| 20 | 35.3799 | 22.9779 | 133.165 | -82.7447 |
| 35 | 39.817 | 25.8597 |  |  |

## Notes

- Loading is absolute adsorption from the GCMC molecule count.
- mol/kg and mmol/g have the same numeric value.
- Reference values are matched exactly where possible and otherwise linearly interpolated.
- NIST references may have unknown absolute/excess basis; this is reported in the CSV.
