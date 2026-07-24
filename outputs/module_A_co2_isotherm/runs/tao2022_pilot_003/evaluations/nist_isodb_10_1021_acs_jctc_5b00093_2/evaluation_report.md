# Isotherm Evaluation

Source: `outputs/module_A_co2_isotherm/runs/tao2022_pilot_003/isotherm_summary.json`

## Summary

- Pressure points: 8
- Mean absolute error: 0.95724 mol/kg
- Mean absolute relative error: 7.40656 %

## Isotherm Table

| pressure / bar | CO2 / cell | loading / mol kg^-1 | reference / mol kg^-1 | rel. error / % |
|---:|---:|---:|---:|---:|
| 0.1 | 0.129574 | 0.0841536 |  |  |
| 0.5 | 0.683863 | 0.444144 | 0.496151 | -10.4822 |
| 1 | 1.42352 | 0.92452 | 0.9464 | -2.31187 |
| 2 | 3.016 | 1.95878 | 2.01034 | -2.56464 |
| 5 | 8.87263 | 5.76244 | 6.16969 | -6.60083 |
| 10 | 23.4743 | 15.2457 | 14.5377 | 4.86987 |
| 20 | 35.3799 | 22.9779 | 20.7655 | 10.6546 |
| 35 | 39.817 | 25.8597 | 22.6121 | 14.3619 |

## Notes

- Loading is absolute adsorption from the GCMC molecule count.
- mol/kg and mmol/g have the same numeric value.
- Reference values are matched exactly where possible and otherwise linearly interpolated.
- NIST references may have unknown absolute/excess basis; this is reported in the CSV.
