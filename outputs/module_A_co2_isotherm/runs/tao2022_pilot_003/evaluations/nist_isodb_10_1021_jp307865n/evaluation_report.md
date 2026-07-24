# Isotherm Evaluation

Source: `outputs/module_A_co2_isotherm/runs/tao2022_pilot_003/isotherm_summary.json`

## Summary

- Pressure points: 8
- Mean absolute error: 2.76736 mol/kg
- Mean absolute relative error: 22.7386 %

## Isotherm Table

| pressure / bar | CO2 / cell | loading / mol kg^-1 | reference / mol kg^-1 | rel. error / % |
|---:|---:|---:|---:|---:|
| 0.1 | 0.129574 | 0.0841536 |  |  |
| 0.5 | 0.683863 | 0.444144 |  |  |
| 1 | 1.42352 | 0.92452 |  |  |
| 2 | 3.016 | 1.95878 | 1.82083 | 7.57625 |
| 5 | 8.87263 | 5.76244 | 4.58345 | 25.7227 |
| 10 | 23.4743 | 15.2457 | 10.8708 | 40.2443 |
| 20 | 35.3799 | 22.9779 | 18.9955 | 20.9655 |
| 35 | 39.817 | 25.8597 | 21.6972 | 19.1844 |

## Notes

- Loading is absolute adsorption from the GCMC molecule count.
- mol/kg and mmol/g have the same numeric value.
- Reference values are matched exactly where possible and otherwise linearly interpolated.
- NIST references may have unknown absolute/excess basis; this is reported in the CSV.
