# Isotherm Evaluation

Source: `outputs/module_A_co2_isotherm/runs/tao2022_pilot_003/isotherm_summary.json`

## Summary

- Pressure points: 8
- Mean comparison error: 1.99455 mol/kg
- Mean comparison relative error: 26.0368 %

## Isotherm Table

| pressure / bar | CO2 / cell | absolute / mol kg^-1 | excess / mol kg^-1 | reference / mol kg^-1 | compared as | rel. error / % |
|---:|---:|---:|---:|---:|:---|---:|
| 0.1 | 0.129574 | 0.0841536 | 0.080741 |  |  |  |
| 0.5 | 0.683863 | 0.444144 | 0.427047 | 0.320114 | absolute_assumed_for_unknown_reference | 38.7455 |
| 1 | 1.42352 | 0.92452 | 0.890241 | 0.75937 | absolute_assumed_for_unknown_reference | 21.7483 |
| 2 | 3.016 | 1.95878 | 1.88987 | 1.63788 | absolute_assumed_for_unknown_reference | 19.592 |
| 5 | 8.87263 | 5.76244 | 5.58748 | 4.63137 | absolute_assumed_for_unknown_reference | 24.4219 |
| 10 | 23.4743 | 15.2457 | 14.8861 | 10.983 | absolute_assumed_for_unknown_reference | 38.8118 |
| 20 | 35.3799 | 22.9779 | 22.2138 | 19.2277 | absolute_assumed_for_unknown_reference | 19.5043 |
| 35 | 39.817 | 25.8597 | 24.3621 | 21.6519 | absolute_assumed_for_unknown_reference | 19.4337 |

## Notes

- Loading is absolute adsorption from the GCMC molecule count.
- Excess loading is calculated as absolute loading minus bulk gas density times pore volume.
- mol/kg and mmol/g have the same numeric value.
- Reference values are matched exactly where possible and otherwise linearly interpolated.
- Excess references are compared against excess loading; unknown-basis references are reported and treated as absolute for comparison.
