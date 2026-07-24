# Isotherm Evaluation

Source: `outputs/module_A_co2_isotherm/runs/tao2022_pilot_003/isotherm_summary.json`

## Summary

- Pressure points: 8
- Mean comparison error: 0.68022 mol/kg
- Mean comparison relative error: 16.5446 %

## Isotherm Table

| pressure / bar | CO2 / cell | absolute / mol kg^-1 | excess / mol kg^-1 | reference / mol kg^-1 | compared as | rel. error / % |
|---:|---:|---:|---:|---:|:---|---:|
| 0.1 | 0.129574 | 0.0841536 | 0.080741 |  |  |  |
| 0.5 | 0.683863 | 0.444144 | 0.427047 | 0.72967 | absolute_assumed_for_unknown_reference | -39.1309 |
| 1 | 1.42352 | 0.92452 | 0.890241 | 1.3445 | absolute_assumed_for_unknown_reference | -31.2369 |
| 2 | 3.016 | 1.95878 | 1.88987 | 2.57416 | absolute_assumed_for_unknown_reference | -23.9062 |
| 5 | 8.87263 | 5.76244 | 5.58748 | 6.26314 | absolute_assumed_for_unknown_reference | -7.99442 |
| 10 | 23.4743 | 15.2457 | 14.8861 | 15.7656 | absolute_assumed_for_unknown_reference | -3.29757 |
| 20 | 35.3799 | 22.9779 | 22.2138 | 22.5934 | absolute_assumed_for_unknown_reference | 1.70203 |
| 35 | 39.817 | 25.8597 | 24.3621 | 23.8242 | absolute_assumed_for_unknown_reference | 8.54394 |

## Notes

- Loading is absolute adsorption from the GCMC molecule count.
- Excess loading is calculated as absolute loading minus bulk gas density times pore volume.
- mol/kg and mmol/g have the same numeric value.
- Reference values are matched exactly where possible and otherwise linearly interpolated.
- Excess references are compared against excess loading; unknown-basis references are reported and treated as absolute for comparison.
