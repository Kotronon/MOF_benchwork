# Isotherm Evaluation

Source: `outputs/module_A_co2_isotherm/runs/tao2022_pilot_003/isotherm_summary.json`

## Summary

- Pressure points: 8
- Mean comparison error: 4.26643 mol/kg
- Mean comparison relative error: 51.4358 %

## Isotherm Table

| pressure / bar | CO2 / cell | absolute / mol kg^-1 | excess / mol kg^-1 | reference / mol kg^-1 | compared as | rel. error / % |
|---:|---:|---:|---:|---:|:---|---:|
| 0.1 | 0.129574 | 0.0841536 | 0.080741 |  |  |  |
| 0.5 | 0.683863 | 0.444144 | 0.427047 |  |  |  |
| 1 | 1.42352 | 0.92452 | 0.890241 | 0.61541 | absolute_assumed_for_unknown_reference | 50.2284 |
| 2 | 3.016 | 1.95878 | 1.88987 | 1.38677 | absolute_assumed_for_unknown_reference | 41.247 |
| 5 | 8.87263 | 5.76244 | 5.58748 | 4.08611 | absolute_assumed_for_unknown_reference | 41.0251 |
| 10 | 23.4743 | 15.2457 | 14.8861 | 8.81103 | absolute_assumed_for_unknown_reference | 73.0296 |
| 20 | 35.3799 | 22.9779 | 22.2138 | 15.1313 | absolute_assumed_for_unknown_reference | 51.8575 |
| 35 | 39.817 | 25.8597 | 24.3621 | 17.0999 | absolute_assumed_for_unknown_reference | 51.227 |

## Notes

- Loading is absolute adsorption from the GCMC molecule count.
- Excess loading is calculated as absolute loading minus bulk gas density times pore volume.
- mol/kg and mmol/g have the same numeric value.
- Reference values are matched exactly where possible and otherwise linearly interpolated.
- Excess references are compared against excess loading; unknown-basis references are reported and treated as absolute for comparison.
