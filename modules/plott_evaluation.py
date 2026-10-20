"""Plot static force errors. These plots do not validate adsorption isotherms."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import matplotlib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.evaluate import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('.'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    if not args.show: matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    df = pd.read_csv(args.input_dir / 'cp2k_model_comparison.csv')
    required = {'model_kind', 'force_error_sum', 'n_atoms', 'configuration_type', 'n_guest'}
    if not required.issubset(df.columns):
        raise ValueError('CSV uses an old schema. Regenerate with modules/evaluate.py first.')
    if df.duplicated(['model', 'candidate_id']).any():
        raise ValueError('Duplicate model/configuration rows')
    sets = df.groupby('model').candidate_id.apply(set).tolist()
    if not sets or any(ids != sets[0] for ids in sets):
        raise ValueError('All models must cover the same test configurations')
    output = args.output_dir or args.input_dir
    output.mkdir(parents=True, exist_ok=True)
    label = 'Mean atomic force-vector error (eV/Å)'
    summary = summarize(df, ['model', 'model_kind']).sort_values('force_mae_ev_A')
    order = summary.model.tolist()
    categories = ['Framework', 'Single CO₂', 'Multiple CO₂']

    def save(filename, note=None):
        if note: plt.figtext(.5, .01, note, ha='center', fontsize=8)
        plt.tight_layout(rect=(0, .05 if note else 0, 1, 1))
        plt.savefig(output / filename, dpi=240)
        if args.show: plt.show()
        plt.close()

    plt.figure(figsize=(11, 5))
    bars = plt.barh(summary.model, summary.force_mae_ev_A)
    plt.gca().invert_yaxis()
    for bar, value in zip(bars, summary.force_mae_ev_A):
        plt.text(value, bar.get_y()+bar.get_height()/2, f' {value:.4f}', va='center')
    plt.xlabel(label); plt.title('Force agreement with CP2K PBE-D3(BJ)')
    save('force_mae_comparison.png', 'Atom-weighted means; uMOF training reference: r2SCAN-D4. No isotherm validation.')

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='model', y='force_mae_ev_A', order=order, color='lightblue')
    sns.stripplot(data=df, x='model', y='force_mae_ev_A', order=order, color='black', size=3)
    plt.ylabel(label); plt.xlabel(''); plt.xticks(rotation=20, ha='right')
    plt.title('Distribution of configuration-mean force errors')
    save('force_mae_boxplot.png', 'One point per configuration; spread is not training-seed uncertainty.')

    grouped = summarize(df, ['model', 'model_kind', 'configuration_type'])
    plt.figure(figsize=(12, 6))
    sns.barplot(data=grouped, x='configuration_type', y='force_mae_ev_A', hue='model',
                order=categories, hue_order=order, errorbar=None)
    plt.xlabel('Configuration type'); plt.ylabel(label)
    plt.title('Atom-weighted force errors by configuration type')
    save('force_mae_by_type.png')

    plt.figure(figsize=(12, 9))
    sns.heatmap(df.pivot(index='candidate_id', columns='model', values='force_mae_ev_A')[order],
                cmap='YlOrRd', annot=True, fmt='.3f', cbar_kws={'label':label})
    plt.title('Mean force-vector error per test configuration'); plt.xlabel(''); plt.ylabel('')
    save('force_error_heatmap.png')

    fine = df[df.model_kind.isin(['finetuned_single', 'finetuned_committee'])]
    if not fine.empty:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=fine, x='configuration_type', y='force_mae_ev_A', hue='model', order=categories)
        plt.axhline(.1, color='red', ls='--', label='Force criterion only: 0.10 eV/Å')
        plt.legend(fontsize=8); plt.xlabel('Configuration type'); plt.ylabel(label)
        plt.title('Fine-tuned single models and committee: configuration errors')
        save('finetuned_force_errors.png', 'The force criterion alone does not indicate full validation.')

    groups = summary.melt(id_vars=['model'], value_vars=['framework_force_mae','co2_force_mae'],
                          var_name='atom_group', value_name='error').dropna(subset=['error'])
    groups.atom_group = groups.atom_group.replace({'framework_force_mae':'Framework', 'co2_force_mae':'CO₂'})
    plt.figure(figsize=(12, 6))
    sns.barplot(data=groups, x='model', y='error', hue='atom_group', order=order, errorbar=None)
    plt.ylabel(label); plt.xlabel(''); plt.xticks(rotation=20, ha='right')
    plt.title('Atom-weighted framework and CO₂ force errors')
    save('framework_vs_co2_force_mae.png', 'CO₂ means exclude configurations with no adsorbate.')
    print(f'Saved plots to {output.resolve()}')


if __name__ == '__main__':
    main()
