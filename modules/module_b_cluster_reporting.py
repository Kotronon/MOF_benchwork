"""English figures and auditable classic/raw uncertainty tables for fixed campaigns."""
import csv
from pathlib import Path
import numpy as np


def write_cluster_report(state, prediction, result, curve, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target = Path(target)
    material = state['material_key'].upper()
    names = state['components']
    def save(fig, name):
        for ext in ('png','svg'):
            fig.savefig(target / f'{name}.{ext}', dpi=160, bbox_inches='tight')
        plt.close(fig)
    observations=[]
    for field, role in (('results','training'),('holdout_results','holdout')):
        for r in state[field]:
            for c in names:
                raw=r['components'][c]
                classic=r.get('classic_seed_components',r['components'])[c]
                observations.append(dict(role=role,candidate_id=r['candidate_id'],pressure_bar=r['pressure_bar'],
                    y_CO2=r['mixture']['CO2'], component=c,mean_molecules=raw['mean_molecules'],
                    classic_standard_error=classic.get('standard_error_molecules'),
                    gp_standard_error=raw.get('standard_error_molecules'),
                    classic_ci95_lower=classic.get('ci95_lower'),classic_ci95_upper=classic.get('ci95_upper'),
                    conservative_ci95_lower=raw.get('ci95_lower'),conservative_ci95_upper=raw.get('ci95_upper'),
                    precision_met=raw.get('precision_met',False),point_precision_met=r['precision_met'],
                    precision_retries=r.get('precision_retries',0),framework_mass_amu=r.get('framework_mass_amu'),
                    source_summary=r['source_summary']))
    if observations:
        with (target/'uncertainties.csv').open('w',newline='') as h:
            w=csv.DictWriter(h,fieldnames=list(observations[0]));w.writeheader();w.writerows(observations)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    ax.scatter([p['pressure_bar'] for p in state['holdout']], [p['mixture']['CO2'] for p in state['holdout']],
               label='Locked holdouts',marker='s',facecolors='none',edgecolors='tab:orange')
    for field,label,marker in [('initial_design','Initial design','x'),('results','Observed training','o')]:
        points=state[field]
        ax.scatter([p['pressure_bar'] for p in points],[p['mixture']['CO2'] for p in points],label=label,marker=marker)
    ax.set(xscale='log',xlabel='Total pressure / bar',ylabel='CO2 gas mole fraction',title=f'{material}: selected states')
    ax.legend();save(fig,'selected_states')
    if prediction:
        pts=state['candidates'];x=[p['pressure_bar'] for p in pts];y=[p['mixture']['CO2'] for p in pts]
        quantities={c:np.array([v['value'] for v in prediction['components'][c]]) for c in names}
        quantities['Total']=sum(quantities.values())
        from modules.module_b_reporting import ratio
        quantities['CO2/N2 selectivity']=np.array([
            ratio(quantities['CO2'][i],quantities['N2'][i],p['mixture']['CO2'],p['mixture']['N2'])
            for i,p in enumerate(pts)],dtype=float)
        fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
        for ax,(c,values) in zip(axes.flat,quantities.items()):
            sc=ax.scatter(x,y,c=values,s=14)
            ax.set(xscale='log',xlabel='Total pressure / bar',ylabel='CO2 gas mole fraction',title=c)
            label = ('Dimensionless (ratio of component medians)' if 'selectivity' in c
                     else 'Sum of component medians / molecules per cell' if c == 'Total'
                     else 'GP median / molecules per cell')
            fig.colorbar(sc,ax=ax,label=label)
        fig.suptitle(f'{material}: surrogate predictions (technical model, external validation open)')
        save(fig,'loading_and_selectivity')
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for ax,c in zip(axes,names):
            values=[(r['upper']-r['lower'])/2 for r in prediction['components'][c]]
            sc=ax.scatter(x,y,c=values,s=14)
            ax.set(xscale='log',xlabel='Total pressure / bar',ylabel='CO2 gas mole fraction',title=c)
            fig.colorbar(sc,ax=ax,label='Latent GP 95% half-width / molecules per cell')
        fig.suptitle(f'{material}: surrogate uncertainty');save(fig,'uncertainty_surfaces')
        if state['holdout_results']:
            index={p['candidate_id']:i for i,p in enumerate(pts)}
            fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
            parity=[]
            for ax,c in zip(axes,names):
                for r in state['holdout_results']:
                    est=prediction['components'][c][index[r['candidate_id']]]
                    obs=r['components'][c];v=obs['mean_molecules']
                    lo,hi=obs.get('ci95_lower'),obs.get('ci95_upper')
                    xerr=[[max(0,v-lo)],[max(0,hi-v)]] if lo is not None and hi is not None else None
                    ax.errorbar(v,est['value'],xerr=xerr,
                                yerr=[[max(0,est['value']-est['lower'])],[max(0,est['upper']-est['value'])]],
                                fmt='o',color='tab:blue' if r['precision_met'] else 'tab:orange')
                    parity.append(dict(component=c,candidate_id=r['candidate_id'],lammps=v,gp=est['value'],
                                       lower=est['lower'],upper=est['upper'],holdout_precision_met=r['precision_met']))
                lim=max(*ax.get_xlim(),*ax.get_ylim(),.01)
                ax.plot([0,lim],[0,lim],'k--',lw=.8)
                ax.set(xlabel='LAMMPS / molecules per cell',ylabel='GP median / molecules per cell',title=c)
            fig.suptitle('Locked holdouts: orange = precision unmet; bars: LAMMPS conservative 95%, GP latent 95%')
            save(fig,'holdout_parity')
            with (target/'holdout_predictions.csv').open('w',newline='') as h:
                w=csv.DictWriter(h,fieldnames=list(parity[0]));w.writeheader();w.writerows(parity)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    curve=sorted(curve,key=lambda r:r['training_points'])
    for c in names:
        valid=[r for r in curve if c in r['metrics']]
        ax.plot([r['training_points'] for r in valid],[r['metrics'][c]['rmse_molecules_per_cell'] for r in valid],'o-',label=c)
    ax.set(xlabel='Training states',ylabel='Holdout RMSE / molecules per cell',title=f'{material}: learning curve')
    if any(r['metrics'] for r in curve): ax.legend()
    else: ax.text(.5,.5,'No holdout labels available yet',ha='center',transform=ax.transAxes)
    save(fig,'learning_curve')
    with (target/'learning_curve.csv').open('w',newline='') as h:
        w=csv.writer(h);w.writerow(['training_states','holdout_states','component','rmse_molecules_per_cell'])
        for r in curve:
            for c in names:
                w.writerow([r['training_points'],r['holdout_points'],c,r['metrics'].get(c,{}).get('rmse_molecules_per_cell')])
    with (target/'report.md').open('a') as h:
        h.write('\nCampaign kind: '+state['campaign_kind']+'.\n\n')
        h.write('Conservative raw-log intervals add between-seed and autocorrelation batch variances and may double-count sampling noise. They do not establish equilibrium. Classic seed intervals are retained in uncertainties.csv and campaign.json.\n\n')
        h.write('Pure-gas boundaries have undefined selectivity; only present-component precision is tested there. Constant traces at positive fugacity fail precision and use a documented 0.05 molecule GP noise floor.\n\n')
        h.write(f"Holdouts failing precision: {sum(not r['precision_met'] for r in state['holdout_results'])}/{len(state['holdout_results'])}. Holdout labels never influence training, acquisition or stopping.\n")
