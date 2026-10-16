"""Assemble stored graph/U1 evidence into manuscript figures; no experiments.

All numerical marks come from saved JSON. E1 is explicitly qualitative because
paired score artifacts have not been located in this checkout.
"""
from pathlib import Path
import hashlib
import json
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/graph_paper_bundle'
FIG = OUT / 'figures'
SOURCES = {}
REGISTER = []
C = dict(ink='#172B3A', dim='#526777', blue='#2368A2', teal='#168477',
         orange='#BD6A31', red='#AE4652', gray='#8C98A3', light='#EDF3F7')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
    'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':C['ink'],
    'text.color':C['ink'],'axes.edgecolor':'#A9B5BE','xtick.color':C['dim'],
    'ytick.color':C['dim'],'figure.facecolor':'white','savefig.facecolor':'white',
    'svg.fonttype':'none','pdf.fonttype':42})


def load(rel):
    path=ROOT/rel
    SOURCES[rel]=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return json.loads(path.read_text(encoding='utf-8-sig'))


def title(fig,head,sub):
    fig.suptitle(head,x=.04,y=.98,ha='left',fontsize=17,fontweight='bold')
    fig.text(.04,.915,sub,ha='left',va='top',fontsize=10,color=C['dim'])


def finish(fig,name,head,caption,sources,placement='main'):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    from matplotlib.text import Text
    outside=[]
    fw,fh=fig.canvas.get_width_height()
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        box=artist.get_window_extent(renderer)
        if box.x0 < -2 or box.y0 < -2 or box.x1 > fw+2 or box.y1 > fh+2:
            outside.append(artist.get_text())
    for ext in ('png','svg','pdf'):
        fig.savefig(FIG/(name+'.'+ext),dpi=190)
    plt.close(fig)
    REGISTER.append(dict(id=name,title=head,caption=caption,sources=sources,placement=placement,
                         out_of_canvas_text=outside))


def dot(ax,labels,values,colors,xlim=(0,100),xlabel='Top-1 accuracy (%)'):
    yy=np.arange(len(labels))
    ax.scatter(values,yy,c=colors,s=65,zorder=3)
    for x,y in zip(values,yy):
        ax.annotate(f'{x:.2f}',(x,y),xytext=(7,0),textcoords='offset points',va='center',fontsize=10)
    ax.set_yticks(yy,labels); ax.set_ylim(len(labels)-.6,-.6)
    ax.set_xlim(*xlim); ax.set_xlabel(xlabel)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6,prune='both'))
    ax.grid(axis='x',color='#E1E7EB',zorder=0)


def forest(ax,labels,values,cis,colors,xlim=None):
    yy=np.arange(len(labels))
    for y,v,ci,col in zip(yy,values,cis,colors):
        ax.plot(ci,[y,y],color=col,lw=2.5)
        ax.scatter([v],[y],s=45,c=col,zorder=3)
        ax.annotate(f'{v:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]',
                    (0.98,y),xycoords=('axes fraction','data'),ha='right',va='bottom',
                    xytext=(0,7),textcoords='offset points',fontsize=8,color=col)
    ax.set_yticks(yy,labels); ax.set_ylim(len(labels)-.45,-.7)
    ax.axvline(0,color=C['gray'],lw=1,ls='--')
    ax.set_xlabel('Accuracy difference (percentage points), 95% interval')
    if xlim: ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6,prune='both'))
    ax.grid(axis='x',color='#E1E7EB')


def overview():
    fig,ax=plt.subplots(figsize=(13,7.3)); ax.set_axis_off()
    title(fig,'Figure 1 | From a graph hypothesis to a history diagnostic',
          'Three scientific questions; the graph is one representation under test, not the claimed method.')
    cards=[('01  Is there useful information?',
            'E1b / E2: representation controls\nG1b: true successor identity vs matched nulls',
            'Signal exists under oracle identity.',C['blue']),
           ('02  Is full structure necessary?',
            'Ladder: cache vs support\nB1 / B2: recurrence, branching, granularity',
            'Extra structural value is not established.',C['teal']),
           ('03  Can the benefit survive online?',
            'D: remove oracle dependencies\nU1: causal belief query + content nulls',
            'The direct fix does not close the gap.',C['red'])]
    for i,(head,body,result,col) in enumerate(cards):
        x=.025+i*.333
        ax.add_patch(FancyBboxPatch((x,.30),.29,.49,boxstyle='round,pad=.012',
                     transform=ax.transAxes,facecolor=C['light'],edgecolor=col,lw=1.5))
        ax.text(x+.015,.735,head,transform=ax.transAxes,fontsize=11,fontweight='bold',color=col)
        ax.text(x+.015,.63,body,transform=ax.transAxes,fontsize=10,linespacing=1.8,va='top')
        ax.text(x+.015,.40,textwrap.fill(result,35),transform=ax.transAxes,
                fontsize=11,fontweight='bold',va='center')
        if i<2:
            ax.annotate('',xy=(x+.323,.55),xytext=(x+.298,.55),xycoords='axes fraction',
                        arrowprops=dict(arrowstyle='->',color=C['gray'],lw=1.5))
    ax.text(.025,.17,'Final thesis',transform=ax.transAxes,fontweight='bold',fontsize=12)
    ax.text(.025,.07,'Evaluate history content, structural opportunity, and reliable identity separately.\n'
            'Recurrence + meaningful branching concern added value beyond a simple successor cache.',
            transform=ax.transAxes,fontsize=12,linespacing=1.6)
    fig.subplots_adjust(top=.87,bottom=.08,left=.035,right=.98)
    finish(fig,'fig01_study_logic','Study logic',
       '概念图，不是测量图。三层分别回答信息、表示复杂度和在线可用性；不将 U1 画成成功方法。',
       ['docs/graph_line_manuscript_status.md','docs/graph_u1_final_result.md'])


def representation(e):
    fig=plt.figure(figsize=(14,7.5)); gs=fig.add_gridspec(1,2,width_ratios=[1,1.75])
    ax=fig.add_subplot(gs[0]); ax.axis('off')
    title(fig,'Figure 2 | Representation effects are not evidence of structural use',
          'E1b qualitative synthesis (left); E2 measured contrasts, n=307 decisions / 49 videos (right).')
    ax.text(0,.94,'A  E1b: input-dependent relation bias',fontsize=12,fontweight='bold')
    for y,label,desc,col in [(.70,'Video observations','Visual continuity\nBias toward SAME',C['blue']),
                            (.38,'Text descriptions','Wording differences\nBias toward DIFFERENT',C['orange'])]:
        ax.add_patch(FancyBboxPatch((0,y-.13),.95,.25,boxstyle='round,pad=.01',facecolor=C['light'],edgecolor=col))
        ax.text(.04,y+.055,label,fontweight='bold',fontsize=11,color=col)
        ax.text(.04,y-.065,desc,fontsize=11,linespacing=1.5)
    ax.text(0,.05,'Qualitative only: no E1b recall values\nare inferred or fabricated here.',fontsize=9,color=C['dim'])
    ax=fig.add_subplot(gs[1]); ax.set_title('B  E2: semantic content controls',loc='left',pad=20,fontweight='bold')
    effects=e['effects']
    labels=['Long shuffled context − no history','True structure − shuffled structure',
            'Semantic labels − permuted labels','Semantic labels − opaque placeholders']
    forest(ax,labels,[r['delta_points'] for r in effects],[r['ci'] for r in effects],
           [C['red'],C['blue'],C['teal'],C['teal']],(-29,12))
    fig.text(.04,.045,'Intervals are stored E2 contrasts. A non-significant semantic-label contrast is not a gain from random labels.\n'
             'The long-context contrast changes more than token count; it is not a pure causal length estimate.',fontsize=9,color=C['dim'])
    fig.subplots_adjust(left=.045,right=.98,top=.80,bottom=.19,wspace=.95)
    finish(fig,'fig02_representation_controls','Representation controls',
       '左侧为 E1b 文档中的偏置方向示意，当前 checkout 缺成对评分产物，故不绘数值。右侧直接取 E2 10-12 重算的四个差值与区间；0.7/2.6pp 是真实语义相对置换/占位符差值，均不显著。',
       ['results/auditor/paper_fig4_e2_measured.json'])


def content(g,l):
    fig,axs=plt.subplots(2,2,figsize=(14,10))
    title(fig,'Figure 3 | Useful successor content does not establish a need for full graphs',
          'Different panels use different tasks. G1b: prior-only ranking; ladder: visual state identification.')
    a=axs[0,0]; a.set_title('A  G1b | 116 decisions / 25 videos',loc='left',fontweight='bold')
    labels=['Uniform prior','Node frequency','Degree-matched shuffle','True successor support']
    keys=['U','F','Rs','Rt']
    dot(a,labels,[100*g['all']['top1'][k] for k in keys],
        [C['gray'],C['orange'],C['gray'],C['blue']],(0,105))
    a=axs[0,1]; a.set_title('B  Which prior information matters?',loc='left',fontweight='bold')
    keys=['Rt-Rs','Rs-U','F-U']
    forest(a,['True − shuffled','Shuffled − uniform','Frequency − uniform'],
           [100*g['all']['contrasts'][k]['delta'] for k in keys],
           [[100*v for v in g['all']['contrasts'][k]['ci']] for k in keys],
           [C['blue'],C['gray'],C['orange']],(-25,60))
    a=axs[1,0]; a.set_title('C  Ladder | 322 decisions / 49 videos',loc='left',fontweight='bold')
    keys=['visual','cache','support']
    dot(a,['Visual','Single successor cache','Full successor support'],
         [100*l['arms'][k]['overall'] for k in keys],[C['gray'],C['teal'],C['blue']],(60,90))
    a.text(.03,.02,'Support − cache: +0.31pp; interval includes zero.',transform=a.transAxes,fontsize=9)
    a=axs[1,1]; a.set_title('D  Attribution of the 19 extra correct decisions',loc='left',fontweight='bold',fontsize=11)
    regs=['no-support','deterministic','branching','novel']
    gains=[round((l['arms']['support'][r]-l['arms']['visual'][r])*l['regimes'][r]) for r in regs]
    bars=a.bar(np.arange(4),gains,color=[C['gray'],C['blue'],C['gray'],C['gray']],width=.55)
    a.set_xticks(np.arange(4),['No support\nn=117','Deterministic\nn=185','Branching\nn=11','Novel\nn=9'])
    a.set_ylabel('Net additional correct decisions'); a.set_ylim(0,24)
    a.bar_label(bars,padding=4); a.grid(axis='y',color='#E1E7EB'); a.set_axisbelow(True)
    a.text(.98,.9,'Branching: visual already 11/11',transform=a.transAxes,ha='right',fontsize=9)
    fig.text(.04,.025,'G1b is not the visual baseline. Frequency below uniform is a point estimate; its interval includes zero.\n'
             'The ladder is a historical oracle-prefix diagnostic; “indistinguishable” is not a proof of equivalence.',fontsize=9,color=C['dim'])
    fig.subplots_adjust(left=.16,right=.98,top=.83,bottom=.12,wspace=.60,hspace=.60)
    finish(fig,'fig03_content_and_complexity','Content and complexity',
       '上排将 G1b 的真实内容与度匹配置换区分；下排展示完整 support 相对单条 cache 未建立额外收益，19题净增益全在确定性 regime。不同任务分母分别标出，不将均匀先验41.8%当视觉基线。',
       ['results/auditor/interaction_graph_g1b_summary.json','results/auditor/interaction_graph_ladder 2.json'])


def opportunity(points,bike):
    fig=plt.figure(figsize=(16,7.6)); gs=fig.add_gridspec(1,3,width_ratios=[1.5,1,1])
    title(fig,'Figure 4 | Apparent branching is not automatically decision-relevant structure',
          'Scenario census at 180s (left); the same 10 manually annotated bike windows under folding (middle/right).')
    a=fig.add_subplot(gs[0]); a.set_title('A  Sparse joint opportunity, not an empty quadrant',loc='left',fontweight='bold',fontsize=11)
    labels={'First Aid - CPR':('CPR',(-7,12)),'Cooking Noodles':('Noodles',(8,9)),
            'Covid-19 Rapid Antigen Test':('Antigen',(8,-14)),
            'Making Cucumber & Tomato Salad':('Salad',(-10,-20))}
    for p in points['points']:
        internal=p['source']=='internal'; human=p['grain']=='human'
        x=p['hit']*100; y=p['p_S_ge_2']*100
        col=C['red'] if internal else C['orange'] if human else C['blue']
        a.scatter(x,y,s=22+np.sqrt(p['n_decisions'])*3.5,c=col,alpha=.65,
                  marker='D' if internal else '^' if human else 'o',edgecolors='white',linewidth=.5)
        if p['label'] in labels:
            name,offset=labels[p['label']]
            a.annotate(name,(x,y),xytext=offset,textcoords='offset points',fontsize=9)
        if internal:
            a.annotate('Internal: 9/11 vacuous',(x,y),xytext=(-145,13),textcoords='offset points',fontsize=9,color=C['red'])
            a.scatter([x],[p['p_g_branch']*100],marker='x',s=55,c=C['red'])
            a.plot([x,x],[p['p_g_branch']*100,y],ls=':',color=C['red'])
        if human and p['n_decisions']==55:
            a.annotate('Bike action-level',(x,y),xytext=(8,0),textcoords='offset points',fontsize=8,color=C['orange'])
    a.set(xlim=(-4,105),ylim=(-.45,10.8),xlabel='Historical support coverage quality (%)\nP(gold in support | support nonempty)',
          ylabel='Decisions with ≥2 observed successors (%)')
    a.grid(color='#E1E7EB'); a.text(.03,.96,'Area grows with decision count\nCircles: keystep; triangles: human\nDiamond: internal; ×: strict G-branch',
                                          transform=a.transAxes,va='top',fontsize=8,color=C['dim'])
    a=fig.add_subplot(gs[1]); a.set_title('B  Folding acquisition-only states',loc='left',fontweight='bold',fontsize=11)
    vals=[bike['steps'][0]['g_branch'],bike['steps'][-1]['g_branch']]
    bars=a.bar([0,1],vals,color=[C['orange'],C['teal']],width=.55)
    a.bar_label(bars,padding=4); a.set_xticks([0,1],['Action-level\nraw labels','Acquisition\nfolded'])
    a.set_ylabel('Strict G-branch decisions (10 windows)'); a.set_ylim(0,6.8)
    a.text(.05,.92,'All 5 raw branches originate\nfrom “pick up a tool”.',transform=a.transAxes,va='top',fontsize=9)
    a.grid(axis='y',color='#E1E7EB'); a.set_axisbelow(True)
    a=fig.add_subplot(gs[2]); a.set_title('C  Which states actually recur?',loc='left',fontweight='bold',fontsize=11)
    real=[v['real_visits'] for v in bike['visits']]; acq=[v['acq_visits'] for v in bike['visits']]
    x=np.arange(5); w=.36
    a.bar(x-w/2,[real.count(i) for i in x],w,label='Work states',color=C['teal'])
    a.bar(x+w/2,[acq.count(i) for i in x],w,label='Acquisition states',color=C['orange'])
    a.axvline(2.5,color=C['red'],ls='--',lw=1); a.set_xticks(x)
    a.set(xlabel='Maximum visits per state in a window',ylabel='Number of windows',ylim=(0,7.8))
    a.legend(frameon=False,fontsize=9); a.grid(axis='y',color='#E1E7EB'); a.set_axisbelow(True)
    fig.text(.04,.04,'The x-axis is conditional support coverage, not literal recurrence frequency. Nominal branching can be vacuous.\n'
          'Granularity claims are local to this probe; recurrence + branching concern added value beyond deterministic cache.',fontsize=9,color=C['dim'])
    fig.subplots_adjust(left=.06,right=.985,top=.80,bottom=.22,wspace=.42)
    finish(fig,'fig04_structural_opportunity','Structural opportunity',
       '左图保留19个点与样本大小差异，内部名义多后继与严格G-branch用连接标记区分；不画“联合区间为空”的误导图。中/右为同10个人工窗口的折叠与访问分布，不把30个keystep窗口当成配对对照。',
       ['results/auditor/b2_egoexo/paper_regime_points.json','results/auditor/b2_egoexo/b2_bike_fold_r1.json'])


def online(d1,d2):
    fig,axs=plt.subplots(1,2,figsize=(13.5,6.8),gridspec_kw={'width_ratios':[1.45,1]})
    title(fig,'Figure 5 | Oracle utility does not guarantee online utility',
          'Experiment D reproduction | 322 decisions / 49 videos | two fusion families shown separately.')
    a=axs[0]; keys=['visual','oracle','prev','full']; yy=np.arange(4)
    for offset,data,col,label in [(-.12,d1,C['blue'],'Two-stage'),(.12,d2,C['orange'],'Soft fusion')]:
        values=[100*data['modes'][k]['identity_top1'] for k in keys]
        a.plot(values,yy+offset,'o-',color=col,label=label,lw=1.2)
        for x,y in zip(values,yy+offset):
            a.annotate(f'{x:.2f}',(x,y),xytext=(6,-3 if offset<0 else 4),textcoords='offset points',fontsize=9)
    a.set_yticks(yy,['Visual','Oracle prefix','Predicted predecessor','Self-maintained\nhistory + prototypes'])
    a.set_ylim(3.6,-.8); a.set_xlim(54,88); a.set_xlabel('Identity top-1 accuracy (%)'); a.grid(axis='x',color='#E1E7EB')
    a.legend(frameon=False,loc='lower right')
    a=axs[1]; x=np.arange(2); w=.32
    for offset,d,col,label in [(-w/2,d1,C['blue'],'Two-stage'),(w/2,d2,C['orange'],'Soft fusion')]:
        vals=[d['modes'][k]['prior_override'] for k in ('oracle','prev')]
        bars=a.bar(x+offset,vals,width=w,color=col,label=label); a.bar_label(bars,padding=4)
    a.set_xticks(x,['Oracle query','Predicted query']); a.set_ylim(0,23)
    a.set_ylabel('Harmful prior overrides (decisions)'); a.grid(axis='y',color='#E1E7EB'); a.set_axisbelow(True)
    fig.text(.04,.04,'Historical D protocol: parameters selected once, not independent outer-CV evaluation; predicted-NEW uses gold alignment.\n'
        'Full changes both transition memory and prototype grouping. It cannot isolate either as the sole cause.',fontsize=9,color=C['dim'])
    fig.subplots_adjust(left=.20,right=.98,top=.81,bottom=.20,wspace=.42)
    finish(fig,'fig05_oracle_online','Oracle-to-online gap',
       '分别绘两阶段与soft融合，不混算；右侧将oracle与预测查询的harmful override并列。Full同时改变转移与原型，不作单因素因果归因。准确率是点估计，未伪造绝对准确率区间。',
       ['results/graph_u1/d_twostage_reproduction.json','results/graph_u1/d_soft_reproduction.json'])


def u1(u,null,soft,bias):
    fig,axs=plt.subplots(1,2,figsize=(14,7.6),gridspec_kw={'width_ratios':[1,1.55]})
    title(fig,'Figure 6 | U1: a small belief gain without established content specificity',
          'Final causal snapshots + train-fold calibration | 5×4 video CV | 322 decisions / 49 videos.')
    a=axs[0]; a.set_title('A  Final query-only conditions',loc='left',fontweight='bold')
    keys=['V','P','B','PG','BG']
    dot(a,['Visual','Point','Belief','Point + gate','Belief + gate'],
        [100*u['summary'][k]['accuracy'] for k in keys],
        [C['gray'],C['gray'],C['blue'],C['orange'],C['orange']],(68,79))
    a=axs[1]; a.set_title('B  The decisive content controls',loc='left',fontweight='bold')
    vals=[100*u['deltas']['B-P']['delta'],
          100*null['comparisons']['support']['real_minus_shuffle_mean'],
          100*null['comparisons']['cache']['real_minus_shuffle_mean'],
          100*soft['contrasts']['support:B-M']['delta'],100*bias['contrasts']['B-K']['delta']]
    cis=[[100*v for v in u['deltas']['B-P']['ci']],
         [100*v for v in null['comparisons']['support']['crossed_seed_video_ci']],
         [100*v for v in null['comparisons']['cache']['crossed_seed_video_ci']],
         [100*v for v in soft['contrasts']['support:B-M']['ci']],
         [100*v for v in bias['contrasts']['B-K']['ci']]]
    forest(a,['Belief − point / visual','True − shuffled support','True − shuffled cache',
              'Belief − point/uniform mixture','Belief − fixed NEW bias'],vals,cis,
              [C['blue'],C['red'],C['red'],C['teal'],C['gray']],(-1.7,5.2))
    fig.text(.04,.04,'Belief adds 4 correct decisions; oracle-prefix memory is still provided. True-vs-shuffled intervals cross zero.\n'
        '20 permutation seeds; null intervals resample seeds × videos, others resample videos. No refit uncertainty. U1 CLOSED; FBG not run.',
        fontsize=9,color=C['dim'])
    fig.subplots_adjust(left=.12,right=.985,top=.80,bottom=.21,wspace=.80)
    finish(fig,'fig06_u1_final_controls','U1 final conditions and controls',
       '主表只用最终因果+校准版本；右图必须和B>V一起出现。真实support 73.91%、打乱均值73.60%，CI跨零；不能解释为uncertainty-aware memory已成功。bootstrap类型标清，停止状态与未运行FBG写在图内。',
       ['results/graph_u1/u1_causal_nested_calibrated.json','results/graph_u1/u1_causal_memory_null.json',
        'results/graph_u1/u1_causal_softening.json','results/graph_u1/u1_causal_known_bias.json'])


def supplementary(audit,cal,sens):
    fig,axs=plt.subplots(1,2,figsize=(13,6.7))
    title(fig,'Supplement S1 | Causal-query calibration and the four rescue cases',
          'Calibration fitted on train videos. Case audit is selected post hoc; it is not an independent test.')
    a=axs[0]
    for key,col,label in [('nll',C['blue'],'NLL'),('ece',C['teal'],'ECE')]:
        before=[s['held_before'][key] for s in cal['selection']]
        after=[s['held_after'][key] for s in cal['selection']]
        a.plot(np.arange(5),before,'o--',color=col,alpha=.5,label=label+' before')
        a.plot(np.arange(5),after,'o-',color=col,label=label+' after')
    a.set(xlabel='Outer video fold',ylabel='Calibration metric (lower is better)',ylim=(0,1.1))
    a.set_xticks(range(5)); a.legend(frameon=False,fontsize=9); a.grid(color='#E1E7EB')
    a=axs[1]; records=audit['audit']
    labels=[r['segment'].replace('mosaic_graph_','M').replace('graph_review_','R') for r in records]
    mass=[next(c['predecessor_probability'] for c in r['candidates'] if c['node']==r['true_predecessor']) for r in records]
    dot(a,labels,[100*m for m in mass],[C['red']]+[C['blue']]*3,(0,105),'Probability mass on true predecessor (%)')
    fig.text(.06,.045,'Only M0015#06 uniquely needs non-top1 predecessor mass. Matched-parameter point lookup also rescues the other three.',fontsize=9,color=C['dim'])
    fig.subplots_adjust(left=.08,right=.98,top=.80,bottom=.20,wspace=.50)
    finish(fig,'supp01_calibration_and_cases','Calibration and four cases',
       '左为因果query校准的各fold NLL/ECE；右为4题正确前驱质量，仅第一题需要非top1查询。非因果早期98.4%接口不混入最终表。',
       ['results/graph_u1/u1_query_causal_calibrated.json','results/graph_u1/u1_causal_rescue_audit.json'],'appendix')
    fig,axs=plt.subplots(1,2,figsize=(13,6.7))
    title(fig,'Supplement S2 | Structural opportunity depends on the observation window',
          'Stored 60 / 90 / 180s scenario measurements; these are not paired predictions of the same decisions.')
    chosen=['First Aid - CPR','Making Cucumber & Tomato Salad','Cooking Noodles','Covid-19 Rapid Antigen Test']
    colors=[C['red'],C['teal'],C['blue'],C['orange']]
    for name,col in zip(chosen,colors):
        p=[next(p for p in sens[w]['points'] if p['label']==name) for w in (60,90,180)]
        for ax,key in zip(axs,['hit','p_S_ge_2']):
            ax.plot([60,90,180],[100*r[key] for r in p],'o-',color=col,label=name.replace('Making Cucumber & Tomato Salad','Cucumber / tomato salad').replace('Covid-19 Rapid Antigen Test','Antigen test'))
    axs[0].set_ylabel('P(gold in support | support nonempty) (%)')
    axs[1].set_ylabel('Decisions with ≥2 observed successors (%)')
    for ax in axs:
        ax.set_xlabel('Window length (seconds)'); ax.set_xticks([60,90,180]); ax.grid(color='#E1E7EB')
    axs[0].legend(frameon=False,fontsize=9)
    fig.subplots_adjust(left=.09,right=.98,top=.80,bottom=.16,wspace=.35)
    finish(fig,'supp02_window_sensitivity','Window sensitivity',
       '从保存的三个窗口敏感性JSON取数，展示覆盖质量与多后继机会对窗口的不同依赖。横轴改变评测单位，不把点连线当相同决策的配对因果干预。',
       [f'results/auditor/b2_egoexo/sensitivity/w{w}_uid.json' for w in (60,90,180)],'appendix')


def gallery():
    tiles=[]
    for item in REGISTER:
        p=FIG/(item['id']+'.png')
        im=Image.open(p).convert('RGB')
        im.thumbnail((780,570))
        tile=Image.new('RGB',(820,625),'#F1F5F8')
        tile.paste(im,((820-im.width)//2,25))
        ImageDraw.Draw(tile).text((22,595),item['id'],fill=C['ink'])
        tiles.append(tile)
    sheet=Image.new('RGB',(1640,625*((len(tiles)+1)//2)),'white')
    for i,tile in enumerate(tiles): sheet.paste(tile,((i%2)*820,(i//2)*625))
    sheet.save(OUT/'contact_sheet.jpg',quality=90)
    from html import escape
    cards=[]
    for item in REGISTER:
        name=item['id']
        cards.append(f'<article><h2>{escape(name)} · {escape(item["title"])}</h2>'
          f'<img src="figures/{name}.svg" alt="{escape(item["title"])}">'
          f'<p>{escape(item["caption"])}</p><p><a href="figures/{name}.pdf">PDF</a> · '
          f'<a href="figures/{name}.svg">SVG</a> · <a href="figures/{name}.png">PNG</a></p></article>')
    page='<!doctype html><meta charset="utf-8"><title>Graph evidence figure book</title><style>'\
        'body{font:16px/1.65 system-ui;max-width:1400px;margin:40px auto;padding:0 30px;color:#172b3a;background:#f3f6f8}'\
        'article{background:white;padding:26px;margin:25px 0;border-radius:12px}img{width:100%}h1{font-size:32px}'\
        '</style><h1>Graph / Interaction History — 实验与图</h1>'\
        '<p>6 张正文图 + 2 张附录图。全部数值来自保存产物；未运行新实验。U1 已关闭。</p>'+''.join(cards)
    (OUT/'index.html').write_text(page,encoding='utf-8')


def tables(joint,bike,e):
    lines=['# Core tables — graph / interaction history','',
        'Generated from saved JSON; no new model runs. See figure_manifest.json for source hashes.','',
        '## Table 1. Structural opportunity by annotation unit','',
        '| Dataset / annotation | Units | Decisions | Support nonempty | Gold in support, conditional | Multi-successor decisions | Strict G-branch | Vacuous / multi-successor |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    aliases=['Internal / contract','Ego-Exo cooking / coarse human','Ego-Exo cooking / keystep','Ego-Exo antigen / keystep']
    for label,r in zip(aliases,joint['rows']):
        vacuous=f'{r["n_vacuous"]}/{r["n_S_ge_2"]}' if r['n_S_ge_2'] else '—'
        lines.append(f'| {label} | {r["n_units"]} | {r["n_decisions"]} | {r["p_support"]*100:.2f}% | '
                     f'{r["hit_given_support"]*100:.2f}% | {r["n_S_ge_2"]} | {r["g_branch"]} | {vacuous} |')
    for label,r in [('Bike action-level / raw',bike['steps'][0]),('Bike action-level / acquisition folded',bike['steps'][-1])]:
        lines.append(f'| {label} | {bike["n_final"]} | {r["n_decisions"]} | — | — | {r["n_S_ge_2"]} | {r["g_branch"]} | — |')
    lines+=['','A dash means not extracted from this result, never zero. The bike comparison uses the same 10 human windows; the 30-window keystep pool is not paired.',
        'branch_b1 is a separate document-supported result: 15 selected clips, 96 decisions, 0 G-branch, 62/62 repeated successors deterministic. It is not a new rerun.','',
        '## Table 2. Oracle dependencies and evaluation interfaces','',
        '| Condition | Previous-state query | Transition history | Prototype grouping | NEW handling | Evaluation |',
        '|---|---|---|---|---|---|',
        '| D visual | None | None used | Oracle prefix | Visual threshold | Historical D selection/reporting |',
        '| D oracle | Gold | Oracle prefix | Oracle prefix | Visual threshold | Historical D selection/reporting |',
        '| D prev | Recursive point | Oracle prefix | Oracle prefix | NEW later aligned to gold ID | Historical D selection/reporting |',
        '| D full | Recursive predicted identity | Predicted | Predicted | Mint predicted node | Historical D; majority-map evaluation |',
        '| U1 V | None | None used | Oracle prefix | NEW candidate logit | 5 outer / 4 inner video folds |',
        '| U1 P/B/PG/BG | Saved causal visual posterior; point or belief | Oracle prefix | Oracle prefix | NEW mass retained / uniform fallback | Nested video CV + train-fold NLL calibration |',
        '| U1 FBG | Not run | Not run | Not run | Not run | Gate not passed; U1 closed |','',
        'D fusion families and U1 protocols are distinct: do not compute cross-protocol method gains.','',
        '## Appendix Table A1. E2 rescored arm accuracies','',
        '| Arm | Accuracy |','|---|---:|']
    for arm,acc in e['accuracy'].items(): lines.append(f'| {arm} | {acc*100:.1f}% |')
    lines+=['','Use measured correction +11.4 → −3.9pp. Earlier prose +20.4 → −6.6pp does not reproduce. Semantic-versus-permuted (+0.7pp) and semantic-versus-opaque (+2.6pp) are non-significant contrasts.','',
        '## Appendix Table A2. U1 result families','',
        '| Result file | Role | Main evidence? |','|---|---|---|',
        '| u1_query_nested.json | Optimistic self-inclusive predecessor prototype | No; exploratory audit |',
        '| u1_query_calibrated.json | Optimistic interface calibration | No; exploratory audit |',
        '| u1_query_diagnostics.json | Optimistic-interface matched/noise analysis | No; not final causal robustness |',
        '| u1_query_causal.json | Causal snapshot before query calibration | Intermediate |',
        '| u1_query_causal_calibrated.json | Calibrated, but frozen P parameters select lambda=0 | Calibration diagnostic, not final method table |',
        '| u1_causal_nested_calibrated.json | Final calibrated causal nested-CV query | Yes |',
        '| u1_causal_rescue_audit.json | Four cases + nonzero matched lambda controls | Appendix mechanism |',
        '| u1_causal_softening.json | Point-uniform / cache controls | Yes, with primary table |',
        '| u1_causal_memory_null.json | True vs successor-identity permutation | Yes; must accompany B>V |',
        '| u1_causal_known_bias.json | Content-free NEW threshold control | Secondary control |','',
        'Full result paths are under results/graph_u1/. These are repeated analyses of the same benchmark, not independent replications.']
    (OUT/'tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    FIG.mkdir(parents=True,exist_ok=True)
    e=load('results/auditor/paper_fig4_e2_measured.json')
    g=load('results/auditor/interaction_graph_g1b_summary.json')
    l=load('results/auditor/interaction_graph_ladder 2.json')
    points=load('results/auditor/b2_egoexo/paper_regime_points.json')
    bike=load('results/auditor/b2_egoexo/b2_bike_fold_r1.json')
    joint=load('results/auditor/b2_egoexo/joint_regime.json')
    d1=load('results/graph_u1/d_twostage_reproduction.json')
    d2=load('results/graph_u1/d_soft_reproduction.json')
    u=load('results/graph_u1/u1_causal_nested_calibrated.json')
    null=load('results/graph_u1/u1_causal_memory_null.json')
    soft=load('results/graph_u1/u1_causal_softening.json')
    bias=load('results/graph_u1/u1_causal_known_bias.json')
    audit=load('results/graph_u1/u1_causal_rescue_audit.json')
    cal=load('results/graph_u1/u1_query_causal_calibrated.json')
    sens={w:load(f'results/auditor/b2_egoexo/sensitivity/w{w}_uid.json') for w in (60,90,180)}
    assert g['n_decisions']==116 and l['n_decisions']==322
    assert round((l['arms']['support']['overall']-l['arms']['visual']['overall'])*322)==19
    assert len(audit['audit'])==4 and len(points['points'])==19
    overview(); representation(e); content(g,l); opportunity(points,bike)
    online(d1,d2); u1(u,null,soft,bias); supplementary(audit,cal,sens); gallery(); tables(joint,bike,e)
    (OUT/'figure_manifest.json').write_text(json.dumps(dict(figures=REGISTER,sources=SOURCES,
       note='Stored evidence only; no new model experiments; E1 qualitative pending paired scores.'),
       ensure_ascii=False,indent=2),encoding='utf-8')
    captions=['# Figure captions / 图注','']
    for item in REGISTER:
        captions += [f'## {item["id"]} — {item["title"]}', '', item['caption'], '',
                     '数据源：'+', '.join('`'+s+'`' for s in item['sources']), '']
    (OUT/'captions.md').write_text('\n'.join(captions),encoding='utf-8')
    print(f'Generated {len(REGISTER)} figures, each in PNG / SVG / PDF; gallery and contact sheet saved.')
    print('Out-of-canvas text:',{r['id']:r['out_of_canvas_text'] for r in REGISTER if r['out_of_canvas_text']})


if __name__=='__main__':
    main()
