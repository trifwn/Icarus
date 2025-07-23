ations.applicce erformanigh-pnd new hrkflows axisting woor both eable fon suitntatiis implemethmakes eatures rful JAX f poweibility withAPI compatntained  maiination ofhe combs. Tmmon issueot coublesho troion, andmplementat inalm the origiigrate fro mementation,foil implthe JAX Airly use ou effective help yuldve guide shosihenThis compre
```

UG)logging.DEBl=onfig(leveging.basicClogging
logrt impo
```python
formation
or intailed erring for deerbose loggble v- Enaons
estisugg helpful en contain- they oftor messages  the errsesues
- Uc isAX-specifition for Jta documenAX
- Review Jil conceptsal airfon for generumentatio ICARUS docck theelp

- Che## Getting H
#
   ```
)}"){type(arrayray type: f"Ar
   print()ay.dtype}"type: {arr(f"Array d  print")
 array.shape}ay shape: {(f"Arr
   print  ```pythond types**
 pes an sha arrayheck **C

3.```ta)
   daon(ncti some_jit_fu    result =jit():
   able_th jax.dis
   wiython   ```pbugging**
IT for dee Jablis2. **D   ```


n x**2 retur, x)
      "{}"x = nt(.debug.pri
       jax(x):ug_functionef debx.jit
   d @ja`python
  **
   ``de JITinsir debugging  foprint()`ax.debug.e `j

1. **Using TipsDebugg
###
```
  # Fastay([0.5]))(jnp.arrthicknessirfoil.kness = a   thic(100):
 r i in rangefast
fot calls are bsequen
# Now sumpiles
call co))  # First ray([0.5].arjnpickness(airfoil.th = )
_=100", n_points412("2foil.nacaAirirfoil = Jaxation
a compile JIT Warm up th`python
#.
``fastere  buns willt rbsequenion. Sulat JIT compi due tois normal*: This ution*ol run

**Se on firstformancow per*: Sl*Problem*ssues

*ance Irmfo5. Per``

#### alar
`urns scRet*2)  # ss_dist*ean(thickneurn jnp.met
    r0, 1, 10))nspace(p.liickness(jn airfoil.thss_dist =neick
    thl(params)rfoi create_ai   airfoil =
 :ams)ctive(parr
def objes scala- returnorrect rray

# Curns a))  # Retce(0, 1, 10linspass(jnp.il.thicknen airfo  returams)
  pare_airfoil(= creatirfoil   a):
  ive(paramsectef obj
drns arrayrong - retuython
# W`p
``aluesalar vs scturnn reiocttive fun objecnsure E*:ution***Solts`

d inpuomplex-values real- or cre: grad requirrorpeEblem**: `TyPro

**ssuesadient I## 4. Gr
##
```
)airfoilst(rom_liste_batch_feal.cr JaxAirfoivalid =plits, n_ks, upper_sh_masoords, batcls
batch_cfoior many airng ftch processiuse ba
# Or 4)
er_size=102 buffdinates,or(cofoil JaxAir =ilirfoge_ae
larfer sizufncrease b# Ion
``pyth
`rationsse batch ope size or u buffercreaseution**: In
**Soldetected`
rflow vefer omeError: Buf: `Runti**emrobl
**Psues
3. Memory Is

#### s
```   pasentation
 # Implemints):
    _po n(data,unction_f)
def somes=(1,)ic_argnumax.jit, stat(jial
@parttic_argnumsse statter: u)

# Beta, n_pointsction(da= some_fun
result .n_pointsirfoilnts = a
n_poicontextn JIT es issuy cause i This ma
#
```pythonompilation JIT cnts fortic argumese station**: Uolu**S

e`typof integer te values es of concre 1D sequencust beor: Shapes mrr**: `TypeEoblem
**Pr
 Errorshape### 2. S`

#query_x)
``l.thickness(ss = airfoi
thickne, 1, 10)space(0= jnp.lin_x query
# Correct
query_x)
ess(.thicknrfoilaiss = thickne
0, 1, 10)e( np.linspacy_x =uerg
qonhon
# Wrays
```pytrrare JAX all inputs e ansurlution**: E**Soype`

d JAX ta vali is not ray'>umpy.ndarclass 'nf type <x' orgument 'rror: ATypeE: `**roblemrors

**Pmpilation Er
#### 1. CoSolutions
and ssues on I

### Commbleshooting# Trou
   ```

#utation")ess compin thicknected N values detg: Na"Warninint(       prckness)):
hiisnan(tp.any(jnp.if jnx)
   ness(query_ck airfoil.thithickness =es
   N valus for Naesult  # Check rython
 *
   ```plues* for NaN Va**Check ```

2.
  ne return No    ")
      {e}ror: ed er(f"Unexpect       printe:
    ception as except Ex      urn None
   ret        ")
 {e}rror:  eontida"Valif   print(  s e:
       aordationErrfoilValipt Air     exceme)
  me=naes, nail(coordinat JaxAirfoeturn    r          try:

    ):Airfoil"me="dinates, na(coorsafelyairfoil_ef create_   d
```python**
   puts Earlye Inalidat**V

1. ror Handling
### Er  ```


       )mbda: 0.0      la     nator,
enomimerator / dnuda:     lamb
        1e-10,nominator) >s(dep.ab  jn      nd(
    jax.lax.co     returnr):
  ato, denominumeratorsion(nvief safe_di
   dsonatil opertionadi conlax.cond forjax.e on
   # Us
   ```pythtions**iable Operarent Non-Diffe
2. **Handle```

   ction))jective_fungrad(objit(jax.ad_fn = jax.
   grction tooent fundi JIT the gra

   #tive_valuebjec    return oon
   putatiective comour obj     # Yrams):
  ion(pauncte_fobjectivef
   djit
   @jax.   ```pythons**
nt Functionor GradieUse JIT fon

1. **ati Computient

### Grad```ed)
   pend(flappnts.apariaflapped_v       e)
 angl.flap(0.75,oilairfbase_ped =  flap   , 20]:
    [5, 10, 15e in
   for anglriants = []valapped_ns
   fformatiotransse for
   # Reu)
   _points=200", na("2412l.nacoiil = JaxAirffo_airsey
   bas repeatedlnew airfoilg f creatinstead o  # In
 on   ```pyth
ible**ssls When Po Airfoi2. **Reuse
")
   ```
size:.2%}uffer_airfoil.bs/ointl.n_pairfoiiency: {ory efficnt(f"Mem  pri
 _points}")rfoil.ns: {ail pointint(f"Actuapr)
   er_size}"buff{airfoil.r size: rfoil buffef"Ai   print(
hon   ```pyt**
uffer Sizes **Monitor Bment

1.anagey M Memor##
#  ```

 ffer_size)uery_x, bu n_valid, qplits,r_sperds, upch_coos(batch_thicknesrfoil.bats = JaxAiult_resatchoils)
   bom_list(airfbatch_frte_il.cread = JaxAirfoliplits, n_vaks, upper_stch_mash_coords, bas
   batc operationbatch Use   #
 nd(result)
ppe results.a
      ry_x)ss(queil.thicknesult = airfo     rels:
  airfoiil in    for airfo= []
ults   ress
 erationndividual opead of i  # Inst
 thon`pyble**
   ``hen PossiOperations W3. **Batch    ```

ized
# Auto-s)  oints=500, n_p12"a("24foil.nac JaxAirairfoil =e_rge
   laoptimal sizine ermystem det, let the sfoilsirr large a# Fo)

   r_size=64uffe0, bnts=5poi2412", n_oil.naca("rfAi Jaxil =small_airfomemory
   save to ler buffer pecify smalirfoils, ssmall ar hon
   # Foyt   ```pzes**
ffer Sipriate Buppro*Use A *``

2. )
   `0]
   100, 20ist=[50,  n_points_l      256],
64, 128,ffer_sizes=[     buns(
  atioperommon_orecompile_coilOps.prfAiedJaxOptimizr sizes
   ffe bumonfor compile -com
   # PreoilOps
 rfAiedJaxt Optimiztion imporementa_implaxs.jS.airfoil from ICARUon
    ```pyth*
 erations* Common Ope-compilen

1. **Pr Optimizatioerformance
### P
sceest Practi

## Bhow()
```out()
plt.st_lay
plt.tigh')
('y/cabeli].set_yl   axes[)
 label('x/c's[i].set_x
    axerue)i].grid(Taxes[l')
    .axis('equai]
    axes[.1f})')ues[i]:alη={eta_voil {i+1} (rphed Airftitle(f'Moxes[i].set_    a=2)
, linewidth-', :], 'bds[1ed_coororphs[0, :], morphed_coord.plot(m  axes[i]

  ed]oints_morphh[i, :, :n_pmorphed_batc_coords =  morphed
   ize mask sriginal  # Use o.astype(int)])s1[iatch_mask = jnp.sum(bs_morphedint   n_pordinates
  coo morphed# Extract  :
  (3) rangefor i in(15, 5))

e=, figsiz 3subplots(1,xes = plt.lts
fig, aphed resuisualize mor)

# Vshape}"d_batch.orphehape: {mphed batch s(f"Mor
printes
)
 eta_valumasks2, batch_atch_masks1,oords2, bch_crds1, batatch_cooorph(
    batch_mxAirfoil.batch = Jaorphed_b
m.7])3, 0.5, 0ay([0.rr = jnp.ata_valuespair
es for each nt eta value differehing with morp

# Batchirfoils)target_a_list(_batch_fromeatecrrfoil.xAi2, _, _ = Jatch_masks_coords2, ba
batchrfoils)st(base_airom_lie_batch_fl.creatxAirfoi, _, _ = Jatch_masks1 ba_coords1,rays
batchbatch areate # Cr)
]

 n_points=80",15l.naca("44irfoi,
    JaxAints=80)_po", n("2415nacafoil.xAirJa
    oints=80),015", n_pnaca("0rfoil.JaxAi
    ls = [get_airfoi0)
]

tarts=82", n_poin441ca("il.naJaxAirfo),
    nts=80 n_poi",412a("2rfoil.nac,
    JaxAiints=80)", n_poca("0012Airfoil.na    Jaxils = [

base_airfoemplhing exa# Batch morp()

owplt.sh()
ight_layoutd()

plt.t)
ax2.legenruerid(T
ax2.g')butionsistriThickness Ditle(')
ax2.set_tkness''Thicbel(set_yla
ax2.)el('x/c'x2.set_xlabend()

a1.lege)
axgrid(Trux1.
a'equal')1.axis(ax')
il Shapes('Airfotlex1.set_tiel('y/c')
aax1.set_ylabbel('x/c')
.set_xlae}')

ax1odca_cnael=f'NACA {r=color, lab, :], colockness[iatch_thit(query_x, b   ax2.plotribution
  disess thickn
    # Plot   de}')
 A {naca_coabel=f'NACcolor, l :], color=1, coords[ :],t(coords[0,.plo
    ax1l shapefoilot air
    # Pi]]
   lid[, :n_vai, :coords[ds = batch_ coorrfoil
   r this aites foct coordina Extra  #:
  es, colors))(naca_codumerate(zipcolor) in en_code, aca i, (n
forcodes)))naca_1, len(nspace(0, 0(jnp.li.tab1plt.cm= ors s
colfoillot all air P)

#size=(15, 6)(1, 2, figplotst.sub = pl2)g, (ax1, axfiibutions
trs disknesthic
# Analyze
")pe}has.shicknestch_tape: {backness sh thiint(f"Batch
)

pr.shape[2]coords_x, batch_ queryd,lits, n_vali_spds, upperch_coorss(
    battch_thicknexAirfoil.baess = Jaknhic0)
batch_t 1, 2nspace(0, = jnp.li
query_xiontatompuckness cthih

# Batchape}")_coords.s{batchhape: t(f"Batch srinfoils")
p_codes)} airlen(nacad batch of {reate
print(f"C
)
=100ntspois, n_   naca_codeca4(
 tch_nal.ba= JaxAirfoin_valid s,  upper_splitasks,atch_mrds, booch_c5"]
bat015", "241, "0, "6412" "4412""2412",", 012 = ["0des_cols
nacaoi airf NACAate batch of Crefoil

#xAirort Jation impntame_impleaxls.joi.airfom ICARUS
fryplot as pltlotlib.pimport matppy as jnp
port jax.numthon
im
```pyocessing
 4: Batch Pr### Example`

w()
``plt.sho)')
ngle:.1f}°timal_a{op: nglemal Flap Aarison (Optiomp Cile(f'Airfo_titlnd()
ax.set)
ax.lege='Optimized', labeld'reor='e_colidx, overx=afoil.plot(aairized_
optimal')el='Originblue', labor='eride_colx, ovax=aot(_airfoil.plse
ba 6))ize=(10,bplots(figslt.su= p
fig, ax ge=0.5
)
ss_percentahickne_hinge_tap   flangle,
 le=optimal_flap_ang0.75,
    rcentage=rd_pee_choing  flap_hflap(
  l.foiair = base_foild_airoptimize)
points=1002412", n_foil.naca("ir JaxAoil =s
base_airfized airfoilnd optiml ana origire# Compa

)
plt.show(t()tight_layou

plt.e)Trugrid(tory')
ax2.ce Hisonvergen('Citle)
ax2.set_tue'bjective Valt_ylabel('O2.setion')
axlabel('Itera2.set_xsize=3)
axmarkeres, 'r-o', ivbjecttions, oy(iterailog)

ax2.semd(Trueax1.griion')
le Optimizat('Flap Anget_title
ax1.s') (degrees)gle('Flap Anlabelt_y1.se)
axtion''Itera_xlabel(=3)
ax1.setrsize marke 'b-o',ngles, as,erationt(it.plox1

a)) 5size=(12, 2, fig1,bplots(t.su pl2) =, ax, (ax1

fighistory]imization_opt] for h in objective' = [h['ctivestory]
objezation_hisoptimih in for ngle'] [h['a]
angles = n_historyoptimizatio in ] for heration''its = [h[
iteration as plt
plotpy matplotlib.portory
imzation histimioptualize
# Vis}°")
.2fe:angl{optimal_p angle: mal fla"\nOptit(f

prinflap_angle()e_ry = optimizhistoion_zate, optimimal_angltimization
opRun optiy

# ngle, historeturn a

    r}")val:.6fj={obj_ob, }°le:.2fnggle={a {i}: aneration"Itprint(f             10 == 0:
      if i %
    })
      l)
d_vaoat(grat': flradien       'g     ,
val)oat(obj_ve': flcti    'obje
        angle),at(ngle': flo 'a
           on': i,terati  'i       pend({
   y.ap   histor
          )
  .0, 30.0e, -30(anglipnp.cl angle = j
     ngeraeasonable to rtrain angle Cons
        #  al
       rad_vng_rate * ge - learningl= aangle
        Update angle        #
)
 fn(angle grad_  grad_val =e)
      ve(angle_objectiormancil_perf airfol =j_va
        obations):ange(n_iteri in rr
    fory = []
      histotial_angle
 = ini
    angle."""ion optimizatap anglecent for flesnt dradieSimple g
    """):s=50_iterationrate=0.1, n learning_5.0,e=l_anglitialap_angle(in_f optimizeon
def optimizatintscet de gradienSimple# f}")

:.6adientdient: {gr(f"Gra}")
printue:.6fctive_vale: {objeve valu"Objectif
print(gle}°")le: {test_anFlap ang
print(f"e)
st_angl_fn(te grad
gradient =le)t_ang(tes_objectivence_performa airfoillue =jective_va
obegrees0.0  # d 1st_angle =ation
temput gradient co
# Testjective)
ce_ob_performanilairfox.grad( = ja
grad_fngradientte

# Compuess_termrm + thicknmber_teturn ca  re

  thicknessTarget 12% # 2  ss - 0.12)**pped_thickneterm = (flaess_   thickn
 % camberet +2  # Targ02)**2camber - 0. - original_d_camberm = (flapper_terbe camvation
   ness preserthickse with  increaance camber: baltive   # Objec
 ickness
   ax_thairfoil.md_ flappeckness =lapped_thi    fmax_camber
l.ped_airfoiamber = flap_c  flappedmber
  max_case_airfoil._camber = ba   originalhickness
 g taininle mainthier change wmize cambive: minijectobSimple   #
   )
      =0.5
centagekness_per_hinge_thic   flapngle,
     p_afla flap_angle=
       =0.75,ercentagechord_pp_hinge_   fla    p(
 _airfoil.flabasefoil = _airflappedy flap

    # Appl00)
    ints=1_po2412", nil.naca("= JaxAirfose_airfoil foil
    baate base air
    # Cre
    """s tools.c analysiodynamiwith aer
    ateuld integrwo, you ceracti- in p example a simplified is
    This.
  izationtim angle opion for flaptive funct    Objec"
    ""gle):
p_anbjective(flance_oformaairfoil_per
def il
rfoport JaxAientation implem_imils.jaxrfoICARUS.ais jnp
from mpy at jax.nu
imporimport jax```python

ation
iz-Based Optim: Gradientple 3 Exam`

###:.4f}")
``x_camberoil.matimal_airfopamber: {Resulting cnt(f"
priess:.4f}")ax_thicknal_airfoil.mess: {optimting thicknResulprint(f"150
)

oints= n_pta,_eimal optambered,metric, c    symls(
foirom_two_ph_new_frfoil.morxAioil = Jatimal_airfil
opirfoe optimal ae and analyzreat

# C.3f}")al_eta:imη = {opt: kness10% thicrameter for pahing Optimal morp
print(f" result.x
eta =
optimal_bounded')hod='(0, 1), metive, bounds=ness_objecthickar(tscale_ = minimizlt

resu*2hickness)*- target_tx_thickness orphed.maurn (m    ret
 0.10 =sshickne target_t   150
    )
a, n_points=ed, ettric, cambersymme
        _foils(_from_twoil.morph_new = JaxAirfohedrp""
    mo.".10of 0hickness e target tachievive: ject""Ob    "ve(eta):
ctickness_objedef thi
cknessfic thir specimeter foraing paal morphtimnd op
# Fihow()

plt.sht_layout()

plt.tiga=0.8))phwhite", allor=".3", facecoound,pad=0"rt(boxstyle=  bbox=dic                 ='top',
 calalignment    verti                nsAxes,
s[i].tra=axeransform    t             ",
   mber']:.3f}max_ca{result['"Max c:            f   "
      s']:.3f}\nneshickax_t'mresult[t: {f"Max                     .95,
0.02, 0es[i].text(
        ax")eta']:.1f} = {result['t_title(f"η[i].se   axesrue)
      camber=T],ax=axes[ilot('].pfoilresult['air
        len(axes):
    if i < ther resultvery o):  # Plot e2]::ults[resmerate(enult in r i, resu

folatten()= axes.f))
axes e=(15, 10igsiz 3, f(2,lt.subplots pes =nce
fig, axrphing sequeot moes]

# Plalu_vetata in or e)) ft(etaysis(floaanalrphing_lts = [mo 11)
resupace(0, 1,linsnp.lues = j_vaence
etasequng alyze morphi   }

# Aned
 orphil': m     'airfocamber,
   d.max_morphe': 'max_camber  ,
      knessx_thiched.ma: morps'max_thicknes  'eta,
          'eta':
    turn {
    re )
       ints=150
ta, n_pored, eric, cambesymmet
        two_foils(_from_rph_new.moaxAirfoil= Jphed    mor."""
 ropertiesirfoil pphed a"Analyze mor"
    "a):ysis(etg_anal morphinefts=150)

dpoin"4412", n_ca(xAirfoil.naed = Ja
camber150)points="0012", n_aca(Airfoil.nmetric = Jaxirfoils
symase aCreate boil

# AirfJax import mentationls.jax_impleRUS.airfoiICA
from calar_snimizet mimporimize iscipy.opt
from ps jnmpy a jax.nux
importjamport
i
```pythonon
tih Optimizaitg whin Morpfoilample 2: Air## Ex
```

#
plt.show()out()ay
plt.tight_l)
ue.grid(Trnd()
ax2.lege)
ax2stributions'd Camber DiThickness an.set_title('c')
ax2el('y/t_ylab
ax2.seabel('x/c')ax2.set_xl
el='Camber')lab-', r_dist, 'r-mbex_query, calot(.pess')
ax2cknbel='Thiladist, 'b-',  thickness__query,lot(xtions
ax2.pdistribuPlot

# ")rfoil12 AiACA 24"N1.set_title(rue)
axss=T max_thicknerue,x1, camber=T.plot(ax=al
airfoilrfoi Plot ai

#2, 5))=(12, figsizets(1, bplo= plt.suax2) g, (ax1,
filot resultsry)

# Puee(x_qer_linairfoil.cambist = r_dery)
cambeness(x_quthickoil._dist = airfssckne, 50)
thiinspace(0, 1ery = jnp.ltion
x_qubuistri thickness dte Compu

#:.3f}")ber_locationcamairfoil.max_.4f} at x={ber:_camirfoil.maxer: {ax camb(f"Marint")
pcation:.3f}kness_lo_thic.max x={airfoilat} 4fss:.kne.max_thicil: {airfossknehicax tnt(f"M}")
prioil.n_pointsairfs: {int(f"Point
pr").name}il: {airfoil"Airfot(finprroperties
metric peoze gnaly200)

# Ants= n_poi",ca("2412oil.nairffoil = JaxAirfoil
airA 2412 a NACCreateil

# t JaxAirfoion imporplementatoils.jax_imCARUS.airfom It
frs plt aotlib.pyplort matplimpo as jnp
mpyax.numport j``python
i

`nalysis AAirfoilc : Basile 1mp# Exaples

##

## Examdinates. (x, y) coor surface
```
Lowery, Array]Arra Tuple[lf) ->points(sesurface_f lower_rty
deopen
@pr
```pythotes.
coordina y) (x,r surface
Uppe
```rray, Array]e[Alf) -> Tuplsepoints(urface_upper_sdef roperty
thon
@p`py

``gth.lenord
Ch
```oatf) -> flngth(selle
def chord_roperty`python
@pber.

``imum came of max-coordinat`
X
``f) -> floateltion(samber_loca_cy
def maxpertython
@pro

```pber value.camMaximum loat
```
) -> fr(selfax_cambeperty
def mron
@p```pythoickness.

f maximum thcoordinate oX-oat
```
-> flion(self) s_locatx_thicknesmaerty
def n
@propytho```pe.

ness valumum thick
Maxi``oat
`) -> flickness(selfef max_thrty
dprope
@python
```r size.
urrent buffe
Cint
``` -> f)eluffer_size(sperty
def bn
@proytho``pints.

` valid poer of``
Numb> int
`ints(self) -def n_poroperty
`python
@p``me.

foil na
```
Air -> strme(self)y
def na@propertthon
``pyies

` Propert##ons.

##ptivarious owith the airfoil lot `
Pxes]
``ional[Aptargs
) -> O **kw= None,
   onal[Axes] : Opti,
    axsel = Fal: booess_thickn max   = False,
 catter: bool s
 False,er: bool = f,
    camb    sel(
lotn
def p

```pytho## Plottingat.

###edge formleading l in Save airfoi None
```
e
) ->= Fals bool der:one, hea N =onal[str]tirectory: Op di(
    self,leef save_n
dtho
```pyformat.
n Selig e airfoil iavne
```
S Noalse
) -> bool = Fe, header:onnal[str] = NOptiodirectory: lf,   seig(
  el
def save_spython I/O

```#### Fileon.

#t distributioinnew poil with Repanel airf
```
> JaxAirfoil"
) -er = "cosin stion:stributt = 200, diints: inn_po
    self, epanel(on
def r```pythion.

nsformat tra flap`
Applyfoil
``axAir 1.0
) -> J =oatxtension: flchord_e.5,
    float = 0entage: rc_peessthicknlap_hinge_
    ft,e: floaangl    flap_ float,
rcentage:nge_chord_pe flap_hi,
   elf    slap(
thon
def fons

```pyformatins
##### Trardinates.
ce y-coosurfauery lower ray
```
QArat]) -> floy, rraon[A_x: Uniuerywer(self, q y_loython
defes.

```prdinaty-cooper surface `
Query upray
``]) -> Arloat[Array, f_x: Unionqueryr(self,  y_uppen
def`pytho.

`` pointsne at querycamber lipute ``
Com-> Array
`loat]) y, f[Arranionry_x: Une(self, queer_lief cambpython
d

```ints.query pot bution aness distriickompute th
```
Crrayat]) -> A, floArrayn[_x: Uniolf, querys(sees
def thickn``pythonQueries

`c eometri
##### GMethods
ce ### Instans.

#ng airfoilwo existitween tfoil bemorphed air`
Create oil
``rf
) -> JaxAioints: int n_ploat,a: f etfoil,il2: JaxAirfoil, airrfo: JaxAil1fois(
    air_foilew_from_twoh_noil.morprfd
JaxAihosmethon
@clas```pytrphing

### Mo
##m file.
rfoil froai```
Load irfoil
tr) -> JaxA sme:filena.from_file(axAirfoil
Jodclassmeth
@pythonates.

```face coordinwer sur upper/lom separateeate fro```
CrAirfoil
) -> Jaxgs
l", **kwarirfoi"JaxA= str ame: rray, nay, lower: A: Arrper   upr(
 lowepper_.from_uxAirfoilthod
Jasme
@clas```pythonom Data

frion Construct##

###it).or 5 dig4 tects -deto airfoil (aueate NACA`
CrxAirfoil
`` -> Jawargs) = 200, **kpoints: intr, n_on: stca(designatil.naaxAirfoiod
Jsmeththon
@clasil.

```py-digit airfoeate NACA 5
```
CrJaxAirfoil -> , **kwargs) = 200intints: r, n_po(digits: staca5Airfoil.nmethod
Jaxssn
@clathopy``foil.

`git airNACA 4-di
Create ```foil
) -> JaxAirkwargs = 200, **oints: intn_ps: str, itl.naca4(dig
JaxAirfoiethodlassmthon
@c`py

``onatiACA Gener

##### Nsss MethodCla

#### a metadattionalional addidata`: Opt`metaf None)
- ined io-determze (autfer siOptional bufr_size`: fferfoil
- `buthe aiame of e`: N `nam_points)
-, n (2 formatselignates in coordi: Airfoil nates`oordirs:**
- `camete

**Pare
)
``` Non]] =Anytr, al[Dict[sa: Optionmetadatne,
    ] = Nonal[intiopt_size: Oer",
    bufflrfoiaxAi str = "J,
    name: None]] =ndarray, np.ray[Arional[Uniontes: Opt  coordinail(
  Airfopython
Jaxctor

```truonsss

#### CAirfoil Clae

### Jaxeferenc
## API R
```
ms)airfoil_parad_fn(dients = grax)
graction_jactive_fun(objegradax.n = j_fdients
grad exact graetvalue

# Gective_turn obj   re
 on)tati implemenblempatiAX-co(J  # ... e
  jectiv obomputend cairfoil aeate # Crams):
    oil_parairfjax(ction_ctive_fun
def obje
@jax.jitt jax
ation
imporc differenti: Automati
# Afterradients)
ay(grn np.arr   retu

    adient)s.append(grnt     gradie   h
base_obj) / j_plus - t = (ob     gradien
   s)
      pluams_ion(parctun_fobjective_plus =      obj
   s[i] += hs_pluram   paopy()
     _params.cirfoils = ams_pluara p      s)):
 l_paramirfoi range(len(ar i in
    fo
    dients = []
    gral_params)oition(airfective_func = obj   base_obj=1e-6):
 l_params, hirfoi(aite_diffent_finpute_gradief comferences
dte difal finiefore: Manu
# Bhonn

```pytratiorkflow Migmization Wo
### Opti``
d
`lly converte Automaticas)  #(numpy_coordirfoilrfoil = JaxA]])
ai.05, 0.08, 0 0.0], [0.01.0, 0.5,array([[np.ds = umpy_coor cases)
ntic in mostutoma to JAX (a
# NumPy
(jax_array).array= npy y_arramp)
nu 10)(0, 1,np.linspacethickness(jairfoil.x_array = o NumPy
ja tnp

# JAXx.numpy as jort ja
impnumpy as npys
import ray and JAX areen NumPerting betwhon
# Conv
```pytces
fferenray Type DiHandling Arable

### here applics wapabilitient cradiew g neverage ] Le- [risons
rray compaandle JAX a tests to h[ ] Updatede
-  cocalcritiformance-ion for peratompilnsider JIT cys
- [ ] CoraAX arwork with Je to ing codray handl] Update ar
- [ mentstatece import s Replat

- [ ]is Checklongrati### Mi
#
ficiencyfor efcation fer alloatic bufUses st**Memory**: ilation
4.  JIT compdue toer  may be slowlls First ca**:*Compilations
3. *ew instancens return nnsformatioe; trablmuta imirfoils are*: JAX aility*tabs
2. **Immu arrayPyead of Numrrays instX aJAn returns iomentat JAX impleypes**:rray Ter

1. **Ato Considfferences  Key Di```

####
PI A # Sameot()
airfoil.pl  # Same APIss(0.5)hickneoil.trfckness = aie API
thi0)  # Samts=20", n_poin2naca4("241rfoil.foil = JaxAiir
aAirfoil
ax Jrttion impontamelex_impils.jaairforom ICARUS.
fentation)implem (JAX
# After)
l.plot(
airfoiickness(0.5)ths = airfoil.knes)
thicts=200, n_poin4("2412"Airfoil.naca = airfoilrfoil

rt Aioils impoairfRUS.
from ICAtion)mentanal impleore (Origion
# Befpythnt

```lacemeple Rep# Simard:

###rwn straightfong migratiomaki, tibilityll API compamaintains fumentation le
The JAX imprfoil
xAirfoil to Jainal Airom Orig F
###n Guide
gratio Mi``

##e()
`ry_usage_memo

measur.2f}x")usage:sage/jax__urigficiency: {ory ef"Memoint(f)
    pr2f} MB"ig_usage:.rfoils): {or(1000 aiemory usage l minal Airfoi"Orig(f   printf} MB")
 ge:.2ax_usals): {j000 airfoie (1agry usfoil memo"JAX Air
    print(fory
    _membaselineg_memory - _usage = ori    orig # MB
 1024 rss / 1024 /().ory_infoocess.memry = pr_memo
    orig  l)
 end(airfois.appig_airfoilor0)
        ints=102412", n_pol.naca4("l = Airfoi   airfoi    ):
 000(1ngein ra i   for  oils = []
g_airfrifoils
    oiginal airny orreate ma C #
   ils
foir   del jax_airfoils
  alear JAX
    # Cy
   emor baseline_max_memory - = jx_usage
    ja24  # MB / 1024 / 10o().rssory_infs.memcesy = pro_memor   jax
     airfoil)
d(ppen.ax_airfoils       ja0)
 nts=10412", n_poica("2oil.nal = JaxAirfairfoi        000):
 in range(1
    for i []_airfoils =ils
    jaxfoair JAX Create many    #

     MB/ 1024  #1024 nfo().rss / s.memory_iproces= ry line_memo   basery
 line memo   # Base
 id())
ess(os.getproctil.Pss = psuoce    pr"

s.""mentationeen implege betwmemory usapare Com """
   e():_usagmemorye_sur

def meaport os
imort psutiln
imp```pythoon

ge Comparisory Usa

### Memon()
```atienchmark_cre
b2f}x")
batch_time:.time/jax_tch__baorigeedup: {nt(f"Sp   pri")
 .4f}sime:orig_batch_t: {tion0x crearfoil 10inal Aif"Orig
    print(f}s")me:.4atch_tix_btion: {jaeacrl 100x foi(f"JAX Air   print
 me
- start_ti) r(counte.perf_ time_time =g_batch
    orits=200)12", n_poin"24il.naca4(il = Airfoig_airfo   or     nge(100):
 in ra
    for _counter()me.perf_time = tit_
    starme
   _tier() - startntperf_cou = time.x_batch_time0)
    ja n_points=20","2412rfoil.naca(Ai = Jaxx_airfoil     ja   :
0) range(10for _ inter()
    .perf_coun_time = time
    startn)ompilatiodue to cfaster ld be X shoulls (JA caquentSubse
    #    f}s")
ime:.4tion_t{orig_crea creation: irfoilal At(f"Originprin")
    _time:.4f}sx_creationation: {jaAirfoil cre(f"JAX     printme

 start_tiounter() -perf_cime = time.n_teatio
    orig_crs=200)point2412", n_naca4("irfoil. = Arfoil_aiig   ornter()
 perf_coume.e = titimstart_  ntation
  emeginal impl# Ori

 start_timeunter() - e.perf_co = timeation_timex_cr
    jaints=200)412", n_pooil.naca("2= JaxAirfil airfo jax_   ter()
e.perf_coun_time = tim    startt run)
ime on firs tcompilationn (includes ioimplementatAX  # J

   "n times.""reatiooil ck airfnchmarBe""":
    on()atimark_cref benchtion

dementaal impleil  # Origin Airfols import.airfoiRUS
from ICAxAirfoilrt Jaion impoentatx_implems.jailRUS.airfoe
from ICA
import timpython```on

 Comparislation Timempion

### Coe ComparisPerformanc

## a}")
```{optimal_eter: ng parametrphi mo"Optimalt(fprinresult.x
_eta = mal
)

optiounded'method='b   0, 1),
  bounds=()),
   ta(eivejecthing_oborploat(ma: fambda et  l
  ize_scalar( = minimltalar

resuinimize_scort mmize impcipy.opti
from ser parametorphingtimal m

# Find opkness)**2et_thicss - targal_thickne(actun tur
    reness
   hick.max_torphedckness = mthi   actual_ss = 0.08
 rget_thickne
    taiontributs dis thicknesr specific optimize foExample:
    #
    )
    nts=200ta, n_poirfoil2, eoil1, ai     airfls(
   _two_foiew_from.morph_nil= JaxAirfophed
    mortion."""zaer optimihing parametrpn for moioctive funct """Obje
   eta):ve(ti_objecorphing
def mythonnts

```pdieh Grag wit## Morphin
```

#_coords)tchribution(bass_diste_thickneput = comkness_thicchs)
batpoint(3, 2, n_# Shape: s3])  2, coordrds1, coordsstack([coo = jnp.coords
batch_ of airfoils batchpply to

# Aery_x)s(quil.thicknesirforn a  retu 20)
  (0, 1,p.linspacejnuery_x =
    qoords)oil_cil(airfirfoxA= Jafoil   airords):
  foil_coution(airibss_distrute_thickneompf cvmap
de@jax.airfoils
ltiple ross mu acionsorize operatct
# Ve```python vmap

thrization wiecto`

### V}")
``ntss: {gradiedient"Gra
print(frams)
_fn(pa grad
gradients =2]) 0.4, 0.1[0.02,ray(.ars = jnpparamobjective)
(airfoil_ = jax.gradd_fnrats
gadiente gr

# Compu02)**2r - 0._cambe*2 + (max0.12)*s - cknesx_thin (maetur    r
camberaintaining hile mckness we thiinimizive: mobjectle amp # Ex

 rbecammax_airfoil.ber =    max_camckness
 x_thiairfoil.mass = knemax_thicties
    mpute proper   # Co

 ords)oil(base_coAirf Jax =foil
    air 0.0]])
  T, -0.05*M,5*M, 0.08*0.0     [0.0,
    .0],0.0, 0.5, 1[1.0, 0.5, array([oords = jnp. base_c
   rturbationpenate ordiuse co, we'll demo For  #
   ble NACA)ifferentiar d foontiimplementaneed custom is would  (thairfoile NACA    # Creatams

 T = naca_par P, ss]
    M, thickneposition,r, camber_cambe [max__params =  # naca"
  zation.""foil optimiirn for ave functiobjecti"""O
    ms):(naca_paraveil_objectie
def airfoion exampltatient compu# Gradort jax

python
imp

```nrentiatiofematic Difuto```

### As)
ordrties(coil_propee_airfo= computamber ness, c
thick0.0]])5, 0.0 - 0.05, 0.08,[0.0,           ,
        .0]5, 10.5, 0.0, 0.[[1.0, array( jnp.oords = arrays
cUse with JAXer

# mbss, caknen thicetur
    r5]))
    , 0.7 0.5.25,p.array([0ne(jnl.camber_li= airfoier   camb  5]))
 0.5, 0.7rray([0.25,ess(jnp.ahicknrfoil.tess = aikne
    thicompatibl-cre JIToperations a
    # All    ords)
 l_col(airfoiJaxAirfoi = airfoiles
     coordinatairfoil from Create ):
    #dsoil_coorerties(airfop_airfoil_prpute com
defax.jittions
@joil operaile airfJIT comp# ax


import j
```pythonlation
 Compi

### JITic Featuresecif JAX-Sp

##"]
)
```greenlue", ""red", "blors=[    co
12"],NACA 44"A 0012", 412", "NACACA 2"N labels=[ls,
      airfoifoils(
 air_multiple_lot = plotter.pigtter()
ffoilPloirlotter = A
pter
irfoilPlotmport A ionmentatiax_imples.jRUS.airfoil
from ICAPlotterfoiling with Airvanced plotte)

# Adegend=Truue, show_lthickness=TrTrue, max_=ax, camber=axirfoil.plot(
alots()plt.subpfig, ax = plotting


# Basic as pltt yplootlib.pport matpl
imhon```pytotting


### Pl``

`edge format Leading put/")  #tory="outdirecl.save_le(e)
airfoider=Tru", heautput/irectory="oe_selig(dsave
airfoil.ave to fil
# S")
foil.dat/airh/tole("patl.from_fiJaxAirfoi=
airfoil  filerom f# Load```python
le I/O

## Fi
)
```

#ints=150 n_po4412"],0012", "2412", ""ca4(
    [batch_naoil.= JaxAirf n_valid s,it_splmasks, upperbatch_ds, _coorbatchration
ene g# Batch NACAe[2]
)

oords.shaph_cy_x, batcalid, quern_v, pper_splitscoords, uch_
    bat(icknessoil.batch_th= JaxAirfkness atch_thic1, 20)
be(0, inspac.l= jnp
query_x  operations
# Batchoils)
st(airflifrom__batch_il.createrfod = JaxAits, n_valispliper_asks, up batch_ms,oord
batch_c
]
00)n_points=1, aca("4412"oil.nrfxAi0),
    Jan_points=10("0012", ca.na  JaxAirfoil00),
  s=1n_point12", .naca("24foil  JaxAir   = [
oilsirf
aairfoilsm list of tch froCreate ba``python
#

`cessingtch Pro## Bas

# FeaturevancedAd
```

## 00
)ints=2.5, n_po0012, eta=0, naca412
    naca2foils(om_two_morph_new_froil.ed = JaxAirffoils
morpho airetween tw# Morph b")

sineution="co, distribs=300intepanel(n_porfoil.rled = aiil
repane airfo
# Repanel=0.5
)
gercentakness_pege_thicin    flap_hs
# degreele=15.0,  ng    flap_atage=0.7,
chord_percenhinge_  flap_l.flap(
  foi= air
flapped ionransformatp tpply fla
# A``pythontions

`nsformara``

### T
`ace_pointsurffoil.lower_slower = air
x_lower, y_ce_points_surfafoil.upperr = air y_uppeper,tes()
x_upet_coordinal.grds = airfoids, y_coo
x_cooratate dcoordinath

# Get .chord_lengh = airfoilchord_lengt
_camberoil.maxairf =
max_camber_location_thickness airfoil.maxcation =kness_lomax_thicss
kneil.max_thic= airfoness s
max_thickropertiefoil pairGet
# 0))
e(0, 1, 5.linspacjnpr(oil.y_lowey = airfer_
low 1, 50))e(0,.linspac.y_upper(jnpairfoilr_y = 0.25)
uppecamber_line(il. airfo_chord =_at_quarterer.5)
cambthickness(0l. airfoi =chords_at_midthicknesties
etric propergeomQuery n
# ythons

```pperatioric Oasic Geomet``

### B")
`teSurfacesame="Separaer, now, lupperupper_lower(rom_oil.fAirfirfoil = Jax0]])
a 0.0.05, [0.08, -0],.5, 1.0, 0[0.rray([er = jnp.a.08]])
low5, 0, [0.0, 0.0, 0.0] 0.5[1.0,.array([ jnpr =rfaces
upper su upper/lowereate from# C
oil")
tomAirfname="Cuses, ordinat(coaxAirfoilfoil = J
custom_air 0.0]])05,, -0.05, 0.08.0, 0.       [0
       1.0],, 0.5,  0.5, 0.0[[1.0,p.array(s = jnoordinatedinates
cte from coorea0)

# Croints=2523012", n_pil.naca5("rfo= JaxAia23012 ac150)
noints=12", n_pnaca4("00foil.= JaxAir012 aca00)
n n_points=2012",naca("24l. = JaxAirfoiaca2412
n airfoilste NACA

# Creail JaxAirfoorton impatilement_imp.jaxUS.airfoilsm ICARas jnp
fro jax.numpy thon
importils

```pyg Airfo Creatin

###Usage
## Basic )
```
foilOps
mizedJaxAir   OptilOps,
 hAirfoi  Batc
  tter,ilPlo
    Airfort (tion impoenta.jax_implemUS.airfoils
from ICARl utilitiest additiona: Impor
# OptionalAirfoil
axmport Jon intatix_implemeairfoils.jaARUS.on
from ICntatiimplemefoil  JAX airrt theImpon
#

```pythoon and Setuptallati# Instion

#ecompilaled rith controlanagement went memory mes**: Efficiatic Shapons
- **Sterati opbatch for .vmap`le with `jaxib: Compatrization**ctoad`
- **Vegrd_value_an`jax.d`, ax.gra `j: Works withutation**dient Comp- **Graompilation
it` c.jt `jaxs supporl method Alation**:IT Compil- **Jormations
X transf with JAibilityull compattion**: Fstraree RegiPytn

- **ioratX Integ## JAfoils

#le airtipn mulns oed operatioizient vector: Efficcessing**roch PBatity
- **rentiabild diffe preserven withutio redistribng**: Pointneli
- **Repaormationsansfle flap tr-compatibradientons**: Gap Operati- **Fltries
o geomebetween twing rfoil morphaientiable Differphing**: ons
- **Moratih JAX opers witACA airfoilgit Ngit and 5-dion**: 4-di Generati**NACAport
- ient supith grades wce querir, surfambe, caessThicknions**: eratometric Op

- **Geiesabilitap C
### Coreures
# Key Featde

#sting coment for exiin replaceity**: Drop-ilibatmp **API Co
-airfoils multiple ations fortorized operssing**: Vecch Proceion
- **Batecompilats rpreventagement buffer man: Efficient **cationory AlloMem*Static
- *owsorkflmization wr optiort foadient supp: Full griation**ic Different*Automat
- *ncerformar maximum pe-compiled foITions are Jperaton**: All oT Compilati
- **100% JIBenefits
y ## Keflows.

#orkional wtatompuements for covmprnce int performaficaigni se providinginal whilhe origy with tlitcompatibilete API ns compintaimentation maplehis imiation. T differenttic automailation and JIT comphat enablesfoil class tICARUS Airf the or oble refactticompay JAX-s a fullementation iAirfoil impl

The JAX oduction)

## Intringeshootublrooting](#t [Troublesho)
12.acticest-prtices](#besst Prac11. [Bees)
mplxaes](#e[Exampl10. ference)
-reence](#apiAPI Referuide)
9. [on-gigratin Guide](#mMigratiorison)
8. [paformance-com#permparison](ormance Cos)
7. [Perffeaturefic-pecis](#jax-stureific Fea-Spec6. [JAX
ures)d-feat](#advanceaturesanced FeAdv [sage)
5.-u#basicsic Usage]( [Batup)
4.-and-sestallationSetup](#intion and nstalla [Itures)
3.s](#key-feaature [Key Fe
2.roduction)ion](#introducts

1. [Inttentle of Con# Tabide

#ive Guens - Comprehtionmenta Imple JAX Airfoil#
