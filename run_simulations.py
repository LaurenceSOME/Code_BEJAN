# ============================================================
# run_simulations.py  (version corrigée)
# ============================================================
# Reproduction du modele de Vargas, Ordonez et Bejan (2000)
#
# Corrections apportees par rapport a la version precedente :
#   1. bejan_optimizer.py : eta_II utilise maintenant mu (rapport de
#      capacites thermiques) au lieu de M (rapport de debits massiques).
#      C'est la seule expression dimensionnellement correcte (voir le
#      commentaire dans bejan_optimizer.py). Avec l'ancienne formule,
#      eta_II,max augmentait artificiellement avec cp du gaz chaud.
#   2. Les conditions du "cas de reference" sont maintenant EXACTEMENT
#      celles de la Fig. 3a / Fig. 4 / Fig. 6 de l'article
#      (TH=1193 K, Tb=584 K, T1=537 K, T0=298 K, N=3), et non plus des
#      valeurs arbitraires (900 K / eau a pression atmospherique).
#   3. Les proprietes de l'eau/vapeur (hfg, cw, cs) sont maintenant
#      calculees via CoolProp a 100 bar (la pression utilisee par
#      l'article pour Tb=584 K), au lieu de valeurs approximatives a
#      pression atmospherique (cw=4180, hfg=2257000...) qui ne sont
#      valables qu'a 1 atm et 100 degC.
#
# Remarque importante :
#   L'article NE DONNE PAS la valeur de cp du gaz chaud utilisee pour
#   construire la Fig. 4. Avec la formule corrigee, eta_II,max ne
#   depend PLUS de cp (c'est une vraie efficacite de 2e loi,
#   independante de ce choix) -- seul M_opt en depend, via
#   M_opt = mu_opt * cp/cs. On a trouve par tatonnement qu'un
#   cp = 1437 J/(kg.K) (valeur tout a fait realiste pour un gaz
#   d'echappement chaud) redonne exactement M_opt = 0.234, comme dans
#   la Fig. 4/5 de l'article. Change CP_GAS ci-dessous si tu as une
#   meilleure estimation du cp reellement utilise par les auteurs.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import CoolProp.CoolProp as CP

from bejan_optimizer import Properties, optimize_M


# ============================================================
# 1. CONDITIONS DE FONCTIONNEMENT (= Fig. 3a / 4 / 6 de l'article)
# ============================================================

T0 = 298.0        # K, temperature ambiante (correspond a tau=1)
TH = 1193.0       # K, tauH = 4.00
Tb = 584.0        # K, taub = 1.96  (saturation a 100 bar)
T1 = 537.0        # K, tau1 = 1.80  (condensat entrant a 100 bar)

tauH = TH / T0
taub = Tb / T0
tau1 = T1 / T0

print(f"tauH = {tauH:.3f}  (article: 4.00)")
print(f"taub = {taub:.3f}  (article: 1.96)")
print(f"tau1 = {tau1:.3f}  (article: 1.80)")


# ============================================================
# 2. PROPRIETES THERMOPHYSIQUES (eau/vapeur a 100 bar via CoolProp)
# ============================================================

P_BOILING = 100e5   # Pa, 100 bar (pression utilisee dans l'article)

hf = CP.PropsSI('H', 'P', P_BOILING, 'Q', 0, 'Water')
hg = CP.PropsSI('H', 'P', P_BOILING, 'Q', 1, 'Water')
hfg = hg - hf                                            # J/kg

cw = CP.PropsSI('C', 'P', P_BOILING, 'Q', 0, 'Water')     # liquide sature a Tb

# cs moyen = moyenne entre cp de la vapeur saturee et cp a T2 (estimation
# initiale ~ 0.85*TH ; affinee ci-dessous une fois M_opt connu)
cs_sat = CP.PropsSI('C', 'P', P_BOILING, 'Q', 1, 'Water')
cs_guess = CP.PropsSI('C', 'P', P_BOILING, 'T', 0.85 * TH, 'Water')
cs = 0.5 * (cs_sat + cs_guess)

# cp du gaz chaud : NON DONNE par l'article. Valeur choisie pour
# retrouver M_opt = 0.234 (voir remarque en tete de fichier).
CP_GAS = 1437.0    # J/(kg.K)

props = Properties(cp=CP_GAS, cw=cw, cs=cs, hfg=hfg)

print("\n----- PROPRIETES (CoolProp, 100 bar) -----")
print(f"hfg = {hfg:.1f} J/kg")
print(f"cw  = {cw:.1f} J/(kg.K)")
print(f"cs  = {cs:.1f} J/(kg.K)  (moyenne vapeur sat. / T2 estime)")
print(f"cp (gaz chaud, choisi) = {CP_GAS:.1f} J/(kg.K)")


# ============================================================
# 3. PARAMETRES DE L'ECHANGEUR
# ============================================================

N = 3.0            # article: Fig. 3a / Fig. 4 -> N = 3
Ub_Us = 1.0
Uw_Us = 1.0
M_bounds = (0.01, 0.45)


# ============================================================
# 4. SIMULATION DU CAS DE REFERENCE
# ============================================================

print("\n====================================================")
print("       SIMULATION DU CAS DE REFERENCE (Fig. 3a / 4)")
print("====================================================")

M_opt, state, Ms, etas = optimize_M(
    N=N, tauH=tauH, taub=taub, tau1=tau1,
    Ub_Us=Ub_Us, Uw_Us=Uw_Us, props=props, T0=T0, M_bounds=M_bounds
)

print(f"\nM_opt      = {M_opt:.4f}   (article : 0.234)")
print(f"eta_II,max = {state['eta_II']:.4f}   (article, modele gaz ideal : ~0.65-0.70)")

print("\n----- TEMPERATURES ADIMENSIONNELLES -----")
print(f"tau2 = {state['tau2']:.4f}   tau3 = {state['tau3']:.4f}   "
      f"tau4 = {state['tau4']:.4f}   tauout = {state['tauout']:.4f}")

print("\n----- REPARTITION DE LA SURFACE -----")
print(f"As/A = {state['x']:.4f}   Ab/A = {state['y']:.4f}   Aw/A = {state['aw']:.4f}")
print(f"somme = {state['x']+state['y']+state['aw']:.6f}")

print("\n----- EFFECTIVENESS -----")
print(f"eps_s = {state['eps_s']:.4f}   eps_b = {state['eps_b']:.4f}   eps_w = {state['eps_w']:.4f}")


# ============================================================
# 5. FIGURE : eta_II en fonction de M  (reproduction Fig. 4)
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(Ms, etas, linewidth=2, label=r"$\eta_{II}(M)$ ")
plt.scatter(M_opt, state["eta_II"], s=80, color="C1",
            label=f"M_opt = {M_opt:.3f}, eta_II,max = {state['eta_II']:.3f}")
plt.axvline(M_opt, linestyle="--", color="gray")
plt.axvline(0.234, linestyle=":", color="red")
plt.xlabel(r"$M = \dot{m}_w / \dot{m}$")
plt.ylabel(r"Second-law efficiency $\eta_{II}$")
plt.title("Optimisation du debit d'eau froide ")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("eta_vs_M.png", dpi=300)
plt.show()


# ============================================================
# 6. ETUDE PARAMETRIQUE EN N (reproduction Fig. 6)
# ============================================================

print("\n====================================================")
print("       ETUDE PARAMETRIQUE EN FONCTION DE N (Fig. 6)")
print("====================================================")

N_values = np.linspace(1.0, 10.0, 25)
rows = []

for N_i in N_values:
    try:
        M_opt_i, state_i, _, _ = optimize_M(
            N=N_i, tauH=tauH, taub=taub, tau1=tau1,
            Ub_Us=Ub_Us, Uw_Us=Uw_Us, props=props, T0=T0, M_bounds=M_bounds
        )
        rows.append(dict(N=N_i,M_opt=M_opt_i, eta_II_max=state_i["eta_II"],
                         As_A=state_i["x"],Ab_A=state_i["y"],Aw_A=state_i["aw"],
                         # Températures adimensionnelles
                         tau2=state_i["tau2"],
                         tau3=state_i["tau3"],
                         tau4=state_i["tau4"],
                         tauout=state_i["tauout"],
                         # Températures réelles [K]
                         T2=state_i["tau2"] * T0,
                         T3=state_i["tau3"] * T0,
                         T4=state_i["tau4"] * T0,
                         Tout=state_i["tauout"] * T0))



        
        #rows.append(dict(N=N_i, M_opt=M_opt_i, eta_II_max=state_i["eta_II"],
                          #As_A=state_i["x"], Ab_A=state_i["y"], Aw_A=state_i["aw"]))
        print(f"N = {N_i:5.2f} | M_opt = {M_opt_i:.4f} | eta_II,max = {state_i['eta_II']:.4f}")
    
    except Exception as e:
        print(f"N = {N_i:5.2f} | echec de la simulation ({e})")
        rows.append(dict(N=N_i, M_opt=np.nan, etat_II_max=np.nan,As_A=np.nan,
                         Ab_A=np.nan,Aw_A=np.nan,
                         
                         tau2=np.nan,
                         tau3=np.nan,
                         tau4=np.nan,
                         tauout=np.nan,
                         
                         T2=np.nan, 
                         T3=np.nan, 
                         T4=np.nan, 
                         Tout=np.nan
                         
                         ))
        
        
        #rows.append(dict(N=N_i, M_opt=np.nan, eta_II_max=np.nan,
                          #As_A=np.nan, Ab_A=np.nan, Aw_A=np.nan))
                          

results = pd.DataFrame(rows)
results.to_csv("results_vs_N.csv", index=False)
print("\nResultats sauvegardes dans results_vs_N.csv")

# Figure : eta_II,max et M_opt vs N (reproduction Fig. 6)
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(results["N"], results["eta_II_max"], marker="o", color="C0", label=r"$\eta_{II,max}$")
ax1.set_xlabel("N")
ax1.set_ylabel(r"$\eta_{II,max}$", color="C0")
ax1.set_ylim(0, 1)
ax2 = ax1.twinx()
ax2.plot(results["N"], results["M_opt"], marker="s", color="C1", linestyle="--", label=r"$M_{opt}$")
ax2.set_ylabel(r"$M_{opt}$", color="C1")
ax2.set_ylim(0, 0.5)
plt.title("Effet de la taille de l'echangeur N (cf. Fig. 6 de l'article)")
fig.tight_layout()
plt.savefig("eta_max_and_Mopt_vs_N.png", dpi=300)
plt.show()

print("\n====================================================")
print("           FIN DES SIMULATIONS")
print("====================================================")
