import numpy as np, pandas as pd, asteca, inspect
B="/Users/notluquis/erotica/data/test/NGC6383"
df=pd.read_csv(f"{B}/ASteCA/output/NGC_6383_dr3_all/NGC_6383_dr3_all.csv",sep=r"\s+")
print("members:",len(df),"cols:",list(df.columns))

clu=asteca.Cluster(
    ra=df["ra"].to_numpy(), dec=df["dec"].to_numpy(),
    mag=df["Gmag"].to_numpy(), e_mag=df["e_Gmag"].to_numpy(),
    color=df["BP_RP"].to_numpy(), e_color=df["e_BP_RP"].to_numpy(),
)

iso=asteca.Isochrones(
    model="MIST",
    isochs_path=f"{B}/MIST/UBVRIplus/",
    mag="Gaia_G_EDR3",
    color=("Gaia_BP_EDR3","Gaia_RP_EDR3"),
    magnitude_effl=6390.0,          # Gaia G eff. wavelength (Angstrom)
    color_effl=(5320.0,7970.0),     # Gaia BP, RP
    z_to_FeH=0.0152,   # interpret 'met' as Z (paper uses Z)
)

synth=asteca.Synthetic(
    iso,
    def_params={"met":0.024,"loga":6.55,"alpha":0.09,"beta":0.94,
                "Rv":3.1,"DR":0.0,"Av":1.24,"dm":10.3},
    IMF_name="chabrier_2014", gamma="D&K", seed=42,
)
synth.calibrate(clu)
# fitted solution (data/40 trace 1724708835, mode values)
model={"met":0.025,"loga":6.553,"alpha":0.09,"beta":0.94,"Rv":3.1,"DR":0.0,"Av":1.136,"dm":10.282}
model_std={"met":0.01,"loga":0.212,"alpha":0.0,"beta":0.0,"Rv":0.0,"DR":0.0,"Av":0.337,"dm":0.109}
synth.get_models(model, model_std, N_models=200)
res=synth.stellar_parameters()
print("\nstellar_parameters keys:", list(res.keys()) if isinstance(res,dict) else type(res))
if isinstance(res,dict):
    import numpy as np
    for k,v in res.items():
        try: print(f"  {k}: shape={np.shape(v)} mean={np.nanmean(v):.3f} max={np.nanmax(v):.3f}")
        except: print(f"  {k}: {type(v)}")
    out=pd.DataFrame({k:v for k,v in res.items() if np.ndim(v)==1})
    out.to_csv("/tmp/asteca069_masses.csv",index=False)
    print("\nsaved /tmp/asteca069_masses.csv shape",out.shape)
