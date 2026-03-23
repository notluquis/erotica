def RDP_bayesian(density_annulus, radius_gen, return_trace=False,progressbar=False):
    with pm.Model() as king_model:
        sigma = pm.HalfNormal('sigma', sigma=5)
        b = pm.TruncatedNormal('b',mu=0.1,sigma=1,lower=0)
        k = pm.TruncatedNormal('k', mu=5, sigma=5, lower=b)
        R_c = pm.TruncatedNormal('R_c', mu=2, sigma=5, lower=0)
        R_t = pm.TruncatedNormal('R_t', mu=20, sigma=5, lower=0)
        r = pm.ConstantData('radius', radius_gen)
        density_points = pm.ConstantData('density', density_annulus)

        king = pm.Deterministic('king', pm.math.switch(r < R_t,
                                                       k * ((1 / pm.math.sqrt(1 + (r / R_c) ** 2)) - (1 / pm.math.sqrt(1 + (R_t / R_c) ** 2))) ** 2 + b,
                                                       b))
        obs_density = pm.Normal('obs_density', mu=king, sigma=sigma, observed=density_points)
    rhat = [2, 2]
    target_accept = 0.8
    tune = 4000
    while any(x > 1 for x in rhat):
        with king_model:
            king_trace = pm_jax.sample_numpyro_nuts(draws=100000, tune=tune, target_accept=target_accept, random_seed=np.random.randint(1, 100000),progressbar=progressbar)
            rhat = az.summary(king_trace, var_names=["sigma", "k", "R_c", "R_t", "b"])['r_hat'].iloc[:]
        if target_accept <= 0.9999999:
            target_accept += 0.05
        tune += 2000

    k_mean = king_trace.posterior['k'].median().item()
    b_mean = king_trace.posterior['b'].median().item()
    R_c_mean = king_trace.posterior['R_c'].median().item()
    R_t_mean = king_trace.posterior['R_t'].median().item()
    k_std = king_trace.posterior['k'].std().item()
    b_std = king_trace.posterior['b'].std().item()
    R_c_std = king_trace.posterior['R_c'].std().item()
    R_t_std = king_trace.posterior['R_t'].std().item()
    king_std = king_trace.posterior['sigma'].median().item()
    C = np.log(king_trace.posterior['R_t']/king_trace.posterior['R_c']).median().item()
    C2 = np.log(king_trace.posterior['R_t'].median().item()/king_trace.posterior['R_c'].median().item())
    results = {
        'k_mean': k_mean,
        'b_mean': b_mean,
        'R_c_mean': R_c_mean*u.arcmin,
        'R_t_mean': R_t_mean*u.arcmin,
        'k_std': k_std,
        'b_std': b_std,
        'R_c_std': R_c_std,
        'R_t_std': R_t_std,
        'king_std': king_std,
        'C' : C
    }
    if return_trace is True:
        results['king_trace'] = king_trace
    return results