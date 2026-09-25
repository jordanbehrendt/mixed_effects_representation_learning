from .linear_mixed_model import fit_lmm


def lmm_adapter(data, **kwargs):
    """
    Adapter so MERL can call LMM consistently.
    """

    return fit_lmm(
        data=data,
        fixed_formula=kwargs["fixed_formula"],
        group_var=kwargs["group_var"],
        random_formula=kwargs.get("random_formula", "1"),
        random_effects_covariance=kwargs.get("random_effects_covariance", "diagonal"),
        residual_covariance=kwargs.get("residual_covariance", "independence"),
        time_var=kwargs.get("time_var", "time"),
        estimation_method=kwargs.get("estimation_method", "REML"),
        script_path=kwargs.get("r_script", None),
    )