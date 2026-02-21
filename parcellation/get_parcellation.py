
def get_parcellation(strategy_name):
    if strategy_name == "schaefer400N17":
        from parcellation.schaefer400N17 import SchaeferParcellation
        return SchaeferParcellation()
    if strategy_name == "schaefer400N7":
        from parcellation.schaefer400N7 import SchaeferParcellation
        return SchaeferParcellation()
    elif strategy_name == "hcpmmp":
        from parcellation.hcpmmp import HCPMMPParcellation
        return HCPMMPParcellation()
    else:
        raise ValueError(f"Unknown parcellation: {strategy_name}")