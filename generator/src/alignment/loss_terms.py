def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared = x**2
    sigma_squared = sigma**2
    loss_val = (sigma_squared * x_squared) / (sigma_squared + x_squared + 1e-8)
    return loss_val
