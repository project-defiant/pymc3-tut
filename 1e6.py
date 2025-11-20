import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1E6

    For each example in 1E5 use SciPy to specify the distribution in Python. Pick parameters that you believe are reasonable, take a random sample of size 1000 and plot the resulting distribution.
    """)
    return


@app.cell
def _():
    from scipy import stats
    from matplotlib import pyplot as plt
    import numpy as np
    return np, plt, stats


@app.cell
def _(np, plt, stats):
    # The number of people visiting your local cafe assuming Poisson distribution
    # Assume ~ 700 people visiting (this would be a Possion distribution)
    mu = 700
    sample_size = 1000000
    dist = stats.poisson(mu=mu).rvs(sample_size)
    # Plot the distribution
    plt.hist(dist, bins=100, density=True)
    plt.title('Poisson Distribution of Cafe Visitors')
    plt.xlabel('Number of Visitors')
    plt.ylabel('Density')
    plt.text(570, 0.01, f'Poisson Distribution\nμ={mu}', fontsize=12)
    plt.text(570, 0.009, f'Sample Size = {sample_size}', fontsize=12)
    plt.text(570, 0.008, f'Mean of Sample = {np.mean(dist):.2f}', fontsize=12)
    plt.text(570, 0.007, f'Std Dev of Sample = {np.std(dist):.2f}', fontsize=12)
    plt.text(570, 0.006, f'Variance of Sample = {np.var(dist):.2f}', fontsize=12)
    plt.show()

    # Note: Indeed Var == mean for Poisson distribution
    return


@app.cell
def _(np, plt, stats):
    # The weight of dogs in kg assuming uniform regression

    low, high = 6, 12 # range for dogs weight
    sample_size2 = 10000000
    loc = low
    scale = high - low
    dist_unif = stats.uniform(loc=loc, scale=scale).rvs(sample_size2)
    # Plot the distribution
    plt.hist(dist_unif, bins=100, density=True)
    plt.title('Uniform Distribution of Dog Weights')
    plt.xlabel('Weight (kg)')
    plt.ylabel('Density')
    plt.text(7, 0.13, f'Uniform Distribution\nLow={low}, High={high}', fontsize=12)
    plt.text(7, 0.12, f'Sample Size = {sample_size2}', fontsize=12)
    plt.text(7, 0.11, f'Mean of Sample = {np.mean(dist_unif):.2f}', fontsize=12)
    plt.text(7, 0.10, f'Std Dev of Sample = {np.std(dist_unif):.2f}', fontsize=12)
    plt.text(7, 0.09, f'Variance of Sample = {np.var(dist_unif):.2f}', fontsize=12)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
