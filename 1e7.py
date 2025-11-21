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
    # 1E7

    Compare priors $Beta(0.5, 0.5)$, $Beta(1,1)$, $Beta(1,4)$. How do the priors differ in terms of shape?
    """)
    return


@app.cell
def _():
    from matplotlib import pyplot as plt
    import numpy as np
    from scipy import stats
    return np, plt, stats


@app.cell
def _(np, plt, stats):
    alpha = np.array([0.5, 1, 1,0.1])
    beta = np.array([0.5, 1, 4, 1])

    _, ax = plt.subplots(1, figsize=(4,4))
    i = 0
    x = np.linspace(0,1, 200)
    for a,b in  zip(alpha, beta):
        dist = stats.beta(a,b)
        ax.plot(x, dist.pdf(x), label=f"Beta({a},{b})", lw=3)
    ax.legend(loc=9)
    
    
    return


@app.cell
def _(plt, stats):
    _, ax2 = plt.subplots(1, figsize=(4,4))
    rvs = stats.beta(0.1, 1).rvs(100000)
    ax2.hist(rvs)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
