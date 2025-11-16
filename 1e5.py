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
    # 1E5.

    Sketch what the distribution of possible observed values could be for the following cases:

    (a) The number of people visiting your local cafe assuming Poisson distribution

    (b) The weight of adult dogs in kilograms assuming a Uniform distribution

    (c) The weight of adult elephants in kilograms assuming Normal distribution

    (d) The weight of adult humans in pounds assuming skew Normal distribution

    We’ll generate “sketches” using scipy for ease of use presentation in this solution manual. Have your students make hand sketches
    """)
    return


@app.cell
def _():
    import numpy as np
    from scipy.stats import poisson, uniform, norm
    from matplotlib import pyplot as plt
    return norm, np, plt, poisson, uniform


@app.cell
def _(np, plt, poisson):
    # 1. (a) Poisson Distribution for number of people visiting a local cafe

    visitors = np.arange(0, 100) # numbers <0, 99>
    pmf = poisson(20).pmf(visitors) # poisson distribution with parameter 
    plt.bar(visitors, pmf)
    plt.xlabel('Number of Visitors')
    plt.ylabel('Probability mass function')
    plt.title('Poisson Distribution for Cafe Visitors')
    return


@app.cell
def _(np, plt, poisson):
    # example observations
    counts = np.array([2, 1, 19, 25, 20, 17, 24])  # replace with your data
    n = counts.size
    K = counts.sum()
    print("n =", n, "sum K =", K, "sample mean (MLE) =", K / n)

    # lambda grid to evaluate
    lambdas = np.linspace(1, 40, 1000)

    # compute log-pmf for each observation and each lambda
    # shape: (n, len(lambdas))
    logpmf_matrix = poisson.logpmf(counts[:, None], lambdas[None, :])

    # joint log-likelihood for each lambda (sum over observations)
    loglik = logpmf_matrix.sum(axis=0)

    # likelihood (may underflow if computed directly) — compute from loglik
    likelihood = np.exp(loglik - loglik.max())   # normalize by max for numerical stability

    # MLE
    lambda_mle = K / n

    # Plot normalized likelihood
    plt.figure(figsize=(8,4))
    plt.plot(lambdas, likelihood, label='Normalized joint likelihood')
    plt.axvline(lambda_mle, color='k', linestyle='--', label=f'MLE = {lambda_mle:.3f}')
    plt.xlabel('λ (rate)')
    plt.ylabel('Relative likelihood (max = 1)')
    plt.title('Joint Likelihood (normalized) for Poisson | observed counts')
    plt.legend()
    plt.show()

    # Plot log-likelihood
    plt.figure(figsize=(8,4))
    plt.plot(lambdas, loglik, label='Log-likelihood')
    plt.axvline(lambda_mle, color='k', linestyle='--', label=f'MLE = {lambda_mle:.3f}')
    plt.xlabel('λ (rate)')
    plt.ylabel('Log-likelihood')
    plt.title('Log-Likelihood for Poisson | observed counts')
    plt.legend()
    plt.show()
    return


@app.cell
def _(np, plt, uniform):
    # 2. (b) Uniform Distribution for weight of adult dogs in kilograms

    dog_weights = np.linspace(0, 10000)
    pdf_dogs = uniform(5, 45).pdf(dog_weights) # uniform distribution between 5 and 50 kg
    plt.plot(dog_weights, pdf_dogs)
    plt.xlabel('Weight (kg)')
    plt.ylabel('Probability density function')
    plt.title('Uniform Distribution for Adult Dog Weights')
    plt.show()
    return


@app.cell
def _(norm, np, plt):
    # 3. (c) Normal Distribution for weight of adult elephants in kilograms
    elephant_weights = np.linspace(0, 10000)
    pdf_elephants = norm(5000, 1000).pdf(elephant_weights) # normal distribution with mean 5000 kg and stddev 1000 kg
    plt.plot(elephant_weights, pdf_elephants)
    return


@app.cell
def _(np, plt):
    # 4. (d) Skew Normal Distribution for weight of adult humans in pounds
    from scipy.stats import skewnorm
    human_weights = np.linspace(50, 400)
    pdf_humans = skewnorm(a=10, loc=150, scale=30).pdf(human_weights) # skew normal distribution
    plt.plot(human_weights, pdf_humans)
    plt.xlabel('Weight (lbs)')
    plt.ylabel('Probability density function')
    plt.title('Skew Normal Distribution for Adult Human Weights')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
