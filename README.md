# THIS ENTIRE LIBRARY WAS VIBE CODED WITH CLAUDE.AI

### Watch Here
https://youtu.be/sGt9LNyktoo

# Comprehensive Quantitative Finance Library

## 📋 Overview

A **PhD-level quantitative finance library** written in pure Java featuring 127+ advanced financial methods, including derivatives pricing models, risk management tools, portfolio optimization, machine learning integration, and cutting-edge numerical methods. Built from the ground up with no external dependencies.

---

## 🎯 Development Journey

This library was built iteratively through a series of user-defined prompts to create an increasingly sophisticated quantitative finance toolkit:

### **Phase 1: Foundation**
**Prompt:** *"Create a comprehensive quantitative finance library in Finance.java"*

Implemented foundational methods covering:
- Black-Scholes option pricing and Greeks (Delta, Gamma, Vega, Theta, Rho)
- Binomial tree models
- Value at Risk (VaR) and Conditional VaR
- Present Value calculations and NPV/IRR
- Bond pricing and duration analysis
- Portfolio statistics and performance ratios
- CAPM and correlation analysis

### **Phase 2: Advanced Models & ML Integration**
**Prompt:** *"Add more complex models and ML integration"*

Extended library with:
- Advanced option models (Merton Jump-Diffusion, Barrier, Asian, Lookback, Heston)
- Exotic Greeks (Vanna, Volga, Charm, Vera)
- Advanced portfolio models (Black-Litterman, Risk Parity)
- Risk analytics (GARCH, Historical VaR, Cornish-Fisher VaR, Stressed VaR)
- Advanced fixed income (CIR Model, Key Rate Duration, Bond Convexity, OAS)
- Credit risk models (Merton PD, Expected Loss, Credit VaR)
- Machine Learning classes:
  - Neural Network for option pricing
  - Linear Regression for price prediction
  - K-Means clustering for volatility regimes
  - Random Forest-like ensemble methods
  - Principal Component Analysis (PCA)
  - Regime switching models
  - Gaussian copula for dependency modeling
  - SABR model for volatility smile
  - Local volatility surfaces

### **Phase 3: PhD-Level Models & Differential Equations**
**Prompt:** *"Add even more advanced quantitative finance models with differential equations and manual optimizers"*

Added sophisticated academic-level models:
- **Black-Scholes PDE Solver** - Finite difference method for option pricing
- **Hamilton-Jacobi-Bellman (HJB) Equations** - Optimal portfolio control
- **Fokker-Planck Solver** - Probability density evolution
- **Ito's Lemma** - Stochastic calculus derivatives
- **Longstaff-Schwartz Algorithm** - American option pricing
- **Gradient Descent Optimizer** - CVaR minimization
- **Newton-Raphson Optimizer** - Implied volatility and root finding
- **Convex Portfolio Optimization** - Markowitz with constraints
- **Martingale Pricing** - Risk-neutral valuation
- **Mean-Field Games** - Multi-agent equilibrium
- **Fredholm Integral Equations** - Credit default intensity
- **Free Boundary Problems** - Optimal American exercise
- **Levenberg-Marquardt Optimizer** - Non-linear least squares calibration
- **Dynamic Programming** - Optimal stopping times

### **Phase 4: Advanced Mathematical Functions**
**Prompt:** *"Add more quantitative finance mathematics functions not yet covered"*

Completed the library with:
- **Numerical Methods:**
  - Cholesky decomposition
  - Matrix multiplication
  - Runge-Kutta 4th-order SDE solver
  - Simpson's rule integration
  - Gauss-Legendre quadrature
  - Bisection method for root finding
  
- **Exotic Options:**
  - Rainbow options
  - Basket options
  - Quanto options
  - Bermuda options
  - Swing options
  
- **Advanced Greeks:**
  - Speed (3rd-order gamma)
  - Color (gamma decay)
  - Zomma (gamma-vega cross)
  - Vomma (vega convexity)
  - Ultima (vega-gamma cross)
  
- **Term Structure Models:**
  - Nelson-Siegel yield curves
  - Svensson curves
  - Cubic spline interpolation
  - Forward Rate Agreements (FRA)
  - Interest rate swaps
  
- **Credit Models:**
  - Spread-adjusted bond pricing
  - Credit Default Swaps (CDS)
  - Structural credit models
  
- **Machine Learning:**
  - Kalman Filter for volatility tracking
  - Characteristic function methods (Carr-Madan FFT)
  - Hidden Markov Models with Viterbi algorithm

---

## 🚀 Quick Start

### **Compilation**
```bash
javac Finance.java Home.java
```

### **Execution**
```bash
java Home
```

---

## 📚 Complete Method Reference

### **1. BLACK-SCHOLES MODEL**

#### `blackScholesCall(S, K, T, r, sigma)`
Calculates the Black-Scholes call option price using the closed-form solution.

**Parameters:**
- `S` - Current stock price
- `K` - Strike price
- `T` - Time to expiration (in years, e.g., 0.25 for 3 months)
- `r` - Risk-free interest rate (e.g., 0.05 for 5%)
- `sigma` - Annualized volatility (e.g., 0.20 for 20%)

**Returns:** Call option price

**Example:**
```java
double callPrice = Finance.blackScholesCall(100, 105, 0.5, 0.05, 0.20);
// Stock at 100, strike 105, 6 months, 5% rate, 20% vol
```

#### `blackScholesPut(S, K, T, r, sigma)`
Calculates the Black-Scholes put option price.

**Example:**
```java
double putPrice = Finance.blackScholesPut(100, 95, 0.5, 0.05, 0.20);
```

---

### **2. OPTION GREEKS**

#### `callDelta(S, K, T, r, sigma)`
Measures the rate of change of option price with respect to stock price. Range: 0 to 1.

**Example:**
```java
double delta = Finance.callDelta(100, 100, 0.5, 0.05, 0.20); // ~0.64
// For every $1 move in stock, option moves ~$0.64
```

#### `callGamma(S, K, T, r, sigma)`
Measures the rate of change of delta. Highest at-the-money.

**Example:**
```java
double gamma = Finance.callGamma(100, 100, 0.5, 0.05, 0.20);
// Gamma is largest when ATM, indicating delta is most sensitive
```

#### `callVega(S, K, T, r, sigma)`
Measures sensitivity to volatility changes. Per 1% change in volatility.

**Example:**
```java
double vega = Finance.callVega(100, 100, 0.5, 0.05, 0.20);
// For each 1% increase in vol, option price increases by ~$vega
```

#### `callTheta(S, K, T, r, sigma)`
Measures time decay. Negative for long options (lose value as time passes).

**Example:**
```java
double theta = Finance.callTheta(100, 100, 0.5, 0.05, 0.20);
// Negative value: option loses this much per day due to time decay
```

#### `callRho(S, K, T, r, sigma)`
Measures interest rate sensitivity. Per 1% change in rates.

**Example:**
```java
double rho = Finance.callRho(100, 100, 0.5, 0.05, 0.20);
```

#### `putDelta(S, K, T, r, sigma)`
Put delta ranges from -1 to 0.

**Example:**
```java
double putDelta = Finance.putDelta(100, 100, 0.5, 0.05, 0.20); // ~-0.36
```

---

### **3. BINOMIAL MODEL**

#### `binomialCall(S, K, T, r, sigma, steps)`
Tree-based option pricing. Converges to Black-Scholes as steps increase.

**Parameters:**
- `steps` - Number of tree steps (10-100 typical)

**Example:**
```java
double binomialPrice10 = Finance.binomialCall(100, 105, 0.5, 0.05, 0.20, 10);
double binomialPrice50 = Finance.binomialCall(100, 105, 0.5, 0.05, 0.20, 50);
double binomialPrice100 = Finance.binomialCall(100, 105, 0.5, 0.05, 0.20, 100);
// Prices converge as steps increase
```

---

### **4. VALUE AT RISK (VAR) & CONDITIONAL VAR**

#### `valueAtRisk(mean, stdDev, confidenceLevel, investmentAmount)`
Parametric VaR using variance-covariance method.

**Parameters:**
- `confidenceLevel` - e.g., 0.95 for 95% confidence
- `investmentAmount` - Portfolio size

**Example:**
```java
double var95 = Finance.valueAtRisk(0.10, 0.15, 0.95, 1000000);
// At 95% confidence, potential loss is $var95
```

#### `conditionalVaR(mean, stdDev, confidenceLevel, investmentAmount)`
Expected Shortfall (CVaR) - average loss beyond VaR.

**Example:**
```java
double cvar95 = Finance.conditionalVaR(0.10, 0.15, 0.95, 1000000);
// Expected loss given we're in the worst 5%
```

---

### **5. TIME VALUE OF MONEY**

#### `presentValue(futureValue, rate, periods)`
Discount future cash flows.

**Example:**
```java
double pv = Finance.presentValue(1100, 0.10, 1); // $1000
// $1100 received in 1 year at 10% discount rate
```

#### `futureValue(presentValue, rate, periods)`
Compound present cash flows.

**Example:**
```java
double fv = Finance.futureValue(1000, 0.10, 1); // $1100
```

#### `netPresentValue(cashFlows, discountRate)`
Sum of discounted cash flows.

**Example:**
```java
double[] cf = {-1000, 300, 300, 300, 300};
double npv = Finance.netPresentValue(cf, 0.10);
```

#### `internalRateOfReturn(cashFlows, initialGuess)`
Finds discount rate where NPV = 0.

**Example:**
```java
double irr = Finance.internalRateOfReturn(cf, 0.10);
```

---

### **6. BOND PRICING**

#### `bondPrice(couponPayment, faceValue, yieldToMaturity, periods)`
Calculate bond price given YTM.

**Example:**
```java
double price = Finance.bondPrice(50, 1000, 0.05, 10);
// Annual coupon $50, face value $1000, 5% YTM, 10 periods
```

#### `macaulayDuration(couponPayment, faceValue, yieldToMaturity, periods)`
Weighted average time to cash flows.

**Example:**
```java
double duration = Finance.macaulayDuration(50, 1000, 0.05, 10);
```

#### `modifiedDuration(couponPayment, faceValue, yieldToMaturity, periods)`
Interest rate sensitivity. Bond price change per 1% yield change.

**Example:**
```java
double modDur = Finance.modifiedDuration(50, 1000, 0.05, 10);
// If yield rises 1%, bond price falls ~modDur%
```

#### `bondConvexity(couponPayment, faceValue, yieldToMaturity, periods)`
Second-order price sensitivity (convexity effect).

**Example:**
```java
double convexity = Finance.bondConvexity(50, 1000, 0.05, 10);
```

---

### **7. PORTFOLIO ANALYSIS**

#### `portfolioReturn(returns, weights)`
Weighted average portfolio return.

**Example:**
```java
double[] returns = {0.10, 0.15, 0.08};
double[] weights = {0.4, 0.35, 0.25};
double portReturn = Finance.portfolioReturn(returns, weights); // ~0.111
```

#### `portfolioStdDev(volatilities, weights, correlationMatrix)`
Portfolio volatility considering correlations.

**Example:**
```java
double[] vols = {0.15, 0.20, 0.10};
double[][] corr = {{1.0, 0.3, 0.2}, {0.3, 1.0, 0.4}, {0.2, 0.4, 1.0}};
double portVol = Finance.portfolioStdDev(vols, weights, corr);
```

#### `sharpeRatio(portfolioReturn, riskFreeRate, portfolioStdDev)`
Return per unit of risk. Higher is better.

**Example:**
```java
double sharpe = Finance.sharpeRatio(0.15, 0.03, 0.12);
// (15% - 3%) / 12% = 1.0
```

#### `sortinoRatio(portfolioReturn, riskFreeRate, returns, minimumAcceptableReturn)`
Sharpe ratio using only downside risk.

**Example:**
```java
double sortino = Finance.sortinoRatio(0.15, 0.03, histReturns, 0.0);
```

#### `treynorRatio(portfolioReturn, riskFreeRate, beta)`
Return per unit of systematic risk.

**Example:**
```java
double treynor = Finance.treynorRatio(0.15, 0.03, 1.2);
```

---

### **8. CAPM & CORRELATIONS**

#### `capm(riskFreeRate, beta, marketReturn)`
Expected return of an asset based on systematic risk.

**Example:**
```java
double expectedReturn = Finance.capm(0.03, 1.0, 0.10);
// 3% + 1.0 * (10% - 3%) = 10%
```

#### `correlation(returns1, returns2)`
Correlation coefficient between two assets.

**Example:**
```java
double corr = Finance.correlation(asset1Returns, asset2Returns);
// Returns correlation from -1 to 1
```

#### `covariance(returns1, returns2)`
Covariance between two return series.

**Example:**
```java
double cov = Finance.covariance(asset1Returns, asset2Returns);
```

---

### **9. STATISTICS**

#### `mean(data)`
Average of data array.

#### `standardDeviation(data)`
Sample standard deviation.

#### `variance(data)`
Sample variance.

#### `skewness(data)`
Third moment - asymmetry in distribution.

#### `kurtosis(data)`
Fourth moment (excess) - tail thickness.

**Example:**
```java
double[] returns = {0.01, -0.02, 0.03, 0.01, -0.01};
double mean = Finance.mean(returns);
double std = Finance.standardDeviation(returns);
double skew = Finance.skewness(returns);
double kurt = Finance.kurtosis(returns);
```

---

### **10. MONTE CARLO SIMULATION**

#### `monteCarloCallOption(S, K, T, r, sigma, simulations)`
Stochastic option pricing via simulation.

**Example:**
```java
double mcPrice1000 = Finance.monteCarloCallOption(100, 105, 0.5, 0.05, 0.20, 1000);
double mcPrice10000 = Finance.monteCarloCallOption(100, 105, 0.5, 0.05, 0.20, 10000);
// More simulations = more accurate
```

#### `monteCarloPortfolio(initialWeights, returns, volatilities, simulations, periods)`
Portfolio value simulation over multiple periods.

**Example:**
```java
double[] weights = {0.4, 0.35, 0.25};
double[] expReturns = {0.10, 0.15, 0.08};
double[] vols = {0.15, 0.20, 0.10};
double[] portfolioPath = Finance.monteCarloPortfolio(weights, expReturns, vols, 100, 5);
```

---

### **11. INTEREST RATE MODELS**

#### `vasicekModel(r0, a, b, sigma, dt, steps)`
Mean-reverting interest rate process.

**Parameters:**
- `r0` - Initial rate
- `a` - Mean reversion speed
- `b` - Long-term mean rate
- `sigma` - Volatility
- `dt` - Time step
- `steps` - Number of simulation steps

**Example:**
```java
double[] rates = Finance.vasicekModel(0.05, 0.15, 0.05, 0.01, 0.01, 252);
// Simulate 1 year of daily rates
```

#### `cirModel(r0, kappa, theta, sigma, dt, steps)`
Cox-Ingersoll-Ross model - ensures non-negative rates.

**Example:**
```java
double[] rates = Finance.cirModel(0.05, 0.15, 0.05, 0.01, 0.01, 252);
```

#### `hullWhiteRates(r0, a, b, sigma, dt, steps)`
One-factor Hull-White model.

**Example:**
```java
double[] rates = Finance.hullWhiteRates(0.05, 0.1, 0.05, 0.01, 0.01, 252);
```

#### `forwardRate(spot1, spot2, t1, t2)`
Calculate forward rate between two dates.

**Example:**
```java
double forward = Finance.forwardRate(0.03, 0.04, 1, 2);
// Forward rate from year 1 to year 2
```

---

### **12. ADVANCED OPTION MODELS**

#### `mertonJumpDiffusionCall(S, K, T, r, sigma, jumpMean, jumpStdDev, jumpIntensity)`
Option pricing with random jump events.

**Example:**
```java
double price = Finance.mertonJumpDiffusionCall(
    100, 105, 0.5, 0.05, 0.20,
    -0.05, 0.1, 2.0);  // 2 jumps per year on average
```

#### `barierDownAndOutCall(S, K, B, T, r, sigma)`
Option that expires if price touches barrier B.

**Example:**
```java
double price = Finance.barierDownAndOutCall(100, 105, 80, 0.5, 0.05, 0.20);
// Barrier at 80, below current price
```

#### `asianArithmeticAverageCall(prices, K, T, r, sigma)`
Payoff based on average price, not final price.

**Example:**
```java
double[] prices = {98, 100, 102, 101, 99};
double price = Finance.asianArithmeticAverageCall(prices, 100, 0.5, 0.05, 0.20);
```

#### `lookbackCallMaxPrice(S, K, maxPrice, T, r, sigma)`
Payoff based on maximum price over period.

**Example:**
```java
double price = Finance.lookbackCallMaxPrice(100, 100, 110, 0.5, 0.05, 0.20);
// Max price so far: 110
```

#### `hestonCall(S, K, T, r, v0, kappa, theta, sigma, rho)`
Stochastic volatility model.

**Example:**
```java
double price = Finance.hestonCall(100, 105, 0.5, 0.05, 0.04, 
    2.0, 0.05, 0.3, 0.3);
```

---

### **13. EXOTIC GREEKS**

#### `vanna(S, K, T, r, sigma)`
Delta-vega cross sensitivity.

#### `volga(S, K, T, r, sigma)`
Vega convexity.

#### `callCharm(S, K, T, r, sigma)`
Delta decay over time (theta of delta).

#### `vera(S, K, T, r, sigma)`
Rho-vega cross sensitivity.

**Example:**
```java
double vanna = Finance.vanna(100, 100, 0.5, 0.05, 0.20);
double volga = Finance.volga(100, 100, 0.5, 0.05, 0.20);
double charm = Finance.callCharm(100, 100, 0.5, 0.05, 0.20);
```

---

### **14. ADVANCED PORTFOLIO MODELS**

#### `minVariancePortfolio(returns, volatilities, correlations, targetReturn)`
Find minimum variance portfolio for target return.

**Example:**
```java
double minVar = Finance.minVariancePortfolio(returns, vols, corr, 0.12);
```

#### `blackLittermanReturns(marketReturns, views, viewConfidence, tau)`
Blend market-implied returns with investor views.

**Example:**
```java
double[] blReturns = Finance.blackLittermanReturns(
    new double[]{0.10, 0.12, 0.08},
    new double[]{0.15, 0.10, 0.09},
    new double[]{0.8, 0.5, 0.3},
    0.05);
```

#### `riskParityWeights(volatilities)`
Allocate to equalize risk contribution.

**Example:**
```java
double[] rpWeights = Finance.riskParityWeights(new double[]{0.15, 0.20, 0.10});
// Weight inversely to volatility
```

#### `maxDrawdown(returns)`
Worst cumulative loss from peak.

**Example:**
```java
double maxDD = Finance.maxDrawdown(dailyReturns);
// Returns -0.20 for 20% maximum drawdown
```

#### `calmarRatio(returns, riskFreeRate)`
Return per unit of drawdown.

**Example:**
```java
double calmar = Finance.calmarRatio(returns, 0.02);
```

#### `informationRatio(portfolioReturns, benchmarkReturns)`
Excess return per unit of tracking error.

**Example:**
```java
double infoRatio = Finance.informationRatio(portReturns, benchReturns);
```

---

### **15. ADVANCED RISK MODELS**

#### `garch11Forecast(returns, omega, alpha, beta, forecastPeriods)`
GARCH(1,1) volatility forecasting with clustering.

**Example:**
```java
double garchVol = Finance.garch11Forecast(returns, 0.00001, 0.05, 0.94, 5);
```

#### `historicalVaR(returns, confidenceLevel)`
Empirical quantile method.

**Example:**
```java
double histVar = Finance.historicalVaR(returns, 0.95);
```

#### `cornishFisherVaR(returns, confidenceLevel, investmentAmount)`
Accounts for skewness and kurtosis.

**Example:**
```java
double cfVar = Finance.cornishFisherVaR(returns, 0.95, 1000000);
```

#### `stressedVaR(baselineReturns, stressScalar, confidenceLevel, investmentAmount)`
VaR under adverse conditions.

**Example:**
```java
double stressVar = Finance.stressedVaR(returns, 1.5, 0.95, 1000000);
```

---

### **16. FIXED INCOME ADVANCED**

#### `keyRateDuration(coupon, faceValue, ytm, maturity, keyMaturity, yieldShift)`
Sensitivity to specific maturity points.

**Example:**
```java
double krd = Finance.keyRateDuration(50, 1000, 0.05, 10, 5, 0.01);
```

#### `optionAdjustedSpread(bondPrice, coupon, faceValue, periods, riskFreeYield, estimatedOAS)`
Option-adjusted spread pricing.

**Example:**
```java
double oas = Finance.optionAdjustedSpread(950, 50, 1000, 10, 0.03, 0.02);
```

---

### **17. CREDIT RISK**

#### `mertonProbabilityOfDefault(assetValue, debtValue, volatility, T, r)`
Probability of default from equity value.

**Example:**
```java
double pd = Finance.mertonProbabilityOfDefault(500, 300, 0.3, 1, 0.05);
```

#### `expectedLoss(debtValue, recoveryRate, probabilityOfDefault)`
Expected credit loss.

**Example:**
```java
double expLoss = Finance.expectedLoss(300, 0.4, pd);
```

#### `creditVaR(debtValue, probabilityOfDefault, recoveryRate, confidenceLevel)`
Unexpected credit loss at confidence level.

**Example:**
```java
double cVaR = Finance.creditVaR(300, pd, 0.4, 0.95);
```

#### `creditValuationAdjustment(exposures, probabilitiesOfDefault, recoveryRates)`
CVA for counterparty risk.

**Example:**
```java
double cva = Finance.creditValuationAdjustment(
    new double[]{100, 200, 150},
    new double[]{0.02, 0.03, 0.01},
    new double[]{0.4, 0.4, 0.4});
```

---

### **18. SENSITIVITY ANALYSIS**

#### `scenarioAnalysis(baselineReturns, riskFreeRate, scenarioMultipliers)`
PV under different scenarios.

**Example:**
```java
double[] scenarios = Finance.scenarioAnalysis(
    returns, 0.02,
    new double[]{0.8, 1.0, 1.2});  // Bear, Base, Bull
```

#### `sensitivityTable(baseValue, var1Min, var1Max, var2Min, var2Max, steps)`
Two-way sensitivity table.

**Example:**
```java
double[][] table = Finance.sensitivityTable(100, 90, 110, 0.15, 0.25, 10);
```

#### `greeksLadder(K, T, r, sigma, spotMin, spotMax, spots)`
Greeks across multiple spot prices.

**Example:**
```java
double[][] ladder = Finance.greeksLadder(100, 0.5, 0.05, 0.20, 80, 120, 20);
```

---

### **19. MACHINE LEARNING MODELS**

#### `NeuralNetworkPricer.predictCallPrice(S, K, T, r, sigma)`
Neural network-based option pricing.

**Example:**
```java
Finance.NeuralNetworkPricer nn = new Finance.NeuralNetworkPricer(10);
double price = nn.predictCallPrice(100, 105, 0.5, 0.05, 0.20);
```

#### `NeuralNetworkPricer.train(trainingInputs, trainingOutputs, epochs)`
Train on market data.

**Example:**
```java
nn.train(trainingData, marketPrices, 1000);
```

#### `LinearRegression.fit(features, targets)`
Fit linear regression model.

**Example:**
```java
Finance.LinearRegression lr = new Finance.LinearRegression();
lr.fit(featureMatrix, targetPrices);
```

#### `LinearRegression.predict(features)`
Predict using trained model.

**Example:**
```java
double prediction = lr.predict(newFeatures);
```

#### `KMeansVolatilityRegime.fit(volatilities, maxIterations)`
Cluster volatility regimes.

**Example:**
```java
Finance.KMeansVolatilityRegime kmeans = new Finance.KMeansVolatilityRegime(3);
kmeans.fit(volatilityTimeSeries, 100);
```

#### `KMeansVolatilityRegime.predictRegime(volatility)`
Identify current regime.

**Example:**
```java
int regime = kmeans.predictRegime(0.25);  // 0, 1, or 2
```

#### `PCA.fit(returns)`
Principal component analysis on returns.

**Example:**
```java
Finance.PCA pca = new Finance.PCA();
pca.fit(returnMatrix);
double explainedVar = pca.getExplainedVarianceRatio();
```

#### `RegimeSwitchingModel.fit(returns, regimes)`
Learn regime switching dynamics.

**Example:**
```java
Finance.RegimeSwitchingModel rsm = new Finance.RegimeSwitchingModel(0.08, 0.12, 0.10, 0.18, 0.95, 0.95);
rsm.fit(returns, regimeLabels);
```

#### `GaussianCopula.fitCorrelation(returns1, returns2)`
Fit copula to asset pairs.

**Example:**
```java
Finance.GaussianCopula copula = new Finance.GaussianCopula();
copula.fitCorrelation(asset1Returns, asset2Returns);
```

#### `HiddenMarkovModel.viterbi(observations, emissionProbs)`
Find most likely regime sequence.

**Example:**
```java
Finance.HiddenMarkovModel hmm = new Finance.HiddenMarkovModel(transitions, initial);
int[] regimeSequence = hmm.viterbi(observations, emissionProbs);
```

---

### **20. EXOTIC OPTIONS**

#### `rainbowCallOnMax(S1, S2, K, T, r, sigma1, sigma2, rho)`
Option on maximum of two assets.

**Example:**
```java
double price = Finance.rainbowCallOnMax(100, 120, 110, 0.5, 0.05, 0.20, 0.25, 0.3);
```

#### `basketCall(prices, K, T, r, volatilities, correlations)`
Option on basket (average) of assets.

**Example:**
```java
double[] prices = {100, 120, 80};
double[] vols = {0.20, 0.25, 0.15};
double[][] corr = {{1, 0.3, 0.2}, {0.3, 1, 0.4}, {0.2, 0.4, 1}};
double price = Finance.basketCall(prices, 100, 0.5, 0.05, vols, corr);
```

#### `quantoCall(S, K, exchangeRate, T, r_domestic, r_foreign, sigma_S, sigma_FX, rho_SFX)`
Cross-currency option with FX risk.

**Example:**
```java
double price = Finance.quantoCall(100, 105, 1.2, 0.5, 0.05, 0.03, 0.20, 0.15, 0.3);
```

#### `bermudaCall(S, K, exerciseDates, r, sigma)`
Option exercisable on specific dates.

**Example:**
```java
double[] dates = {0.25, 0.5, 0.75, 1.0};
double price = Finance.bermudaCall(100, 105, dates, 0.05, 0.20);
```

#### `swingOption(spotPrice, strikePrice, numExercises, T, r, sigma)`
Multiple exercise rights.

**Example:**
```java
double price = Finance.swingOption(100, 95, 5, 1.0, 0.05, 0.20);
// Can exercise up to 5 times
```

---

### **21. ADVANCED GREEKS (HIGHER ORDER)**

#### `callSpeed(S, K, T, r, sigma)`
Third-order gamma (gamma sensitivity).

#### `callColor(S, K, T, r, sigma)`
Gamma decay with time (theta of gamma).

#### `callZomma(S, K, T, r, sigma)`
Gamma-vega cross sensitivity.

#### `callVomma(S, K, T, r, sigma)`
Vega convexity.

#### `callUltima(S, K, T, r, sigma)`
Vega-gamma cross sensitivity.

**Example:**
```java
double speed = Finance.callSpeed(100, 100, 0.5, 0.05, 0.20);
double color = Finance.callColor(100, 100, 0.5, 0.05, 0.20);
double zomma = Finance.callZomma(100, 100, 0.5, 0.05, 0.20);
```

---

### **22. TERM STRUCTURE MODELS**

#### `nelsonSiegelYield(tau, beta0, beta1, beta2, lambda)`
Parameterized yield curve.

**Example:**
```java
double yield1Y = Finance.nelsonSiegelYield(1.0, 0.02, 0.015, -0.005, 1.0);
```

#### `svensonYield(tau, beta0, beta1, beta2, beta3, lambda1, lambda2)`
Extended Nelson-Siegel yield curve.

**Example:**
```java
double yield = Finance.svensonYield(1.0, 0.02, 0.015, -0.005, 0.003, 1.0, 0.5);
```

#### `cubicSplineInterpolation(knots, values, queryPoints)`
Smooth curve fitting.

**Example:**
```java
double[] knots = {0.5, 1.0, 2.0, 5.0, 10.0};
double[] yields = {0.02, 0.025, 0.03, 0.032, 0.035};
double[] query = {0.75, 1.5, 3.0};
double[] interpolated = Finance.cubicSplineInterpolation(knots, yields, query);
```

#### `fraPrice(notional, strikeRate, startDate, endDate, spotCurve, timesGrid)`
Forward Rate Agreement pricing.

**Example:**
```java
double fra = Finance.fraPrice(1000000, 0.025, 1.0, 1.25, spotCurve, times);
```

#### `swapValue(notional, fixedRate, paymentDates, floatingRates, discountFactors)`
Interest rate swap valuation.

**Example:**
```java
double[] dates = {0.25, 0.5, 0.75, 1.0};
double[] floatRates = {0.02, 0.022, 0.025, 0.027};
double[] discounts = {0.99, 0.98, 0.97, 0.96};
double swapVal = Finance.swapValue(1000000, 0.025, dates, floatRates, discounts);
```

---

### **23. CREDIT MODELS**

#### `creditSpreadBondPrice(coupon, faceValue, yieldToMaturity, creditSpread, periods)`
Bond pricing with credit spread.

**Example:**
```java
double price = Finance.creditSpreadBondPrice(50, 1000, 0.05, 0.02, 10);
```

#### `cdsValue(notional, cdsSpread, paymentTimes, discountFactors, probabilityOfDefault)`
Credit Default Swap valuation.

**Example:**
```java
double cds = Finance.cdsValue(1000000, 0.01, dates, discounts, 0.02);
```

#### `structuralCreditSpread(firmValue, debtFaceValue, T, volatility, r)`
Merton structural model for credit spread.

**Example:**
```java
double spread = Finance.structuralCreditSpread(500, 300, 1.0, 0.3, 0.05);
```

---

### **24. NUMERICAL METHODS**

#### `choleskyDecomposition(matrix)`
Matrix decomposition for sampling.

**Example:**
```java
double[][] covMatrix = {{1, 0.3}, {0.3, 1}};
double[][] L = Finance.choleskyDecomposition(covMatrix);
// Lower triangular for correlated random sampling
```

#### `matrixMultiply(A, B)`
Matrix multiplication.

**Example:**
```java
double[][] result = Finance.matrixMultiply(A, B);
```

#### `rungeKutta4(x0, T, steps, drift, diffusion)`
Advanced SDE solver.

**Example:**
```java
double[] path = Finance.rungeKutta4(100, 1.0, 252,
    x -> 0.05 * x,  // drift
    x -> 0.20 * x); // diffusion
```

#### `simpsonsRule(f, a, b, n)`
Simpson's numerical integration.

**Example:**
```java
double integral = Finance.simpsonsRule(
    x -> Math.exp(-x*x/2) / Math.sqrt(2*Math.PI),
    -3, 3, 100);
```

#### `gaussLegendreQuadrature(f, a, b)`
High-precision Gauss-Legendre integration.

**Example:**
```java
double integral = Finance.gaussLegendreQuadrature(
    x -> x * x,
    0, 1);
```

#### `bisectionMethod(f, a, b, tolerance)`
Root finding algorithm.

**Example:**
```java
double root = Finance.bisectionMethod(
    x -> x*x - 2,  // Find sqrt(2)
    1, 2, 1e-6);
```

---

### **25. KALMAN FILTER**

#### `KalmanFilter.update(measurement)`
Update filter with new measurement.

**Example:**
```java
Finance.KalmanFilter kf = new Finance.KalmanFilter(100, 0, 0.01, 0.1, 0.01);
double[] measurements = {100.5, 100.2, 100.8, 101.0};
for (double m : measurements) {
    double filtered = kf.update(m);
}
```

#### `KalmanFilter.getFilteredLevel()`
Get current smoothed level.

**Example:**
```java
double level = kf.getFilteredLevel();
```

---

### **26. HIDDEN MARKOV MODEL**

#### `HiddenMarkovModel.viterbi(observations, emissionProbs)`
Viterbi algorithm for most likely regime sequence.

**Example:**
```java
double[][] transMatrix = {{0.9, 0.1}, {0.1, 0.9}};
double[] initial = {0.5, 0.5};
Finance.HiddenMarkovModel hmm = new Finance.HiddenMarkovModel(transMatrix, initial);
int[] regimes = hmm.viterbi(observations, emissionProbs);
```

---

### **27. PDE SOLVERS & STOCHASTIC CALCULUS**

#### `BlackScholesPDESolver.solveCall(K)`
Solve Black-Scholes PDE via finite differences.

**Example:**
```java
Finance.BlackScholesPDESolver pde = new Finance.BlackScholesPDESolver(
    100, 100, 50, 150, 1.0, 0.05, 0.20);
double[][] grid = pde.solveCall(100);
double callPrice = pde.getCallPrice(100, 100);
```

#### `HJBOptimalControl.solveOptimalAllocation()`
Hamilton-Jacobi-Bellman for portfolio optimization.

**Example:**
```java
Finance.HJBOptimalControl hjb = new Finance.HJBOptimalControl(
    1.0, 0.05, 0.10, 0.20, 252, 100);
double[] allocation = hjb.solveOptimalAllocation();
```

#### `FokkerPlanckSolver.solveProbabilityDensity(initialDensity, dt, timeSteps)`
Solve probability density evolution.

**Example:**
```java
Finance.FokkerPlanckSolver fp = new Finance.FokkerPlanckSolver(
    0.05, 0.20, 100, 50, 150);
double[] density = fp.solveProbabilityDensity(initialDensity, 0.01, 252);
```

#### `ItosLemma.applyItosLemma(S, K, T, r, sigma, derivativeOrder)`
Apply Ito's lemma for stochastic calculus.

**Example:**
```java
double[] drift_vol = Finance.ItosLemma.applyItosLemma(100, 100, 0.5, 0.05, 0.20, null);
// Returns [drift, volatility] of option value process
```

---

### **28. OPTIMIZATION ALGORITHMS**

#### `LongstaffSchwartz.priceAmericanCall(S, K, T, r, sigma)`
Longstaff-Schwartz for American option pricing.

**Example:**
```java
Finance.LongstaffSchwartz ls = new Finance.LongstaffSchwartz(10000, 50);
double americanPrice = ls.priceAmericanCall(100, 100, 1.0, 0.05, 0.20);
```

#### `GradientDescentOptimizer.minimizeCVaR(returns, targetReturn, confidenceLevel)`
Minimize CVaR via gradient descent.

**Example:**
```java
Finance.GradientDescentOptimizer gd = new Finance.GradientDescentOptimizer(0.01, 1e-6, 1000);
double[] optWeights = gd.minimizeCVaR(returnMatrix, 0.12, 95);
```

#### `NewtonRaphsonOptimizer.solveImpliedVolatility(S, K, T, r, marketPrice, initialGuess)`
Find implied volatility from market price.

**Example:**
```java
Finance.NewtonRaphsonOptimizer nr = new Finance.NewtonRaphsonOptimizer(1e-6, 100);
double impliedVol = nr.solveImpliedVolatility(100, 105, 0.5, 0.05, 6.5, 0.20);
```

#### `ConvexPortfolioOptimizer.optimizePortfolio(returns, covMatrix, riskAversion)`
Markowitz portfolio optimization.

**Example:**
```java
Finance.ConvexPortfolioOptimizer opt = new Finance.ConvexPortfolioOptimizer();
double[] weights = opt.optimizePortfolio(returns, covMatrix, 2.0);
```

---

### **29. ADVANCED PRICING**

#### `sabrVolatility(forward, strike, T, alpha, beta, nu, rho)`
SABR model volatility smile.

**Example:**
```java
double vol = Finance.sabrVolatility(100, 105, 0.5, 0.2, 0.5, 0.3, 0.3);
```

#### `dupireLocalVolatility(S, K, T, r, q, impliedVol, dK, dT)`
Dupire local volatility from implied surface.

**Example:**
```java
double localVol = Finance.dupireLocalVolatility(100, 105, 0.5, 0.05, 0.02, 0.20, 1, 0.01);
```

#### `carrMadanFFTCall(S, K, T, r, sigma, lambda, n)`
Fast Fourier Transform option pricing.

**Example:**
```java
double price = Finance.carrMadanFFTCall(100, 105, 0.5, 0.05, 0.20, 1.5, 4096);
```

#### `ensembleOptionPrice(S, K, T, r, sigma)`
Ensemble of pricing models.

**Example:**
```java
double ensemblePrice = Finance.ensembleOptionPrice(100, 105, 0.5, 0.05, 0.20);
// Blends BS, Binomial, and Monte Carlo
```

#### `mahalanobisDistance(point, mean, covariance)`
Multivariate anomaly detection.

**Example:**
```java
double distance = Finance.mahalanobisDistance(
    new double[]{0.03, 0.15},
    means, covMatrix);
```

---

## 📊 Test Suite

The [Home.java](Home.java) file contains 250+ comprehensive test cases organized in 30+ test methods covering all functionality:

- **Basic pricing:** 50+ tests for Black-Scholes, Greeks, Binomial models
- **Risk metrics:** 40+ tests for VaR, CVaR, portfolio analysis
- **Advanced models:** 60+ tests for jump-diffusion, exotic options, stochastic models
- **ML integration:** 40+ tests for neural networks, regression, clustering, HMM
- **Numerical methods:** 30+ tests for PDE solvers, optimizers, integration methods

Run all tests with:
```bash
java Home
```

---

## 🔧 Requirements

- **Java 8 or higher** (uses lambdas for function passing in numerical methods)
- **No external dependencies** - pure vanilla Java

---

## 📈 Key Features

✅ **127+ financial methods** covering derivatives, risk, and portfolio analysis
✅ **PhD-level models** including PDE solvers and differential equations
✅ **Machine learning** integration (neural networks, regression, clustering, HMM)
✅ **250+ test cases** for validation
✅ **Production-ready** numerical algorithms
✅ **No external libraries** - complete self-contained library
✅ **Comprehensive documentation** for every method
✅ **Educational value** - learn quantitative finance implementation

---

## 💡 Use Cases

- **Derivatives Pricing:** Black-Scholes, Binomial, Monte Carlo, Exotic options
- **Risk Management:** VaR, CVaR, Greeks, scenario analysis
- **Portfolio Optimization:** Efficient frontier, Markowitz, Black-Litterman, risk parity
- **Credit Risk:** Merton model, CDS pricing, credit spread modeling
- **Interest Rates:** Vasicek, CIR, Hull-White, yield curve modeling
- **Machine Learning:** Price prediction, regime detection, anomaly detection
- **Research:** Exotic models, numerical methods, stochastic calculus

---

## 📝 Notes

- All methods are **static** - call directly: `Finance.blackScholesCall(...)`
- Time periods are in **years** - use decimals (0.25 = 3 months)
- Rates and volatilities are **decimals** - 0.05 means 5%
- Returns arrays are **required for statistical methods** - pass historical data
- Machine learning classes use **nested classes** - instantiate with `new Finance.ClassName()`

---

## 🎓 Academic References

This library implements models from:
- Black-Scholes-Merton option pricing
- Hull-White interest rate models
- Longstaff-Schwartz American option pricing
- SABR volatility smile modeling
- Merton structural credit models
- Kalman filtering for time series
- Hidden Markov models
- PCA and clustering techniques

---

## 📄 License

This comprehensive quantitative finance library is provided for educational use.

---



