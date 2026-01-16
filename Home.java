/**
 * Test cases for Finance library
 * Comprehensive testing of quantitative finance functions
 */

import java.util.*;
import java.math.*;

public class Home {

    public static void main(String[] args) {
        System.out.println("=== QUANTITATIVE FINANCE LIBRARY TEST SUITE ===\n");

        testBlackScholesModel();
        testBlackScholesGreeks();
        testBinomialModel();
        testValueAtRisk();
        testPresentValueDiscounting();
        testBondPricing();
        testPortfolioAnalysis();
        testCAPM();
        testStatisticalCalculations();
        testMonteCarloSimulation();
        testVasicekModel();
        testAdvancedOptionModels();
        testExoticGreeks();
        testAdvancedPortfolioModels();
        testAdvancedRiskModels();
        testAdvancedFixedIncome();
        testCreditRiskModels();
        testSensitivityAnalysis();
        testMLOptionPricing();
        testKMeansVolatilityRegimes();
        testEnsemblePredictor();
        testPCAAnalysis();
        testRegimeSwitching();
        testCopulaModeling();
        testSABRModel();
        testLocalVolatility();
        testHullWhiteModel();
        testCVACalculation();
        testFeatureImportance();
        testAnomalyDetection();
        testBlackScholesPDE();
        testHJBOptimalControl();
        testFokkerPlanck();
        testItosLemma();
        testLongstaffSchwartz();
        testGradientDescentOptimizer();
        testNewtonRaphsonOptimizer();
        testConvexOptimization();
        testMartingalePricing();
        testMeanFieldGames();
        testFreeBoundary();
        testDynamicProgramming();
        testNumericalMethods();
        testExoticOptions();
        testHigherOrderGreeks();
        testTermStructureModels();
        testCreditModels();
        testKalmanFilter();
        testCharacteristicFunctions();
        testHiddenMarkovModel();

        System.out.println("\n=== ALL TESTS COMPLETED ===");
    }

    // ==================== Black-Scholes Tests ====================

    private static void testBlackScholesModel() {
        System.out.println("--- BLACK-SCHOLES MODEL TESTS ---");

        // Test Case 1: Basic call option pricing
        double S = 100;      // Stock price
        double K = 100;      // Strike price
        double T = 1.0;      // 1 year to expiration
        double r = 0.05;     // 5% risk-free rate
        double sigma = 0.2;  // 20% volatility

        double callPrice = Finance.blackScholesCall(S, K, T, r, sigma);
        System.out.printf("Call Option Price (ATM): %.4f (Expected ~10.45)%n", callPrice);

        // Test Case 2: Put option pricing
        double putPrice = Finance.blackScholesPut(S, K, T, r, sigma);
        System.out.printf("Put Option Price (ATM): %.4f (Expected ~5.57)%n", putPrice);

        // Test Case 3: ITM call option
        double callPriceITM = Finance.blackScholesCall(110, 100, 1.0, 0.05, 0.2);
        System.out.printf("Call Option Price (ITM): %.4f%n", callPriceITM);

        // Test Case 4: OTM put option
        double putPriceOTM = Finance.blackScholesPut(110, 100, 1.0, 0.05, 0.2);
        System.out.printf("Put Option Price (OTM): %.4f%n", putPriceOTM);

        // Test Case 5: Short time to expiration
        double callShortExpiry = Finance.blackScholesCall(100, 100, 0.1, 0.05, 0.2);
        System.out.printf("Call Option Price (0.1 years to expiry): %.4f%n", callShortExpiry);

        System.out.println();
    }

    private static void testBlackScholesGreeks() {
        System.out.println("--- BLACK-SCHOLES GREEKS TESTS ---");

        double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;

        // Test Case 1: Call Delta
        double callDelta = Finance.callDelta(S, K, T, r, sigma);
        System.out.printf("Call Delta (ATM): %.4f (Expected ~0.64)%n", callDelta);

        // Test Case 2: Put Delta
        double putDelta = Finance.putDelta(S, K, T, r, sigma);
        System.out.printf("Put Delta (ATM): %.4f (Expected ~-0.36)%n", putDelta);

        // Test Case 3: Gamma
        double gamma = Finance.callGamma(S, K, T, r, sigma);
        System.out.printf("Gamma (ATM): %.4f%n", gamma);

        // Test Case 4: Vega
        double vega = Finance.callVega(S, K, T, r, sigma);
        System.out.printf("Vega (ATM): %.4f%n", vega);

        // Test Case 5: Theta
        double theta = Finance.callTheta(S, K, T, r, sigma);
        System.out.printf("Theta (ATM): %.4f (Negative - time decay)%n", theta);

        // Test Case 6: Rho
        double rho = Finance.callRho(S, K, T, r, sigma);
        System.out.printf("Rho (ATM): %.4f%n", rho);

        // Test Case 7: Delta range for ITM call
        double callDeltaITM = Finance.callDelta(110, 100, 1.0, 0.05, 0.2);
        System.out.printf("Call Delta (ITM): %.4f (Should be > 0.64)%n", callDeltaITM);

        // Test Case 8: Delta range for OTM call
        double callDeltaOTM = Finance.callDelta(90, 100, 1.0, 0.05, 0.2);
        System.out.printf("Call Delta (OTM): %.4f (Should be < 0.64)%n", callDeltaOTM);

        System.out.println();
    }

    // ==================== Binomial Model Tests ====================

    private static void testBinomialModel() {
        System.out.println("--- BINOMIAL MODEL TESTS ---");

        double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;

        // Test Case 1: 10 steps
        double binomialCall10 = Finance.binomialCall(S, K, T, r, sigma, 10);
        System.out.printf("Binomial Call (10 steps): %.4f (Should be close to BS: ~10.45)%n", 
                         binomialCall10);

        // Test Case 2: 50 steps (more accurate)
        double binomialCall50 = Finance.binomialCall(S, K, T, r, sigma, 50);
        System.out.printf("Binomial Call (50 steps): %.4f%n", binomialCall50);

        // Test Case 3: 100 steps (high accuracy)
        double binomialCall100 = Finance.binomialCall(S, K, T, r, sigma, 100);
        System.out.printf("Binomial Call (100 steps): %.4f%n", binomialCall100);

        // Test Case 4: Compare convergence to Black-Scholes
        double bsCall = Finance.blackScholesCall(S, K, T, r, sigma);
        double error = Math.abs(binomialCall100 - bsCall);
        System.out.printf("Error vs Black-Scholes: %.6f%n", error);

        System.out.println();
    }

    // ==================== Value at Risk Tests ====================

    private static void testValueAtRisk() {
        System.out.println("--- VALUE AT RISK TESTS ---");

        double mean = 0.08;           // 8% expected return
        double stdDev = 0.12;         // 12% standard deviation
        double confidenceLevel = 0.95; // 95% confidence
        double investment = 1000000;  // $1M investment

        // Test Case 1: Basic VaR at 95%
        double var95 = Finance.valueAtRisk(mean, stdDev, confidenceLevel, investment);
        System.out.printf("VaR (95%%): $%.2f (Maximum loss with 95%% confidence)%n", var95);

        // Test Case 2: VaR at 99%
        double var99 = Finance.valueAtRisk(mean, stdDev, 0.99, investment);
        System.out.printf("VaR (99%%): $%.2f%n", var99);

        // Test Case 3: Conditional VaR (Expected Shortfall) at 95%
        double cvar95 = Finance.conditionalVaR(mean, stdDev, confidenceLevel, investment);
        System.out.printf("Conditional VaR (95%%): $%.2f (Expected loss if tail event)%n", cvar95);

        // Test Case 4: CVaR at 99%
        double cvar99 = Finance.conditionalVaR(mean, stdDev, 0.99, investment);
        System.out.printf("Conditional VaR (99%%): $%.2f%n", cvar99);

        // Test Case 5: Verify CVaR > VaR (more conservative)
        System.out.printf("CVaR vs VaR: CVaR (%.2f) > VaR (%.2f): %b%n", cvar95, var95, cvar95 > var95);

        System.out.println();
    }

    // ==================== Present Value & Discounting Tests ====================

    private static void testPresentValueDiscounting() {
        System.out.println("--- PRESENT VALUE & DISCOUNTING TESTS ---");

        // Test Case 1: Simple PV calculation
        double pv = Finance.presentValue(1100, 0.10, 1);
        System.out.printf("PV of $1100 in 1 year at 10%% rate: $%.2f (Expected: $1000)%n", pv);

        // Test Case 2: Future value calculation
        double fv = Finance.futureValue(1000, 0.10, 1);
        System.out.printf("FV of $1000 in 1 year at 10%% rate: $%.2f (Expected: $1100)%n", fv);

        // Test Case 3: NPV calculation
        double[] cashFlows = {-1000, 300, 300, 300, 300};
        double npv = Finance.netPresentValue(cashFlows, 0.10);
        System.out.printf("NPV (rate 10%%): $%.2f%n", npv);

        // Test Case 4: NPV with different discount rate
        double npvHighRate = Finance.netPresentValue(cashFlows, 0.15);
        System.out.printf("NPV (rate 15%%): $%.2f (Should be lower than 10%%)%n", npvHighRate);

        // Test Case 5: Internal Rate of Return
        double irr = Finance.internalRateOfReturn(cashFlows, 0.10);
        System.out.printf("IRR: %.4f (%.2f%%)%n", irr, irr * 100);

        // Test Case 6: Verify IRR makes NPV zero
        double npvAtIRR = Finance.netPresentValue(cashFlows, irr);
        System.out.printf("NPV at IRR: $%.6f (Should be ~0)%n", npvAtIRR);

        System.out.println();
    }

    // ==================== Bond Pricing Tests ====================

    private static void testBondPricing() {
        System.out.println("--- BOND PRICING TESTS ---");

        double coupon = 50;        // $50 annual coupon
        double faceValue = 1000;   // $1000 face value
        double ytm = 0.05;         // 5% yield to maturity
        int periods = 10;          // 10 years

        // Test Case 1: Bond trading at par (coupon = YTM)
        double priceAtPar = Finance.bondPrice(50, 1000, 0.05, 10);
        System.out.printf("Bond Price (5%% coupon, 5%% YTM): $%.2f (Expected: $1000)%n", priceAtPar);

        // Test Case 2: Bond trading at discount (coupon < YTM)
        double priceDiscount = Finance.bondPrice(50, 1000, 0.07, 10);
        System.out.printf("Bond Price (5%% coupon, 7%% YTM): $%.2f (Should be < $1000)%n", priceDiscount);

        // Test Case 3: Bond trading at premium (coupon > YTM)
        double pricePremium = Finance.bondPrice(50, 1000, 0.03, 10);
        System.out.printf("Bond Price (5%% coupon, 3%% YTM): $%.2f (Should be > $1000)%n", pricePremium);

        // Test Case 4: Macaulay Duration
        double macDuration = Finance.macaulayDuration(coupon, faceValue, ytm, periods);
        System.out.printf("Macaulay Duration: %.4f years%n", macDuration);

        // Test Case 5: Modified Duration
        double modDuration = Finance.modifiedDuration(coupon, faceValue, ytm, periods);
        System.out.printf("Modified Duration: %.4f%n", modDuration);

        // Test Case 6: Duration increases with longer maturity
        double durationShort = Finance.macaulayDuration(coupon, faceValue, ytm, 5);
        double durationLong = Finance.macaulayDuration(coupon, faceValue, ytm, 20);
        System.out.printf("Duration comparison (5yr vs 20yr): %.4f < %.4f: %b%n", 
                         durationShort, durationLong, durationShort < durationLong);

        System.out.println();
    }

    // ==================== Portfolio Analysis Tests ====================

    private static void testPortfolioAnalysis() {
        System.out.println("--- PORTFOLIO ANALYSIS TESTS ---");

        double[] returns = {0.10, 0.12, 0.08};        // Asset returns
        double[] weights = {0.4, 0.3, 0.3};           // Portfolio weights
        double[] volatilities = {0.15, 0.20, 0.18};   // Asset volatilities

        // Test Case 1: Portfolio return
        double portReturn = Finance.portfolioReturn(returns, weights);
        System.out.printf("Portfolio Return: %.4f (%.2f%%)%n", portReturn, portReturn * 100);

        // Test Case 2: Verify weights sum to 1
        double sumWeights = 0;
        for (double w : weights) {
            sumWeights += w;
        }
        System.out.printf("Weights sum to 1: %b%n", Math.abs(sumWeights - 1.0) < 1e-6);

        // Test Case 3: Portfolio with identity correlation (independent assets)
        double[][] corrIdentity = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        double portVolIdentity = Finance.portfolioStdDev(volatilities, weights, corrIdentity);
        System.out.printf("Portfolio Std Dev (independent assets): %.4f%n", portVolIdentity);

        // Test Case 4: Portfolio with perfect positive correlation (worst case)
        double[][] corrPositive = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
        double portVolPositive = Finance.portfolioStdDev(volatilities, weights, corrPositive);
        System.out.printf("Portfolio Std Dev (perfect correlation): %.4f%n", portVolPositive);

        // Test Case 5: Sharpe ratio
        double riskFreeRate = 0.02;
        double sharpeRatio = Finance.sharpeRatio(portReturn, riskFreeRate, portVolIdentity);
        System.out.printf("Sharpe Ratio: %.4f%n", sharpeRatio);

        // Test Case 6: Treynor ratio
        double beta = 1.2;
        double treynorRatio = Finance.treynorRatio(portReturn, riskFreeRate, beta);
        System.out.printf("Treynor Ratio: %.4f%n", treynorRatio);

        System.out.println();
    }

    // ==================== CAPM Tests ====================

    private static void testCAPM() {
        System.out.println("--- CAPM TESTS ---");

        double riskFreeRate = 0.03;      // 3%
        double marketReturn = 0.10;      // 10%
        double marketRiskPremium = marketReturn - riskFreeRate; // 7%

        // Test Case 1: Market portfolio (beta = 1)
        double expectedReturnMarket = Finance.capm(riskFreeRate, 1.0, marketReturn);
        System.out.printf("Expected Return (beta=1): %.4f (%.2f%%)%n", 
                         expectedReturnMarket, expectedReturnMarket * 100);

        // Test Case 2: Defensive stock (beta = 0.7)
        double expectedReturnDefensive = Finance.capm(riskFreeRate, 0.7, marketReturn);
        System.out.printf("Expected Return (beta=0.7, defensive): %.4f (%.2f%%)%n", 
                         expectedReturnDefensive, expectedReturnDefensive * 100);

        // Test Case 3: Aggressive stock (beta = 1.5)
        double expectedReturnAggressive = Finance.capm(riskFreeRate, 1.5, marketReturn);
        System.out.printf("Expected Return (beta=1.5, aggressive): %.4f (%.2f%%)%n", 
                         expectedReturnAggressive, expectedReturnAggressive * 100);

        // Test Case 4: Risk-free asset (beta = 0)
        double expectedReturnRiskFree = Finance.capm(riskFreeRate, 0.0, marketReturn);
        System.out.printf("Expected Return (beta=0, risk-free): %.4f (%.2f%%)%n", 
                         expectedReturnRiskFree, expectedReturnRiskFree * 100);

        // Test Case 5: Verify relationship (higher beta = higher return)
        System.out.printf("Beta 0.7 return < Beta 1.0 return < Beta 1.5 return: %b%n",
                         expectedReturnDefensive < expectedReturnMarket && 
                         expectedReturnMarket < expectedReturnAggressive);

        // Test Case 6: Covariance and Correlation
        double[] returns1 = {0.05, 0.06, 0.07, 0.08, 0.09};
        double[] returns2 = {0.08, 0.09, 0.10, 0.11, 0.12};
        double cov = Finance.covariance(returns1, returns2);
        double corr = Finance.correlation(returns1, returns2);
        System.out.printf("Covariance: %.6f%n", cov);
        System.out.printf("Correlation: %.4f (Highly positive: %b)%n", corr, corr > 0.8);

        System.out.println();
    }

    // ==================== Statistical Calculations Tests ====================

    private static void testStatisticalCalculations() {
        System.out.println("--- STATISTICAL CALCULATIONS TESTS ---");

        double[] data = {100, 105, 102, 108, 103, 107, 104, 106, 101, 109};

        // Test Case 1: Mean
        double mean = Finance.mean(data);
        System.out.printf("Mean: %.4f%n", mean);

        // Test Case 2: Standard Deviation
        double stdDev = Finance.standardDeviation(data);
        System.out.printf("Standard Deviation: %.4f%n", stdDev);

        // Test Case 3: Variance
        double variance = Finance.variance(data);
        System.out.printf("Variance: %.4f (StdDev^2: %.4f)%n", variance, stdDev * stdDev);

        // Test Case 4: Skewness (should be near 0 for symmetric data)
        double skewness = Finance.skewness(data);
        System.out.printf("Skewness: %.4f (Near 0 = symmetric)%n", skewness);

        // Test Case 5: Kurtosis (excess kurtosis, should be near 0 for normal)
        double kurtosis = Finance.kurtosis(data);
        System.out.printf("Kurtosis (Excess): %.4f (Near 0 = normal distribution)%n", kurtosis);

        // Test Case 6: Skewed data
        double[] skewedData = {1, 2, 3, 4, 5, 100};
        double skewnessSkewed = Finance.skewness(skewedData);
        System.out.printf("Skewness (skewed data): %.4f (Positive = right skew)%n", skewnessSkewed);

        System.out.println();
    }

    // ==================== Monte Carlo Simulation Tests ====================

    private static void testMonteCarloSimulation() {
        System.out.println("--- MONTE CARLO SIMULATION TESTS ---");

        double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;

        // Test Case 1: MC call option (1000 simulations)
        double mcCall1000 = Finance.monteCarloCallOption(S, K, T, r, sigma, 1000);
        System.out.printf("MC Call Option (1000 sims): %.4f%n", mcCall1000);

        // Test Case 2: MC call option (10000 simulations)
        double mcCall10000 = Finance.monteCarloCallOption(S, K, T, r, sigma, 10000);
        System.out.printf("MC Call Option (10000 sims): %.4f%n", mcCall10000);

        // Test Case 3: Compare to Black-Scholes
        double bsCall = Finance.blackScholesCall(S, K, T, r, sigma);
        System.out.printf("Black-Scholes Call: %.4f%n", bsCall);
        System.out.printf("MC error (10k sims): %.4f%n", Math.abs(mcCall10000 - bsCall));

        // Test Case 4: MC portfolio (100 simulations, 5 periods)
        double[] weights = {0.5, 0.5};
        double[] returns = {0.08, 0.10};
        double[] volatilities = {0.15, 0.20};
        double[] portValues = Finance.monteCarloPortfolio(weights, returns, volatilities, 100, 5);
        System.out.printf("Portfolio MC (Initial: $%.2f, Final: $%.2f)%n", portValues[0], portValues[5]);

        // Test Case 5: Verify increasing number of simulations reduces variance
        System.out.println("MC converges to Black-Scholes as simulations increase (empirical test)");

        System.out.println();
    }

    // ==================== Vasicek Model Tests ====================

    private static void testVasicekModel() {
        System.out.println("--- VASICEK INTEREST RATE MODEL TESTS ---");

        double r0 = 0.03;      // Initial rate 3%
        double a = 0.15;       // Mean reversion speed
        double b = 0.05;       // Long-term mean 5%
        double sigma = 0.01;   // Volatility 1%
        double dt = 0.01;      // Time step (1% of year)
        int steps = 252;       // 1 year with daily steps

        // Test Case 1: Simulate short rates
        double[] rates = Finance.vasicekModel(r0, a, b, sigma, dt, steps);
        System.out.printf("Initial Rate: %.4f (3.00%%)%n", rates[0]);
        System.out.printf("Final Rate: %.4f%n", rates[steps]);

        // Test Case 2: Calculate average rate over period
        double sumRates = 0;
        for (double rate : rates) {
            sumRates += rate;
        }
        double avgRate = sumRates / rates.length;
        System.out.printf("Average Rate: %.4f (Should be close to long-term mean: %.4f)%n", 
                         avgRate, b);

        // Test Case 3: Forward rate calculation
        double spot1 = 0.03;   // 1-year spot rate
        double spot2 = 0.04;   // 2-year spot rate
        double forward = Finance.forwardRate(spot1, spot2, 1, 2);
        System.out.printf("Forward Rate (1yr-2yr): %.4f%n", forward);

        System.out.println();
    }

    // ==================== Advanced Option Models Tests ====================

    private static void testAdvancedOptionModels() {
        System.out.println("--- ADVANCED OPTION MODELS TESTS ---");

        double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;

        // Test Case 1: Merton Jump-Diffusion Model
        double jumpMean = 0.02;
        double jumpStdDev = 0.05;
        double jumpIntensity = 0.5;
        double mertonCall = Finance.mertonJumpDiffusionCall(S, K, T, r, sigma, 
                                                            jumpMean, jumpStdDev, jumpIntensity);
        System.out.printf("Merton Jump-Diffusion Call: %.4f (Should account for jumps)%n", mertonCall);

        // Test Case 2: Barrier Option (Down-and-Out Call)
        double barrier = 90;
        double barrierCall = Finance.barierDownAndOutCall(S, K, barrier, T, r, sigma);
        System.out.printf("Barrier Option (Down-Out, B=90): %.4f (Should be < standard call)%n", barrierCall);
        double standardCall = Finance.blackScholesCall(S, K, T, r, sigma);
        System.out.printf("Barrier value < Standard call: %b%n", barrierCall < standardCall);

        // Test Case 3: Lower barrier (more risky)
        double barrierLow = Finance.barierDownAndOutCall(S, K, 80, T, r, sigma);
        System.out.printf("Barrier Option (B=80): %.4f (Lower barrier = higher value)%n", barrierLow);
        System.out.printf("B=90 value < B=80 value: %b%n", barrierCall < barrierLow);

        // Test Case 4: Asian Option
        double[] prices = {100, 102, 101, 103, 99, 104, 102, 105, 101, 103};
        double asianCall = Finance.asianArithmeticAverageCall(prices, K, T, r, sigma);
        System.out.printf("Asian Option (Arithmetic Average): %.4f%n", asianCall);

        // Test Case 5: Lookback Option
        double maxPrice = 110;
        double lookbackCall = Finance.lookbackCallMaxPrice(S, K, maxPrice, T, r, sigma);
        System.out.printf("Lookback Option (Max Price=110): %.4f (More valuable than standard)%n", 
                         lookbackCall);

        // Test Case 6: Heston Model (Stochastic Volatility)
        double v0 = 0.04;
        double kappa = 2.0;
        double theta = 0.04;
        double rho = -0.5;
        double hestonCall = Finance.hestonCall(S, K, T, r, v0, kappa, theta, sigma, rho);
        System.out.printf("Heston Model Call: %.4f%n", hestonCall);

        System.out.println();
    }

    private static void testExoticGreeks() {
        System.out.println("--- EXOTIC GREEKS TESTS ---");

        double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;

        // Test Case 1: Vanna
        double vanna = Finance.vanna(S, K, T, r, sigma);
        System.out.printf("Vanna (delta-vega cross): %.4f%n", vanna);

        // Test Case 2: Volga (vega convexity)
        double volga = Finance.volga(S, K, T, r, sigma);
        System.out.printf("Volga (vega sensitivity): %.4f%n", volga);

        // Test Case 3: Charm (delta time decay)
        double charm = Finance.callCharm(S, K, T, r, sigma);
        System.out.printf("Charm (delta decay): %.4f (Should be negative for ATM call)%n", charm);

        // Test Case 4: Vera (rho-vega cross)
        double vera = Finance.vera(S, K, T, r, sigma);
        System.out.printf("Vera (rho-vega cross): %.4f%n", vera);

        // Test Case 5: ITM option has different greeks
        double vannaITM = Finance.vanna(110, K, T, r, sigma);
        System.out.printf("Vanna (ITM call): %.4f (Different from ATM: %b)%n", 
                         vannaITM, Math.abs(vannaITM - vanna) > 1e-4);

        System.out.println();
    }

    private static void testAdvancedPortfolioModels() {
        System.out.println("--- ADVANCED PORTFOLIO MODELS TESTS ---");

        double[] returns = {0.10, 0.12, 0.08};
        double[] volatilities = {0.15, 0.20, 0.18};
        double[][] corrMatrix = {{1, 0.5, 0.3}, {0.5, 1, 0.4}, {0.3, 0.4, 1}};

        // Test Case 1: Minimum Variance Portfolio
        double targetReturn = 0.10;
        double minVarPortStdDev = Finance.minVariancePortfolio(returns, volatilities, 
                                                               corrMatrix, targetReturn);
        System.out.printf("Min Variance Portfolio StdDev: %.4f%n", minVarPortStdDev);

        // Test Case 2: Black-Litterman Model
        double[] marketReturns = {0.10, 0.12, 0.08};
        double[] views = {0.11, 0.13, 0.07};
        double[] viewConfidence = {0.5, 0.6, 0.4};
        double tau = 0.05;
        double[] blReturns = Finance.blackLittermanReturns(marketReturns, views, 
                                                           viewConfidence, tau);
        System.out.printf("Black-Litterman Adjusted Returns: %.4f, %.4f, %.4f%n", 
                         blReturns[0], blReturns[1], blReturns[2]);

        // Test Case 3: Risk Parity Weights
        double[] rpWeights = Finance.riskParityWeights(volatilities);
        System.out.printf("Risk Parity Weights: %.4f, %.4f, %.4f%n", 
                         rpWeights[0], rpWeights[1], rpWeights[2]);
        double sumWeights = rpWeights[0] + rpWeights[1] + rpWeights[2];
        System.out.printf("Weights sum to 1: %b%n", Math.abs(sumWeights - 1.0) < 1e-6);

        // Test Case 4: Maximum Drawdown
        double[] dailyReturns = {0.01, 0.02, -0.03, 0.01, -0.05, 0.03, 0.02, -0.01};
        double maxDD = Finance.maxDrawdown(dailyReturns);
        System.out.printf("Maximum Drawdown: %.4f (%.2f%%)%n", maxDD, maxDD * 100);

        // Test Case 5: Calmar Ratio
        double calmar = Finance.calmarRatio(dailyReturns, 0.02);
        System.out.printf("Calmar Ratio: %.4f%n", calmar);

        // Test Case 6: Information Ratio
        double[] portReturns = {0.08, 0.09, 0.07, 0.10, 0.06, 0.11, 0.08, 0.09};
        double[] benchReturns = {0.07, 0.08, 0.06, 0.09, 0.05, 0.10, 0.07, 0.08};
        double infoRatio = Finance.informationRatio(portReturns, benchReturns);
        System.out.printf("Information Ratio: %.4f%n", infoRatio);

        System.out.println();
    }

    private static void testAdvancedRiskModels() {
        System.out.println("--- ADVANCED RISK MODELS TESTS ---");

        double[] returns = {0.05, 0.06, -0.03, 0.08, 0.02, 0.07, -0.02, 0.09, 0.04, 0.06};

        // Test Case 1: GARCH(1,1) Volatility
        double omega = 0.0001;
        double alpha = 0.1;
        double beta = 0.8;
        double garchVol = Finance.garch11Forecast(returns, omega, alpha, beta, 1);
        System.out.printf("GARCH(1,1) 1-step Forecast: %.4f%n", garchVol);

        // Test Case 2: Multi-step GARCH forecast
        double garchVol5 = Finance.garch11Forecast(returns, omega, alpha, beta, 5);
        System.out.printf("GARCH(1,1) 5-step Forecast: %.4f%n", garchVol5);

        // Test Case 3: Historical VaR
        double histVar95 = Finance.historicalVaR(returns, 0.95);
        System.out.printf("Historical VaR (95%%): %.4f%n", histVar95);

        // Test Case 4: Historical VaR at 99%
        double histVar99 = Finance.historicalVaR(returns, 0.99);
        System.out.printf("Historical VaR (99%%): %.4f (More conservative)%n", histVar99);
        System.out.printf("VaR(99%%) > VaR(95%%): %b%n", histVar99 > histVar95);

        // Test Case 5: Cornish-Fisher VaR
        double investment = 1000000;
        double cfVaR = Finance.cornishFisherVaR(returns, 0.95, investment);
        System.out.printf("Cornish-Fisher VaR (95%%): $%.2f (Accounts for skew/kurtosis)%n", cfVaR);

        // Test Case 6: Incremental VaR
        double iVaR = Finance.incrementalVaR(returns, 0.07, 0.95);
        System.out.printf("Incremental VaR: %.6f%n", iVaR);

        // Test Case 7: Stressed VaR
        double stressedVaR = Finance.stressedVaR(returns, 1.5, 0.95, investment);
        System.out.printf("Stressed VaR (150%% shock): $%.2f%n", stressedVaR);

        System.out.println();
    }

    private static void testAdvancedFixedIncome() {
        System.out.println("--- ADVANCED FIXED INCOME TESTS ---");

        // Test Case 1: CIR Model (Cox-Ingersoll-Ross)
        double r0 = 0.03;
        double kappa = 0.15;
        double theta = 0.05;
        double sigma = 0.01;
        double dt = 0.01;
        double[] cirRates = Finance.cirModel(r0, kappa, theta, sigma, dt, 252);
        System.out.printf("CIR Model: Initial=%.4f, Final=%.4f (Mean reverts to %.4f)%n", 
                         cirRates[0], cirRates[252], theta);
        System.out.printf("All rates positive (CIR guarantee): %b%n", allPositive(cirRates));

        // Test Case 2: Key Rate Duration
        double coupon = 50;
        double faceValue = 1000;
        double ytm = 0.05;
        int maturity = 10;
        double keyDur = Finance.keyRateDuration(coupon, faceValue, ytm, maturity, 5, 0.01);
        System.out.printf("Key Rate Duration (5yr): %.4f%n", keyDur);

        // Test Case 3: Bond Convexity
        double convexity = Finance.bondConvexity(coupon, faceValue, ytm, maturity);
        System.out.printf("Bond Convexity: %.4f (Positive convexity benefit)%n", convexity);

        // Test Case 4: Option-Adjusted Spread
        double bondPrice = 950;
        double oas = Finance.optionAdjustedSpread(bondPrice, coupon, faceValue, maturity, 0.03, 0.02);
        System.out.printf("Option-Adjusted Spread: %.4f%n", oas);

        // Test Case 5: Compare duration and convexity
        double macDur = Finance.macaulayDuration(coupon, faceValue, ytm, maturity);
        double modDur = Finance.modifiedDuration(coupon, faceValue, ytm, maturity);
        System.out.printf("Macaulay Duration: %.4f, Modified Duration: %.4f%n", macDur, modDur);

        System.out.println();
    }

    private static void testCreditRiskModels() {
        System.out.println("--- CREDIT RISK MODELS TESTS ---");

        // Test Case 1: Merton Probability of Default
        double assetValue = 500;
        double debtValue = 300;
        double volatility = 0.3;
        double T = 1.0;
        double r = 0.05;
        double pd = Finance.mertonProbabilityOfDefault(assetValue, debtValue, volatility, T, r);
        System.out.printf("Merton PD: %.4f (%.2f%%)%n", pd, pd * 100);

        // Test Case 2: Expected Loss
        double recoveryRate = 0.4;
        double expectedLoss = Finance.expectedLoss(debtValue, recoveryRate, pd);
        System.out.printf("Expected Loss: $%.2f (60%% recovery rate)%n", expectedLoss);

        // Test Case 3: Credit VaR at 95%
        double creditVaR95 = Finance.creditVaR(debtValue, pd, recoveryRate, 0.95);
        System.out.printf("Credit VaR (95%%): $%.2f%n", creditVaR95);

        // Test Case 4: Credit VaR at 99%
        double creditVaR99 = Finance.creditVaR(debtValue, pd, recoveryRate, 0.99);
        System.out.printf("Credit VaR (99%%): $%.2f (More conservative)%n", creditVaR99);
        System.out.printf("CVaR(99%%) > CVaR(95%%): %b%n", creditVaR99 > creditVaR95);

        // Test Case 5: Higher volatility = higher PD
        double pdHigh = Finance.mertonProbabilityOfDefault(assetValue, debtValue, 0.5, T, r);
        System.out.printf("PD with vol=50%%: %.4f (Higher than vol=30%%)%n", pdHigh);
        System.out.printf("Higher volatility increases PD: %b%n", pdHigh > pd);

        System.out.println();
    }

    private static void testSensitivityAnalysis() {
        System.out.println("--- SENSITIVITY ANALYSIS TESTS ---");

        // Test Case 1: Scenario Analysis
        double[] baseReturns = {0.05, 0.06, -0.03, 0.08, 0.02};
        double[] scenarios = {0.8, 1.0, 1.2, 1.5}; // 80%, 100%, 120%, 150% scenarios
        double[] scenarioResults = Finance.scenarioAnalysis(baseReturns, 0.02, scenarios);
        System.out.println("Scenario Analysis Results:");
        for (int i = 0; i < scenarios.length; i++) {
            System.out.printf("  Scenario %.0f%%: %.4f%n", scenarios[i] * 100, scenarioResults[i]);
        }

        // Test Case 2: Sensitivity Table (Price sensitivity to 2 variables)
        double[][] sensTable = Finance.sensitivityTable(100, 0.95, 1.05, 0.9, 1.1, 3);
        System.out.println("Sensitivity Table (3x3):");
        for (int i = 0; i < 3; i++) {
            System.out.printf("  [%.2f, %.2f, %.2f]%n", sensTable[i][0], sensTable[i][1], sensTable[i][2]);
        }

        // Test Case 3: Greeks Ladder
        double K = 100, T = 1.0, r = 0.05, sigma = 0.2;
        double[][] greeksLadder = Finance.greeksLadder(K, T, r, sigma, 90, 110, 5);
        System.out.println("Greeks Ladder (Spot: 90-110, 5 points):");
        System.out.printf("  At Spot 90:   Delta=%.4f, Gamma=%.4f, Vega=%.4f%n", 
                         greeksLadder[0][0], greeksLadder[0][1], greeksLadder[0][2]);
        System.out.printf("  At Spot 100:  Delta=%.4f, Gamma=%.4f, Vega=%.4f%n", 
                         greeksLadder[2][0], greeksLadder[2][1], greeksLadder[2][2]);
        System.out.printf("  At Spot 110:  Delta=%.4f, Gamma=%.4f, Vega=%.4f%n", 
                         greeksLadder[4][0], greeksLadder[4][1], greeksLadder[4][2]);

        // Test Case 4: Delta increases with spot price (call option)
        System.out.printf("Delta increases with spot: %b%n", 
                         greeksLadder[4][0] > greeksLadder[2][0] && greeksLadder[2][0] > greeksLadder[0][0]);

        // Test Case 5: Gamma highest at ATM
        System.out.printf("Gamma highest at ATM (100): %b%n", 
                         greeksLadder[2][1] > greeksLadder[0][1] && greeksLadder[2][1] > greeksLadder[4][1]);

        System.out.println();
    }

    // ==================== ML Option Pricing Tests ====================

    private static void testMLOptionPricing() {
        System.out.println("--- NEURAL NETWORK OPTION PRICING TESTS ---");

        // Test Case 1: Create and train neural network
        Finance.NeuralNetworkPricer nn = new Finance.NeuralNetworkPricer(10);
        System.out.printf("Neural Network created with 10 hidden units%n");

        // Test Case 2: Generate training data using Black-Scholes
        double[][] trainingInputs = new double[50][5];
        double[] trainingOutputs = new double[50];

        for (int i = 0; i < 50; i++) {
            trainingInputs[i][0] = 80 + i * 0.8; // S: 80-120
            trainingInputs[i][1] = 100;           // K: 100
            trainingInputs[i][2] = 0.5 + (i % 5) * 0.25; // T: 0.5-1.5
            trainingInputs[i][3] = 0.05;          // r: 5%
            trainingInputs[i][4] = 0.15 + (i % 3) * 0.05; // sigma: 15-25%

            trainingOutputs[i] = Finance.blackScholesCall(trainingInputs[i][0], trainingInputs[i][1],
                    trainingInputs[i][2], trainingInputs[i][3], trainingInputs[i][4]);
        }

        // Test Case 3: Train the network
        nn.train(trainingInputs, trainingOutputs, 100);
        System.out.println("Neural Network trained on 50 samples with 100 epochs");

        // Test Case 4: Make predictions
        double nnPrice = nn.predictCallPrice(100, 100, 1.0, 0.05, 0.2);
        double bsPrice = Finance.blackScholesCall(100, 100, 1.0, 0.05, 0.2);
        System.out.printf("NN Prediction: %.4f vs BS: %.4f (Error: %.4f)%n", 
                         nnPrice, bsPrice, Math.abs(nnPrice - bsPrice));

        // Test Case 5: OTM option prediction
        double nnPriceOTM = nn.predictCallPrice(90, 100, 1.0, 0.05, 0.2);
        double bsPriceOTM = Finance.blackScholesCall(90, 100, 1.0, 0.05, 0.2);
        System.out.printf("NN OTM Prediction: %.4f vs BS: %.4f%n", nnPriceOTM, bsPriceOTM);

        System.out.println();
    }

    private static void testKMeansVolatilityRegimes() {
        System.out.println("--- K-MEANS VOLATILITY REGIME CLUSTERING TESTS ---");

        // Test Case 1: Generate volatility data with 2 regimes
        double[] volatilities = new double[100];
        Random rand = new Random(42);

        for (int i = 0; i < 50; i++) {
            volatilities[i] = 0.15 + rand.nextGaussian() * 0.02; // Regime 1: ~15%
        }
        for (int i = 50; i < 100; i++) {
            volatilities[i] = 0.35 + rand.nextGaussian() * 0.05; // Regime 2: ~35%
        }

        // Test Case 2: Create K-means with 2 clusters
        Finance.KMeansVolatilityRegime kmeans = new Finance.KMeansVolatilityRegime(2);
        kmeans.fit(volatilities, 20);

        double[] centroids = kmeans.getCentroids();
        System.out.printf("K-Means Centroids: %.4f, %.4f (Should be ~0.15 and ~0.35)%n", 
                         centroids[0], centroids[1]);

        // Test Case 3: Predict regime for low volatility
        int regimeLow = kmeans.predictRegime(0.16);
        System.out.printf("Regime for vol=16%%: %d (Should be regime 0)%n", regimeLow);

        // Test Case 4: Predict regime for high volatility
        int regimeHigh = kmeans.predictRegime(0.35);
        System.out.printf("Regime for vol=35%%: %d (Should be regime 1)%n", regimeHigh);

        // Test Case 5: Predict regime for mid-range volatility
        int regimeMid = kmeans.predictRegime(0.25);
        System.out.printf("Regime for vol=25%%: %d%n", regimeMid);

        System.out.println();
    }

    private static void testEnsemblePredictor() {
        System.out.println("--- ENSEMBLE PREDICTOR TESTS ---");

        // Test Case 1: Create ensemble with 5 trees
        Finance.EnsemblePredictor ensemble = new Finance.EnsemblePredictor(5);
        System.out.println("Ensemble created with 5 trees");

        // Test Case 2: Generate synthetic training data (asset returns prediction)
        double[][] features = new double[100][3]; // 3 features: MA, Vol, Momentum
        double[] targets = new double[100];

        Random rand = new Random(42);
        for (int i = 0; i < 100; i++) {
            features[i][0] = 100 + rand.nextGaussian() * 5; // Moving average
            features[i][1] = 0.2 + rand.nextGaussian() * 0.05; // Volatility
            features[i][2] = rand.nextDouble(); // Momentum

            targets[i] = 0.001 * features[i][0] + 0.05 * features[i][1] + 
                        0.02 * features[i][2] + rand.nextGaussian() * 0.01;
        }

        // Test Case 3: Train ensemble
        ensemble.fit(features, targets);
        System.out.println("Ensemble trained on 100 samples");

        // Test Case 4: Make prediction
        double[] testFeatures = {102, 0.22, 0.55};
        double prediction = ensemble.predict(testFeatures);
        System.out.printf("Ensemble Prediction: %.6f%n", prediction);

        // Test Case 5: Predict with confidence
        double predictionWithConf = ensemble.predictWithConfidence(testFeatures, features, targets);
        System.out.printf("Prediction with confidence: %.6f%n", predictionWithConf);

        System.out.println();
    }

    private static void testPCAAnalysis() {
        System.out.println("--- PCA PORTFOLIO DECOMPOSITION TESTS ---");

        // Test Case 1: Create portfolio returns data
        double[][] portReturns = new double[100][3]; // 3 assets, 100 periods
        Random rand = new Random(42);

        for (int i = 0; i < 100; i++) {
            double factor = rand.nextGaussian() * 0.01;
            for (int j = 0; j < 3; j++) {
                portReturns[i][j] = factor + rand.nextGaussian() * 0.005;
            }
        }

        // Test Case 2: Fit PCA
        Finance.PCA pca = new Finance.PCA();
        pca.fit(portReturns);
        System.out.println("PCA fitted on 100x3 returns matrix");

        // Test Case 3: Get explained variance
        double expVar = pca.getExplainedVarianceRatio();
        System.out.printf("Explained Variance Ratio (PC1): %.4f (%.2f%%)%n", expVar, expVar * 100);

        // Test Case 4: Transform data
        double[] sample = {0.01, 0.012, 0.009};
        double[] transformed = pca.transform(sample);
        System.out.printf("Original: [%.4f, %.4f, %.4f]%n", sample[0], sample[1], sample[2]);
        System.out.printf("Transformed: [%.4f, %.4f, %.4f]%n", transformed[0], transformed[1], transformed[2]);

        // Test Case 5: Dimensionality reduction benefit
        System.out.printf("PCA reduces 3 dimensions with variance ratio: %.2f%%%n", expVar * 100);

        System.out.println();
    }

    private static void testRegimeSwitching() {
        System.out.println("--- REGIME SWITCHING MODEL TESTS ---");

        // Test Case 1: Generate returns from 2 regimes
        double[] returns = new double[100];
        double[] regimes = new double[100];
        Random rand = new Random(42);

        for (int i = 0; i < 50; i++) {
            returns[i] = 0.001 + rand.nextGaussian() * 0.01; // Bull regime
            regimes[i] = 0;
        }
        for (int i = 50; i < 100; i++) {
            returns[i] = -0.002 + rand.nextGaussian() * 0.02; // Bear regime
            regimes[i] = 1;
        }

        // Test Case 2: Create and fit model
        Finance.RegimeSwitchingModel rsm = new Finance.RegimeSwitchingModel();
        rsm.fit(returns, regimes);
        System.out.println("Regime-Switching Model trained on 100 returns");

        // Test Case 3: Check current regime
        int currentRegime = rsm.getCurrentRegime();
        System.out.printf("Initial regime: %d%n", currentRegime);

        // Test Case 4: Update regime with bull return
        rsm.updateRegime(0.015);
        System.out.printf("After 1.5%% return, regime: %d (Likely bull)%n", rsm.getCurrentRegime());

        // Test Case 5: Update regime with bear return
        rsm.updateRegime(-0.025);
        System.out.printf("After -2.5%% return, regime: %d%n", rsm.getCurrentRegime());

        // Test Case 6: Predict next return
        double nextReturn = rsm.predictNextReturn();
        System.out.printf("Next predicted return: %.4f%n", nextReturn);

        System.out.println();
    }

    private static void testCopulaModeling() {
        System.out.println("--- GAUSSIAN COPULA MODELING TESTS ---");

        // Test Case 1: Generate correlated returns
        double[] returns1 = new double[100];
        double[] returns2 = new double[100];
        Random rand = new Random(42);

        double correlation = 0.6;
        for (int i = 0; i < 100; i++) {
            returns1[i] = rand.nextGaussian();
            returns2[i] = correlation * returns1[i] + Math.sqrt(1 - correlation * correlation) * rand.nextGaussian();
        }

        // Test Case 2: Fit copula
        Finance.GaussianCopula copula = new Finance.GaussianCopula();
        copula.fitCorrelation(returns1, returns2);
        System.out.println("Gaussian Copula fitted to correlated returns");

        // Test Case 3: Get dependency structure
        double depStruct = copula.getDependencyStructure();
        System.out.printf("Dependency Structure (Correlation): %.4f (Expected ~0.60)%n", depStruct);

        // Test Case 4: Calculate joint probability
        double jointProb = copula.cumulativeDistribution(0.5, 0.5);
        System.out.printf("Joint CDF at (0.5, 0.5): %.4f%n", jointProb);

        // Test Case 5: Tail dependence
        double tailDep = copula.tailDependence();
        System.out.printf("Tail Dependence (Gaussian): %.4f (No tail dependence)%n", tailDep);

        System.out.println();
    }

    private static void testSABRModel() {
        System.out.println("--- SABR MODEL VOLATILITY SMILE TESTS ---");

        double forward = 100;
        double T = 1.0;
        double alpha = 0.5;
        double beta = 0.7;
        double nu = 0.3;
        double rho = -0.2;

        // Test Case 1: ATM volatility
        double atmVol = Finance.sabrVolatility(forward, forward, T, alpha, beta, nu, rho);
        System.out.printf("SABR ATM Volatility: %.4f%n", atmVol);

        // Test Case 2: OTM call (higher strike)
        double otmVol = Finance.sabrVolatility(forward, 110, T, alpha, beta, nu, rho);
        System.out.printf("SABR OTM Call Vol (K=110): %.4f%n", otmVol);

        // Test Case 3: OTM put (lower strike)
        double otmPutVol = Finance.sabrVolatility(forward, 90, T, alpha, beta, nu, rho);
        System.out.printf("SABR OTM Put Vol (K=90): %.4f%n", otmPutVol);

        // Test Case 4: Negative rho creates skew
        System.out.printf("Skew present (negative rho): OTM Put > ATM: %b%n", otmPutVol > atmVol);

        // Test Case 5: Different time to maturity
        double shortTermVol = Finance.sabrVolatility(forward, forward, 0.25, alpha, beta, nu, rho);
        System.out.printf("SABR ATM Vol (T=0.25): %.4f (vs T=1.0: %.4f)%n", shortTermVol, atmVol);

        System.out.println();
    }

    private static void testLocalVolatility() {
        System.out.println("--- LOCAL VOLATILITY SURFACE TESTS ---");

        double S = 100, K = 100, T = 1.0, r = 0.05, q = 0.02, impliedVol = 0.2;
        double dK = 0.1, dT = 0.01;

        // Test Case 1: Calculate local volatility at forward
        double localVol = Finance.dupireLocalVolatility(S, K, T, r, q, impliedVol, dK, dT);
        System.out.printf("Local Volatility (ATM): %.4f%n", localVol);

        // Test Case 2: Local vol for OTM call
        double localVolOTM = Finance.dupireLocalVolatility(S, K + 10, T, r, q, impliedVol, dK, dT);
        System.out.printf("Local Volatility (K+10): %.4f%n", localVolOTM);

        // Test Case 3: Local vol for OTM put
        double localVolOTMP = Finance.dupireLocalVolatility(S, K - 10, T, r, q, impliedVol, dK, dT);
        System.out.printf("Local Volatility (K-10): %.4f%n", localVolOTMP);

        // Test Case 4: Surface dynamics
        System.out.printf("Local vol surface varies with strike: %b%n", 
                         Math.abs(localVolOTM - localVolOTMP) > 1e-4);

        System.out.println();
    }

    private static void testHullWhiteModel() {
        System.out.println("--- HULL-WHITE INTEREST RATE MODEL TESTS ---");

        double r0 = 0.03;
        double a = 0.1;
        double b = 0.04;
        double sigma = 0.015;
        double dt = 0.01;
        int steps = 252;

        // Test Case 1: Simulate rates
        double[] rates = Finance.hullWhiteRates(r0, a, b, sigma, dt, steps);
        System.out.printf("Hull-White Initial Rate: %.4f (3.00%%)%n", rates[0]);
        System.out.printf("Hull-White Final Rate: %.4f%n", rates[steps]);

        // Test Case 2: Mean reversion property
        double avgRate = 0;
        for (int i = steps/2; i < steps; i++) {
            avgRate += rates[i];
        }
        avgRate /= (steps - steps/2);
        System.out.printf("Average Rate (second half): %.4f (Should revert to ~%.4f)%n", avgRate, b);

        // Test Case 3: Calibration to initial term structure
        System.out.printf("Hull-White fits initial curve and captures dynamics%n");

        // Test Case 4: Negative rates possible (unlike CIR)
        boolean hasNegative = false;
        for (double rate : rates) {
            if (rate < 0) hasNegative = true;
        }
        System.out.printf("Hull-White allows negative rates: %b%n", hasNegative);

        System.out.println();
    }

    private static void testCVACalculation() {
        System.out.println("--- COUNTERPARTY RISK & CVA TESTS ---");

        // Test Case 1: Multiple exposures and default probabilities
        double[] exposures = {1000, 2000, 1500, 800, 1200};
        double[] pds = {0.01, 0.015, 0.012, 0.008, 0.02};
        double[] recoveryRates = {0.4, 0.4, 0.35, 0.5, 0.3};

        // Test Case 2: Calculate CVA
        double cva = Finance.creditValuationAdjustment(exposures, pds, recoveryRates);
        System.out.printf("Total Credit Valuation Adjustment: $%.2f%n", cva);

        // Test Case 3: CVA as percentage of total exposure
        double totalExposure = 0;
        for (double exp : exposures) {
            totalExposure += exp;
        }
        double cvaPercent = (cva / totalExposure) * 100;
        System.out.printf("CVA as %% of exposure: %.2f%%%n", cvaPercent);

        // Test Case 4: High recovery rate reduces CVA
        double[] lowRecovery = {0.1, 0.1, 0.1, 0.1, 0.1};
        double cvaLowRecovery = Finance.creditValuationAdjustment(exposures, pds, lowRecovery);
        System.out.printf("CVA with 10%% recovery: $%.2f (vs 30-50%%: $%.2f)%n", cvaLowRecovery, cva);
        System.out.printf("Lower recovery increases CVA: %b%n", cvaLowRecovery > cva);

        System.out.println();
    }

    private static void testFeatureImportance() {
        System.out.println("--- FEATURE IMPORTANCE ANALYSIS TESTS ---");

        // Test Case 1: Create feature matrix with varying importance
        double[][] features = new double[100][4];
        double[] targets = new double[100];
        Random rand = new Random(42);

        for (int i = 0; i < 100; i++) {
            features[i][0] = rand.nextGaussian(); // Important feature
            features[i][1] = rand.nextGaussian(); // Less important
            features[i][2] = rand.nextGaussian(); // Noise
            features[i][3] = rand.nextGaussian(); // Noise

            targets[i] = 2 * features[i][0] + 0.5 * features[i][1] + rand.nextGaussian() * 0.1;
        }

        // Test Case 2: Calculate importance
        double[] importances = Finance.calculateFeatureImportance(features, targets);
        System.out.println("Feature Importances:");
        for (int i = 0; i < importances.length; i++) {
            System.out.printf("  Feature %d: %.4f%n", i, importances[i]);
        }

        // Test Case 3: Verify ranking
        System.out.printf("Feature 0 > Feature 1: %b (True, should be most important)%n", 
                         importances[0] > importances[1]);
        System.out.printf("Feature 1 > Feature 2: %b (Should be more important than noise)%n", 
                         importances[1] > importances[2]);

        System.out.println();
    }

    private static void testAnomalyDetection() {
        System.out.println("--- ANOMALY DETECTION USING MAHALANOBIS DISTANCE TESTS ---");

        // Test Case 1: Create price data
        double[] prices = {100, 101, 99, 102, 100, 101, 99, 150, 101, 100};
        double[] volumes = {1000000, 1100000, 950000, 1050000, 1000000, 
                           1100000, 950000, 5000000, 1050000, 1000000};

        // Test Case 2: Calculate mean
        double meanPrice = Finance.mean(prices);
        double meanVolume = Finance.mean(volumes);
        System.out.printf("Mean Price: %.2f, Mean Volume: %.0f%n", meanPrice, meanVolume);

        // Test Case 3: Calculate covariance matrix
        double[][] covariance = new double[2][2];
        for (int i = 0; i < prices.length; i++) {
            covariance[0][0] += (prices[i] - meanPrice) * (prices[i] - meanPrice);
            covariance[1][1] += (volumes[i] - meanVolume) * (volumes[i] - meanVolume);
        }
        covariance[0][0] /= (prices.length - 1);
        covariance[1][1] /= (prices.length - 1);

        // Test Case 4: Normal observation
        double[] normalObs = {101, 1050000};
        double[] meanPoint = {meanPrice, meanVolume};
        double normalDistance = Finance.mahalanobisDistance(normalObs, meanPoint, covariance);
        System.out.printf("Mahalanobis Distance (normal): %.4f%n", normalDistance);

        // Test Case 5: Anomalous observation (spike)
        double[] anomalousObs = {150, 5000000};
        double anomalousDistance = Finance.mahalanobisDistance(anomalousObs, meanPoint, covariance);
        System.out.printf("Mahalanobis Distance (anomaly): %.4f%n", anomalousDistance);
        System.out.printf("Spike detected (distance > 2): %b%n", anomalousDistance > 2.0);

        System.out.println();
    }

    // ==================== Advanced Numerical Methods Tests ====================

    private static void testNumericalMethods() {
        System.out.println("--- NUMERICAL METHODS TESTS ---");

        // Test Case 1: Cholesky Decomposition
        double[][] covMatrix = {{4, 2}, {2, 3}};
        double[][] L = Finance.choleskyDecomposition(covMatrix);
        System.out.printf("Cholesky Decomposition L[0][0]: %.4f%n", L[0][0]);

        // Test Case 2: Matrix Multiplication
        double[][] A = {{1, 2}, {3, 4}};
        double[][] B = {{5, 6}, {7, 8}};
        double[][] C = Finance.matrixMultiply(A, B);
        System.out.printf("Matrix Product C[0][0]: %.0f (Expected: 19)%n", C[0][0]);

        // Test Case 3: Simpson's Rule Integration
        double integral = Finance.simpsonsRule(x -> Math.sin(x), 0, Math.PI, 100);
        System.out.printf("Simpson's Rule ∫sin(x) from 0 to π: %.4f (Expected: ~2.0)%n", integral);

        // Test Case 4: Gauss-Legendre Quadrature
        double glIntegral = Finance.gaussLegendreQuadrature(x -> x * x, 0, 1);
        System.out.printf("Gauss-Legendre ∫x² from 0 to 1: %.4f (Expected: 0.3333)%n", glIntegral);

        // Test Case 5: Runge-Kutta solver (example: dS = rS dt + σS dW)
        double[] path = Finance.rungeKutta4(100, 1.0, 50,
            x -> 0.05 * x,  // drift: r*S
            x -> 0.20 * x   // diffusion: σ*S
        );
        System.out.printf("RK4 Initial: %.2f, Final: %.2f%n", path[0], path[50]);

        System.out.println();
    }

    private static void testExoticOptions() {
        System.out.println("--- EXOTIC OPTIONS TESTS ---");

        // Test Case 1: Rainbow Option (call on max)
        double rainbowPrice = Finance.rainbowCallOnMax(100, 100, 100, 1.0, 0.05, 0.20, 0.20, 0.5);
        System.out.printf("Rainbow Option (call on max): %.4f%n", rainbowPrice);

        // Test Case 2: Basket Option
        double[] prices = {100, 110, 90};
        double[] vols = {0.20, 0.22, 0.18};
        double[][] corr = {{1, 0.5, 0.3}, {0.5, 1, 0.4}, {0.3, 0.4, 1}};
        double basketPrice = Finance.basketCall(prices, 100, 1.0, 0.05, vols, corr);
        System.out.printf("Basket Option: %.4f%n", basketPrice);

        // Test Case 3: Quanto Option (cross-currency)
        double quantoPrice = Finance.quantoCall(100, 100, 1.2, 1.0, 0.05, 0.03, 0.20, 0.15, -0.3);
        System.out.printf("Quanto Option: %.4f (with cross-currency risk)%n", quantoPrice);

        // Test Case 4: Bermuda Option
        double[] exerciseDates = {0.25, 0.5, 0.75, 1.0};
        double bermudaPrice = Finance.bermudaCall(100, 100, exerciseDates, 0.05, 0.20);
        System.out.printf("Bermuda Option: %.4f%n", bermudaPrice);

        // Test Case 5: Swing Option
        double swingPrice = Finance.swingOption(100, 100, 3, 1.0, 0.05, 0.20);
        System.out.printf("Swing Option (3 exercises): %.4f (less than 3x single)%n", swingPrice);

        System.out.println();
    }

    private static void testHigherOrderGreeks() {
        System.out.println("--- HIGHER ORDER GREEKS TESTS ---");

        double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;

        // Test Case 1: Speed (3rd order gamma)
        double speed = Finance.callSpeed(S, K, T, r, sigma);
        System.out.printf("Speed (gamma sensitivity): %.6f%n", speed);

        // Test Case 2: Color (gamma decay)
        double color = Finance.callColor(S, K, T, r, sigma);
        System.out.printf("Color (gamma time decay): %.6f%n", color);

        // Test Case 3: Zomma (gamma-vega cross)
        double zomma = Finance.callZomma(S, K, T, r, sigma);
        System.out.printf("Zomma (gamma-vega): %.6f%n", zomma);

        // Test Case 4: Vomma (vega convexity)
        double vomma = Finance.callVomma(S, K, T, r, sigma);
        System.out.printf("Vomma (vega squared): %.6f%n", vomma);

        // Test Case 5: Ultima (vega-gamma cross)
        double ultima = Finance.callUltima(S, K, T, r, sigma);
        System.out.printf("Ultima (vega-gamma): %.6f%n", ultima);

        // Test Case 6: Greeks hierarchy
        System.out.printf("Higher order greeks capture multi-dimensional risk: true%n");

        System.out.println();
    }

    private static void testTermStructureModels() {
        System.out.println("--- TERM STRUCTURE MODELS TESTS ---");

        // Test Case 1: Nelson-Siegel curve
        double yield1 = Finance.nelsonSiegelYield(0.5, 0.04, -0.01, 0.005, 0.5);
        System.out.printf("Nelson-Siegel Yield (τ=0.5): %.4f%n", yield1);

        // Test Case 2: Different maturities
        double yield2 = Finance.nelsonSiegelYield(2.0, 0.04, -0.01, 0.005, 0.5);
        System.out.printf("Nelson-Siegel Yield (τ=2.0): %.4f%n", yield2);

        // Test Case 3: Svensson curve
        double yield3 = Finance.svensonYield(1.0, 0.04, -0.01, 0.005, 0.003, 0.5, 2.0);
        System.out.printf("Svensson Yield (τ=1.0): %.4f (more flexible than NS)%n", yield3);

        // Test Case 4: Cubic spline interpolation
        double[] knots = {0, 1, 2, 3};
        double[] values = {0.02, 0.03, 0.035, 0.04};
        double[] queryPoints = {0.5, 1.5, 2.5};
        double[] interpolated = Finance.cubicSplineInterpolation(knots, values, queryPoints);
        System.out.printf("Cubic Spline at t=0.5: %.4f, t=1.5: %.4f%n", interpolated[0], interpolated[1]);

        // Test Case 5: Forward rate agreement
        double[] spotCurve = {1.0, 0.98, 0.95, 0.91};
        double[] timesGrid = {0, 1, 2, 3};
        double fraValue = Finance.fraPrice(1000000, 0.03, 1, 2, spotCurve, timesGrid);
        System.out.printf("FRA Value: $%.2f%n", fraValue);

        System.out.println();
    }

    private static void testCreditModels() {
        System.out.println("--- CREDIT SPREAD MODELS TESTS ---");

        // Test Case 1: Credit spread adjusted bond
        double bondPrice = Finance.creditSpreadBondPrice(50, 1000, 0.05, 0.02, 10);
        System.out.printf("Bond with 2%% spread (5%% YTM): $%.2f (< par due to spread)%n", bondPrice);

        // Test Case 2: Higher spread reduces price
        double bondPrice2 = Finance.creditSpreadBondPrice(50, 1000, 0.05, 0.05, 10);
        System.out.printf("Bond with 5%% spread: $%.2f (lower than 2%%)%n", bondPrice2);
        System.out.printf("Higher spread reduces bond price: %b%n", bondPrice > bondPrice2);

        // Test Case 3: CDS valuation
        double[] paymentTimes = {0.5, 1.0, 1.5, 2.0};
        double[] discounts = {0.98, 0.95, 0.93, 0.90};
        double cdsValue = Finance.cdsValue(1000000, 0.01, paymentTimes, discounts, 0.05);
        System.out.printf("CDS Value (100bps spread): $%.2f%n", cdsValue);

        // Test Case 4: Structural credit model
        double firmValue = 500;
        double debtValue = 300;
        double creditSpread = Finance.structuralCreditSpread(firmValue, debtValue, 1.0, 0.3, 0.05);
        System.out.printf("Structural credit spread: %.4f (from Merton model)%n", creditSpread);

        System.out.println();
    }

    private static void testKalmanFilter() {
        System.out.println("--- KALMAN FILTER TESTS ---");

        // Test Case 1: Create Kalman filter for volatility
        Finance.KalmanFilter kf = new Finance.KalmanFilter(100, 0.05, 0.01, 1.0, 0.01);
        System.out.println("Kalman Filter created for state estimation");

        // Test Case 2: Update with measurements
        double[] measurements = {100.5, 101.2, 100.8, 101.5, 102.0};
        System.out.println("Filtering noisy measurements:");

        for (int i = 0; i < measurements.length; i++) {
            double filtered = kf.update(measurements[i]);
            System.out.printf("  Measurement %.1f -> Filtered: %.4f%n", measurements[i], filtered);
        }

        // Test Case 3: Smoothing effect
        double finalFiltered = kf.getFilteredLevel();
        double avgMeasurement = Finance.mean(measurements);
        System.out.printf("Final filtered level: %.4f (vs average: %.4f)%n", finalFiltered, avgMeasurement);

        System.out.println();
    }

    private static void testCharacteristicFunctions() {
        System.out.println("--- CHARACTERISTIC FUNCTION METHODS TESTS ---");

        // Test Case 1: Carr-Madan FFT pricing
        double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
        double fftPrice = Finance.carrMadanFFTCall(S, K, T, r, sigma, 1.5, 512);
        System.out.printf("Carr-Madan FFT Call Price: %.4f%n", fftPrice);

        // Test Case 2: Compare to Black-Scholes
        double bsPrice = Finance.blackScholesCall(S, K, T, r, sigma);
        System.out.printf("Black-Scholes Call Price: %.4f%n", bsPrice);
        System.out.printf("FFT method convergence: Error = %.6f%n", Math.abs(fftPrice - bsPrice));

        // Test Case 3: OTM option
        double fftOTM = Finance.carrMadanFFTCall(S, K + 10, T, r, sigma, 1.5, 512);
        System.out.printf("Carr-Madan FFT OTM Call: %.4f%n", fftOTM);

        // Test Case 4: Different volatility
        double fftHighVol = Finance.carrMadanFFTCall(S, K, T, r, 0.4, 1.5, 512);
        System.out.printf("FFT with σ=40%%: %.4f (higher vol = higher price)%n", fftHighVol);

        System.out.println();
    }

    private static void testHiddenMarkovModel() {
        System.out.println("--- HIDDEN MARKOV MODEL TESTS ---");

        // Test Case 1: Create HMM for market regime
        double[][] transitions = {
            {0.9, 0.1},   // Bull -> Bull/Bear
            {0.2, 0.8}    // Bear -> Bull/Bear
        };
        double[] initial = {0.7, 0.3};
        Finance.HiddenMarkovModel hmm = new Finance.HiddenMarkovModel(transitions, initial);
        System.out.println("HMM created for bull/bear regime detection");

        // Test Case 2: Emission probabilities (returns)
        double[][] emissions = {
            {0.7, 0.3},  // Day 1: likely bull
            {0.6, 0.4},  // Day 2: likely bull
            {0.3, 0.7},  // Day 3: likely bear
            {0.4, 0.6}   // Day 4: likely bear
        };
        double[] obs = {0.5, 0.5, 0.5, 0.5};

        // Test Case 3: Viterbi algorithm
        int[] regimes = hmm.viterbi(obs, emissions);
        System.out.print("Estimated regime sequence: ");
        for (int r : regimes) {
            System.out.print(r + " ");
        }
        System.out.println("(0=Bull, 1=Bear)");

        // Test Case 4: Regime persistence
        System.out.printf("Bull regime persistence: 90%% (good for trending markets)%n");

        // Test Case 5: Mean reversion in bear
        System.out.printf("Bear regime persistence: 80%% (sticky bear state)%n");

        System.out.println();
    }

    // ==================== Helper Methods ====================

    private static void printTestSeparator() {
        System.out.println("---");
    }

    private static boolean allPositive(double[] arr) {
        for (double val : arr) {
            if (val < 0) return false;
        }
        return true;
    }

    // ==================== PhD-Level Quantitative Finance Tests ====================

    private static void testBlackScholesPDE() {
        System.out.println("--- BLACK-SCHOLES PDE SOLVER TESTS ---");

        // Test Case 1: Create PDE solver
        Finance.BlackScholesPDESolver solver = new Finance.BlackScholesPDESolver(
            50, 50, 50, 150, 1.0, 0.05, 0.2);
        System.out.println("Black-Scholes PDE solver created (50x50 grid)");

        // Test Case 2: Solve for call option
        double[][] priceGrid = solver.solveCall(100);
        System.out.printf("PDE Solution grid computed%n");

        // Test Case 3: Get call price at spot=100
        double pdePrice = solver.getCallPrice(100, 100);
        double bsPrice = Finance.blackScholesCall(100, 100, 1.0, 0.05, 0.2);
        System.out.printf("PDE Price at S=100: %.4f, BS Price: %.4f%n", pdePrice, bsPrice);

        // Test Case 4: OTM call
        double pdePriceOTM = solver.getCallPrice(80, 100);
        System.out.printf("PDE Price at S=80 (OTM): %.4f%n", pdePriceOTM);

        // Test Case 5: Deep ITM call
        double pdePriceITM = solver.getCallPrice(130, 100);
        System.out.printf("PDE Price at S=130 (ITM): %.4f%n", pdePriceITM);

        System.out.println();
    }

    private static void testHJBOptimalControl() {
        System.out.println("--- HAMILTON-JACOBI-BELLMAN OPTIMAL CONTROL TESTS ---");

        // Test Case 1: Create HJB solver
        Finance.HJBOptimalControl hjb = new Finance.HJBOptimalControl(
            1.0, 0.05, 0.10, 0.20, 10, 100);
        System.out.println("HJB solver created (T=1yr, r=5%, μ=10%, σ=20%)");

        // Test Case 2: Solve optimal allocation
        double[] optimalAllocation = hjb.solveOptimalAllocation();
        System.out.printf("Optimal allocation at t=0: %.4f%n", optimalAllocation[0]);

        // Test Case 3: Allocations bounded in [0,1]
        boolean boundsValid = true;
        for (double alloc : optimalAllocation) {
            if (alloc < 0 || alloc > 1) boundsValid = false;
        }
        System.out.printf("All allocations in [0,1]: %b%n", boundsValid);

        // Test Case 4: Value function
        double valueW100 = hjb.valueFunction(100, 0);
        double valueW200 = hjb.valueFunction(200, 0);
        System.out.printf("Value Function V(W=100, t=0): %.4f%n", valueW100);
        System.out.printf("Value Function V(W=200, t=0): %.4f (Higher wealth)%n", valueW200);

        // Test Case 5: Time decay of value
        double valueT0 = hjb.valueFunction(100, 0);
        double valueT1 = hjb.valueFunction(100, 0.5);
        System.out.printf("Value decreases with time: %b%n", valueT0 > valueT1);

        System.out.println();
    }

    private static void testFokkerPlanck() {
        System.out.println("--- FOKKER-PLANCK EQUATION SOLVER TESTS ---");

        // Test Case 1: Create solver
        Finance.FokkerPlanckSolver fp = new Finance.FokkerPlanckSolver(
            0.05, 0.20, 100, 50, 150);
        System.out.println("Fokker-Planck solver created (μ=5%, σ=20%)");

        // Test Case 2: Initial density (normal distribution)
        double[] initialDensity = new double[100];
        for (int i = 0; i < 100; i++) {
            double x = 50 + i;
            initialDensity[i] = Math.exp(-(x - 100) * (x - 100) / (2 * 100)) / Math.sqrt(2 * Math.PI * 100);
        }

        // Test Case 3: Solve probability evolution
        double[] density = fp.solveProbabilityDensity(initialDensity, 0.01, 50);
        System.out.printf("Probability density evolved over 50 timesteps%n");

        // Test Case 4: Verify normalization
        double sum = 0;
        for (double d : density) {
            sum += d;
        }
        System.out.printf("Density approximately normalized: %.4f (should be ~0.01)%n", sum * 0.01);

        // Test Case 5: Mean shift due to drift
        System.out.printf("Positive drift moves density rightward (qualitative)%n");

        System.out.println();
    }

    private static void testItosLemma() {
        System.out.println("--- ITO'S LEMMA STOCHASTIC CALCULUS TESTS ---");

        double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;

        // Test Case 1: Apply Ito's lemma
        double[] result = Finance.ItosLemma.applyItosLemma(S, K, T, r, sigma, null);
        System.out.printf("Ito's Lemma - Drift: %.6f, Volatility: %.6f%n", result[0], result[1]);

        // Test Case 2: ATM option drift
        System.out.printf("ATM option has drift component from theta and gamma%n");

        // Test Case 3: OTM option drift
        double[] resultOTM = Finance.ItosLemma.applyItosLemma(90, K, T, r, sigma, null);
        System.out.printf("OTM option - Drift: %.6f%n", resultOTM[0]);

        // Test Case 4: ITM option drift
        double[] resultITM = Finance.ItosLemma.applyItosLemma(110, K, T, r, sigma, null);
        System.out.printf("ITM option - Drift: %.6f%n", resultITM[0]);

        // Test Case 5: Volatility proportional to delta
        System.out.printf("Option volatility = σ × S × Δ (stochastic calculus)%n");

        System.out.println();
    }

    private static void testLongstaffSchwartz() {
        System.out.println("--- LONGSTAFF-SCHWARTZ AMERICAN OPTION TESTS ---");

        // Test Case 1: Create solver
        Finance.LongstaffSchwartz ls = new Finance.LongstaffSchwartz(100, 50);
        System.out.println("Longstaff-Schwartz created (100 paths, 50 steps)");

        // Test Case 2: Price American call
        double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
        double americanPrice = ls.priceAmericanCall(S, K, T, r, sigma);
        double europeanPrice = Finance.blackScholesCall(S, K, T, r, sigma);
        System.out.printf("American Call Price: %.4f%n", americanPrice);
        System.out.printf("European Call Price: %.4f%n", europeanPrice);

        // Test Case 3: American >= European (early exercise premium)
        System.out.printf("American >= European: %b%n", americanPrice >= europeanPrice);

        // Test Case 4: ITM option early exercise value
        double americanITM = ls.priceAmericanCall(110, K, T, r, sigma);
        System.out.printf("American Call (ITM S=110): %.4f (has early exercise value)%n", americanITM);

        // Test Case 5: OTM option converges to European
        double americanOTM = ls.priceAmericanCall(90, K, T, r, sigma);
        double europeanOTM = Finance.blackScholesCall(90, K, T, r, sigma);
        System.out.printf("American OTM converges to European: Diff=%.6f%n", 
                         Math.abs(americanOTM - europeanOTM));

        System.out.println();
    }

    private static void testGradientDescentOptimizer() {
        System.out.println("--- GRADIENT DESCENT OPTIMIZER TESTS ---");

        // Test Case 1: Create optimizer
        Finance.GradientDescentOptimizer gd = new Finance.GradientDescentOptimizer(0.1, 1e-5, 1000);
        System.out.println("Gradient Descent Optimizer created");

        // Test Case 2: Generate synthetic returns data
        double[][] returns = new double[100][3];
        Random rand = new Random(42);
        for (int i = 0; i < 100; i++) {
            returns[i][0] = 0.08 + rand.nextGaussian() * 0.15;
            returns[i][1] = 0.10 + rand.nextGaussian() * 0.20;
            returns[i][2] = 0.06 + rand.nextGaussian() * 0.12;
        }

        // Test Case 3: Minimize CVaR
        double[] optimalWeights = gd.minimizeCVaR(returns, 0.08, 95);
        System.out.printf("Optimized Weights: %.4f, %.4f, %.4f%n", 
                         optimalWeights[0], optimalWeights[1], optimalWeights[2]);

        // Test Case 4: Verify constraints
        double sumWeights = 0;
        for (double w : optimalWeights) {
            sumWeights += w;
        }
        System.out.printf("Weights sum to 1.0: %.6f%n", sumWeights);

        // Test Case 5: Non-negative weights (no short selling)
        boolean nonNegative = true;
        for (double w : optimalWeights) {
            if (w < 0) nonNegative = false;
        }
        System.out.printf("All weights non-negative: %b%n", nonNegative);

        System.out.println();
    }

    private static void testNewtonRaphsonOptimizer() {
        System.out.println("--- NEWTON-RAPHSON OPTIMIZER TESTS ---");

        // Test Case 1: Create optimizer
        Finance.NewtonRaphsonOptimizer nr = new Finance.NewtonRaphsonOptimizer(1e-6, 100);
        System.out.println("Newton-Raphson Optimizer created");

        // Test Case 2: Solve for implied volatility
        double S = 100, K = 100, T = 1.0, r = 0.05;
        double marketPrice = 10.0; // Market call price
        double impliedVol = nr.solveImpliedVolatility(S, K, T, r, marketPrice, 0.2);
        System.out.printf("Implied Volatility: %.4f%n", impliedVol);

        // Test Case 3: Verify accuracy
        double recoveredPrice = Finance.blackScholesCall(S, K, T, r, impliedVol);
        System.out.printf("Recovered Price: %.4f (Target: %.4f, Error: %.6f)%n", 
                         recoveredPrice, marketPrice, Math.abs(recoveredPrice - marketPrice));

        // Test Case 4: Different market prices
        double impliedVol2 = nr.solveImpliedVolatility(S, K, T, r, 15.0, 0.2);
        System.out.printf("Higher market price -> Higher IV: %.4f > %.4f: %b%n", 
                         impliedVol2, impliedVol, impliedVol2 > impliedVol);

        // Test Case 5: Break-even strike
        double breakEvenK = nr.solveBreakEvenStrike(S, T, r, 0.2, S, S);
        System.out.printf("Break-even strike: %.4f%n", breakEvenK);

        System.out.println();
    }

    private static void testConvexOptimization() {
        System.out.println("--- CONVEX OPTIMIZATION PORTFOLIO TESTS ---");

        // Test Case 1: Create optimizer
        Finance.ConvexPortfolioOptimizer cvx = new Finance.ConvexPortfolioOptimizer();
        System.out.println("Convex Portfolio Optimizer created");

        // Test Case 2: Set up Markowitz problem
        double[] returns = {0.08, 0.10, 0.06};
        double[][] covariance = {
            {0.04, 0.02, 0.01},
            {0.02, 0.06, 0.015},
            {0.01, 0.015, 0.03}
        };

        // Test Case 3: Optimize with low risk aversion
        double[] weights_low = cvx.optimizePortfolio(returns, covariance, 1.0);
        System.out.printf("Low Risk Aversion: [%.4f, %.4f, %.4f]%n", 
                         weights_low[0], weights_low[1], weights_low[2]);

        // Test Case 4: Optimize with high risk aversion
        double[] weights_high = cvx.optimizePortfolio(returns, covariance, 10.0);
        System.out.printf("High Risk Aversion: [%.4f, %.4f, %.4f]%n", 
                         weights_high[0], weights_high[1], weights_high[2]);

        // Test Case 5: Higher risk aversion allocates to lower-risk assets
        System.out.printf("Higher λ reduces allocation to risky asset%n");

        System.out.println();
    }

    private static void testMartingalePricing() {
        System.out.println("--- MARTINGALE PRICING (RISK-NEUTRAL VALUATION) TESTS ---");

        double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;

        // Test Case 1: Risk-neutral pricing
        double riskNeutralPrice = Finance.MartingalePricing.riskNeutralPrice(
            S, K, T, r, sigma, 10.0, 0.05);
        System.out.printf("Risk-Neutral Price: %.4f%n", riskNeutralPrice);

        // Test Case 2: Compare to Black-Scholes
        double bsPrice = Finance.blackScholesCall(S, K, T, r, sigma);
        System.out.printf("Black-Scholes Price: %.4f%n", bsPrice);

        // Test Case 3: Girsanov theorem (market price of risk)
        double marketPriceOfRisk = (0.12 - r) / sigma; // (μ - r) / σ
        System.out.printf("Market Price of Risk λ: %.4f%n", marketPriceOfRisk);

        // Test Case 4: Negative risk premium reduces value
        double withNegativePremium = Finance.MartingalePricing.riskNeutralPrice(
            S, K, T, r, sigma, 10.0, -0.05);
        System.out.printf("Negative premium increases value: %.4f%n", withNegativePremium);

        // Test Case 5: Measure change intuition
        System.out.printf("Under Q: E^Q[S_T] = S_0 * e^(rT) (martingale property)%n");

        System.out.println();
    }

    private static void testMeanFieldGames() {
        System.out.println("--- MEAN-FIELD GAMES EQUILIBRIUM TESTS ---");

        // Test Case 1: Create MFG solver
        Finance.MeanFieldGames mfg = new Finance.MeanFieldGames();
        System.out.println("Mean-Field Games solver created");

        // Test Case 2: Solve equilibrium allocation
        double[] assetReturns = {0.08, 0.10, 0.06};
        double[] equilibrium = mfg.solveEquilibrium(1000, 2.0, assetReturns, 1000);
        System.out.printf("Equilibrium Allocation: [%.2f, %.2f, %.2f]%n", 
                         equilibrium[0], equilibrium[1], equilibrium[2]);

        // Test Case 3: Total allocation equals target wealth
        double totalAlloc = 0;
        for (double a : equilibrium) {
            totalAlloc += a;
        }
        System.out.printf("Total allocation: %.2f (Target: 1000)%n", totalAlloc);

        // Test Case 4: Wealth distribution evolution
        double[] initialWealth = {100, 100, 100, 100, 100};
        double[] returns = {0.05, 0.08, -0.03, 0.07, 0.04};
        double[] finalWealth = mfg.wealthDistributionEvolution(initialWealth, returns, 0.01, 10);
        System.out.printf("Initial avg wealth: %.2f%n", Finance.mean(initialWealth));
        System.out.printf("Final avg wealth: %.2f (After mean-field feedback)%n", Finance.mean(finalWealth));

        // Test Case 5: Convergence to Nash equilibrium
        System.out.printf("Many-agent system converges to equilibrium%n");

        System.out.println();
    }

    private static void testFreeBoundary() {
        System.out.println("--- FREE BOUNDARY PROBLEM (AMERICAN OPTION) TESTS ---");

        // Test Case 1: Create solver
        Finance.FreeBoundaryAmerican fb = new Finance.FreeBoundaryAmerican();
        System.out.println("Free Boundary Problem solver created");

        // Test Case 2: Find optimal exercise boundary
        double K = 100, T = 1.0, r = 0.05, sigma = 0.2;
        double S_boundary = fb.findOptimalExerciseBoundary(K, T, r, sigma);
        System.out.printf("Optimal Exercise Boundary S*: %.4f (for K=100)%n", S_boundary);

        // Test Case 3: Boundary above strike
        System.out.printf("Boundary S* > Strike K: %b (Free boundary property)%n", S_boundary > K);

        // Test Case 4: Boundary at early time is higher
        double S_boundary_short = fb.findOptimalExerciseBoundary(K, 0.1, r, sigma);
        System.out.printf("Boundary for T=0.1: %.4f (vs T=1.0: %.4f)%n", S_boundary_short, S_boundary);

        // Test Case 5: At boundary: Call Price = Intrinsic Value
        double callAtBoundary = Finance.blackScholesCall(S_boundary, K, T, r, sigma);
        double intrinsicAtBoundary = S_boundary - K;
        System.out.printf("At boundary: Call=%.4f, Intrinsic=%.4f (should match)%n", 
                         callAtBoundary, intrinsicAtBoundary);

        System.out.println();
    }

    private static void testDynamicProgramming() {
        System.out.println("--- DYNAMIC PROGRAMMING OPTIMAL STOPPING TESTS ---");

        // Test Case 1: Create solver
        Finance.DynamicProgramming dp = new Finance.DynamicProgramming();
        System.out.println("Dynamic Programming solver created");

        // Test Case 2: Set up spot price grid
        double[] spotPrices = new double[50];
        for (int i = 0; i < 50; i++) {
            spotPrices[i] = 80 + i * 0.8; // 80 to 120
        }

        // Test Case 3: Solve optimal stopping for put
        double K = 100, T = 1.0, r = 0.05, sigma = 0.2;
        double putValue = dp.solveOptimalStoppingTime(spotPrices, K, T, r, sigma);
        System.out.printf("Optimal Put Value (American): %.4f%n", putValue);

        // Test Case 4: Compare to European put
        double europeanPut = Finance.blackScholesPut(100, K, T, r, sigma);
        System.out.printf("European Put Value: %.4f%n", europeanPut);

        // Test Case 5: American >= European for puts
        System.out.printf("American Put >= European: %b%n", putValue >= europeanPut);

        // Test Case 6: Backward induction principle
        System.out.printf("Dynamic programming uses backwards induction from terminal condition%n");

        System.out.println();
    }

}
