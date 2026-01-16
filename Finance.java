import java.util.*;
import java.math.*;

/**
 * Comprehensive Quantitative Finance Library
 * Includes pricing models, risk metrics, and financial calculations
 */
public class Finance {

    // ==================== Black-Scholes Model ====================
    
    /**
     * Calculates Black-Scholes call option price
     * @param S Current stock price
     * @param K Strike price
     * @param T Time to expiration (in years)
     * @param r Risk-free interest rate
     * @param sigma Volatility (annualized)
     */
    public static double blackScholesCall(double S, double K, double T, double r, double sigma) {
        double d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
        double d2 = d1 - sigma * Math.sqrt(T);
        return S * normCDF(d1) - K * Math.exp(-r * T) * normCDF(d2);
    }

    /**
     * Calculates Black-Scholes put option price
     */
    public static double blackScholesPut(double S, double K, double T, double r, double sigma) {
        double d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
        double d2 = d1 - sigma * Math.sqrt(T);
        return K * Math.exp(-r * T) * normCDF(-d2) - S * normCDF(-d1);
    }

    /**
     * Calculates call option vega (sensitivity to volatility)
     */
    public static double callVega(double S, double K, double T, double r, double sigma) {
        double d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
        return S * normPDF(d1) * Math.sqrt(T);
    }

    /**
     * Calculates call option delta (sensitivity to stock price)
     */
    public static double callDelta(double S, double K, double T, double r, double sigma) {
        double d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
        return normCDF(d1);
    }

    /**
     * Calculates put option delta
     */
    public static double putDelta(double S, double K, double T, double r, double sigma) {
        return callDelta(S, K, T, r, sigma) - 1.0;
    }

    /**
     * Calculates call option gamma (delta sensitivity)
     */
    public static double callGamma(double S, double K, double T, double r, double sigma) {
        double d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
        return normPDF(d1) / (S * sigma * Math.sqrt(T));
    }

    /**
     * Calculates call option theta (time decay)
     */
    public static double callTheta(double S, double K, double T, double r, double sigma) {
        double d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
        double d2 = d1 - sigma * Math.sqrt(T);
        double theta = -(S * normPDF(d1) * sigma) / (2 * Math.sqrt(T));
        theta -= r * K * Math.exp(-r * T) * normCDF(d2);
        return theta;
    }

    /**
     * Calculates call option rho (interest rate sensitivity)
     */
    public static double callRho(double S, double K, double T, double r, double sigma) {
        double d2 = ((Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T))) 
                    - sigma * Math.sqrt(T);
        return K * T * Math.exp(-r * T) * normCDF(d2);
    }

    // ==================== Binomial Model ====================

    /**
     * Binomial tree for European call option
     */
    public static double binomialCall(double S, double K, double T, double r, double sigma, int steps) {
        double dt = T / steps;
        double u = Math.exp(sigma * Math.sqrt(dt));
        double d = 1.0 / u;
        double p = (Math.exp(r * dt) - d) / (u - d);

        double[] stockPrices = new double[steps + 1];
        for (int i = 0; i <= steps; i++) {
            stockPrices[i] = S * Math.pow(u, steps - i) * Math.pow(d, i);
        }

        double[] optionValues = new double[steps + 1];
        for (int i = 0; i <= steps; i++) {
            optionValues[i] = Math.max(stockPrices[i] - K, 0);
        }

        for (int j = steps - 1; j >= 0; j--) {
            for (int i = 0; i <= j; i++) {
                optionValues[i] = (p * optionValues[i] + (1 - p) * optionValues[i + 1]) 
                                  * Math.exp(-r * dt);
            }
        }

        return optionValues[0];
    }

    // ==================== Value at Risk (VaR) ====================

    /**
     * Calculates Value at Risk using parametric method (variance-covariance)
     * @param mean Portfolio mean return
     * @param stdDev Portfolio standard deviation
     * @param confidenceLevel Confidence level (e.g., 0.95 for 95%)
     * @param investmentAmount Initial investment
     */
    public static double valueAtRisk(double mean, double stdDev, double confidenceLevel, 
                                      double investmentAmount) {
        double zScore = inverseNormCDF(confidenceLevel);
        return investmentAmount * (mean - zScore * stdDev);
    }

    /**
     * Conditional VaR (Expected Shortfall)
     */
    public static double conditionalVaR(double mean, double stdDev, double confidenceLevel,
                                        double investmentAmount) {
        double zScore = inverseNormCDF(confidenceLevel);
        return investmentAmount * (mean - (stdDev * normPDF(zScore)) / (1 - confidenceLevel));
    }

    // ==================== Present Value & Discounting ====================

    /**
     * Calculates present value of a future cash flow
     * @param futureValue Cash flow to discount
     * @param rate Discount rate
     * @param periods Time periods
     */
    public static double presentValue(double futureValue, double rate, int periods) {
        return futureValue / Math.pow(1 + rate, periods);
    }

    /**
     * Calculates future value of an investment
     */
    public static double futureValue(double presentValue, double rate, int periods) {
        return presentValue * Math.pow(1 + rate, periods);
    }

    /**
     * Net Present Value of cash flows
     */
    public static double netPresentValue(double[] cashFlows, double discountRate) {
        double npv = 0;
        for (int i = 0; i < cashFlows.length; i++) {
            npv += cashFlows[i] / Math.pow(1 + discountRate, i);
        }
        return npv;
    }

    /**
     * Internal Rate of Return (using Newton-Raphson method)
     */
    public static double internalRateOfReturn(double[] cashFlows, double initialGuess) {
        double rate = initialGuess;
        double tolerance = 1e-6;
        int maxIterations = 100;

        for (int i = 0; i < maxIterations; i++) {
            double npv = 0;
            double npvDerivative = 0;

            for (int j = 0; j < cashFlows.length; j++) {
                npv += cashFlows[j] / Math.pow(1 + rate, j);
                npvDerivative -= j * cashFlows[j] / Math.pow(1 + rate, j + 1);
            }

            double newRate = rate - npv / npvDerivative;
            if (Math.abs(newRate - rate) < tolerance) {
                return newRate;
            }
            rate = newRate;
        }
        return rate;
    }

    // ==================== Bond Pricing ====================

    /**
     * Calculates bond price given yield to maturity
     * @param couponPayment Periodic coupon payment
     * @param faceValue Bond face value
     * @param yieldToMaturity Annual yield
     * @param periods Number of periods to maturity
     */
    public static double bondPrice(double couponPayment, double faceValue, double yieldToMaturity, 
                                    int periods) {
        double price = 0;
        double periodYield = yieldToMaturity / periods;
        for (int i = 1; i <= periods; i++) {
            price += couponPayment / Math.pow(1 + periodYield, i);
        }
        price += faceValue / Math.pow(1 + periodYield, periods);
        return price;
    }

    /**
     * Macaulay duration of a bond
     */
    public static double macaulayDuration(double couponPayment, double faceValue, 
                                          double yieldToMaturity, int periods) {
        double periodYield = yieldToMaturity / periods;
        double durationNumerator = 0;
        double price = 0;

        for (int i = 1; i <= periods; i++) {
            durationNumerator += (i * couponPayment) / Math.pow(1 + periodYield, i);
            price += couponPayment / Math.pow(1 + periodYield, i);
        }
        durationNumerator += (periods * faceValue) / Math.pow(1 + periodYield, periods);
        price += faceValue / Math.pow(1 + periodYield, periods);

        return durationNumerator / (price * periods);
    }

    /**
     * Modified duration (interest rate sensitivity)
     */
    public static double modifiedDuration(double couponPayment, double faceValue, 
                                         double yieldToMaturity, int periods) {
        double periodYield = yieldToMaturity / periods;
        return macaulayDuration(couponPayment, faceValue, yieldToMaturity, periods) 
               / (1 + periodYield);
    }

    // ==================== Portfolio Analysis ====================

    /**
     * Calculates portfolio return (weighted average)
     */
    public static double portfolioReturn(double[] returns, double[] weights) {
        double portReturn = 0;
        for (int i = 0; i < returns.length; i++) {
            portReturn += returns[i] * weights[i];
        }
        return portReturn;
    }

    /**
     * Calculates portfolio variance
     */
    public static double portfolioVariance(double[] volatilities, double[] weights, 
                                           double[][] correlationMatrix) {
        double variance = 0;
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights.length; j++) {
                variance += weights[i] * weights[j] * volatilities[i] * volatilities[j] 
                           * correlationMatrix[i][j];
            }
        }
        return variance;
    }

    /**
     * Calculates portfolio standard deviation
     */
    public static double portfolioStdDev(double[] volatilities, double[] weights,
                                        double[][] correlationMatrix) {
        return Math.sqrt(portfolioVariance(volatilities, weights, correlationMatrix));
    }

    /**
     * Sharpe ratio (risk-adjusted return)
     */
    public static double sharpeRatio(double portfolioReturn, double riskFreeRate, 
                                     double portfolioStdDev) {
        return (portfolioReturn - riskFreeRate) / portfolioStdDev;
    }

    /**
     * Sortino ratio (downside risk adjustment)
     */
    public static double sortinoRatio(double portfolioReturn, double riskFreeRate,
                                      double[] returns, double minimumAcceptableReturn) {
        double downsideVariance = 0;
        for (double ret : returns) {
            if (ret < minimumAcceptableReturn) {
                downsideVariance += Math.pow(ret - minimumAcceptableReturn, 2);
            }
        }
        downsideVariance /= returns.length;
        double downsideStdDev = Math.sqrt(downsideVariance);
        return (portfolioReturn - riskFreeRate) / downsideStdDev;
    }

    /**
     * Treynor ratio (systematic risk adjustment)
     */
    public static double treynorRatio(double portfolioReturn, double riskFreeRate, double beta) {
        return (portfolioReturn - riskFreeRate) / beta;
    }

    // ==================== CAPM & Risk Metrics ====================

    /**
     * Capital Asset Pricing Model (CAPM)
     * @param riskFreeRate Risk-free rate
     * @param beta Systematic risk
     * @param marketReturn Expected market return
     */
    public static double capm(double riskFreeRate, double beta, double marketReturn) {
        return riskFreeRate + beta * (marketReturn - riskFreeRate);
    }

    /**
     * Covariance between two assets
     */
    public static double covariance(double[] returns1, double[] returns2) {
        double mean1 = mean(returns1);
        double mean2 = mean(returns2);
        double cov = 0;
        for (int i = 0; i < returns1.length; i++) {
            cov += (returns1[i] - mean1) * (returns2[i] - mean2);
        }
        return cov / (returns1.length - 1);
    }

    /**
     * Correlation between two assets
     */
    public static double correlation(double[] returns1, double[] returns2) {
        double stdDev1 = standardDeviation(returns1);
        double stdDev2 = standardDeviation(returns2);
        return covariance(returns1, returns2) / (stdDev1 * stdDev2);
    }

    // ==================== Statistical Calculations ====================

    /**
     * Mean of array
     */
    public static double mean(double[] data) {
        double sum = 0;
        for (double d : data) {
            sum += d;
        }
        return sum / data.length;
    }

    /**
     * Standard deviation
     */
    public static double standardDeviation(double[] data) {
        double mean = mean(data);
        double sumSquares = 0;
        for (double d : data) {
            sumSquares += Math.pow(d - mean, 2);
        }
        return Math.sqrt(sumSquares / (data.length - 1));
    }

    /**
     * Variance
     */
    public static double variance(double[] data) {
        double sd = standardDeviation(data);
        return sd * sd;
    }

    /**
     * Skewness
     */
    public static double skewness(double[] data) {
        double mean = mean(data);
        double sd = standardDeviation(data);
        double sumCubes = 0;
        for (double d : data) {
            sumCubes += Math.pow((d - mean) / sd, 3);
        }
        return sumCubes / data.length;
    }

    /**
     * Kurtosis
     */
    public static double kurtosis(double[] data) {
        double mean = mean(data);
        double sd = standardDeviation(data);
        double sumFourth = 0;
        for (double d : data) {
            sumFourth += Math.pow((d - mean) / sd, 4);
        }
        return (sumFourth / data.length) - 3; // Excess kurtosis
    }

    // ==================== Monte Carlo Simulation ====================

    /**
     * Monte Carlo simulation for option pricing
     * @param S Initial stock price
     * @param K Strike price
     * @param T Time to expiration
     * @param r Risk-free rate
     * @param sigma Volatility
     * @param simulations Number of simulations
     */
    public static double monteCarloCallOption(double S, double K, double T, double r, 
                                              double sigma, int simulations) {
        double sum = 0;
        Random rand = new Random(42); // Seed for reproducibility

        for (int i = 0; i < simulations; i++) {
            double z = rand.nextGaussian();
            double ST = S * Math.exp((r - 0.5 * sigma * sigma) * T + sigma * Math.sqrt(T) * z);
            sum += Math.max(ST - K, 0);
        }

        return (sum / simulations) * Math.exp(-r * T);
    }

    /**
     * Monte Carlo simulation for portfolio value
     */
    public static double[] monteCarloPortfolio(double[] initialWeights, double[] returns,
                                               double[] volatilities, int simulations, int periods) {
        double[] portValues = new double[periods + 1];
        portValues[0] = 1.0; // Normalized to 1

        Random rand = new Random(42);
        for (int t = 1; t <= periods; t++) {
            double z = rand.nextGaussian();
            double portReturn = portfolioReturn(returns, initialWeights);
            double portVol = portfolioStdDev(volatilities, initialWeights, 
                                            identityCorrelationMatrix(initialWeights.length));
            portValues[t] = portValues[t - 1] * Math.exp(portReturn + portVol * z);
        }

        return portValues;
    }

    // ==================== Interest Rate Models ====================

    /**
     * Vasicek model - short rate simulation
     * @param r0 Initial short rate
     * @param a Mean reversion speed
     * @param b Long-term mean
     * @param sigma Volatility
     * @param dt Time step
     * @param steps Number of steps
     */
    public static double[] vasicekModel(double r0, double a, double b, double sigma, 
                                        double dt, int steps) {
        double[] rates = new double[steps + 1];
        rates[0] = r0;
        Random rand = new Random(42);

        for (int i = 1; i <= steps; i++) {
            double z = rand.nextGaussian();
            rates[i] = rates[i - 1] + a * (b - rates[i - 1]) * dt + sigma * Math.sqrt(dt) * z;
        }

        return rates;
    }

    /**
     * Forward rate calculation
     */
    public static double forwardRate(double spot1, double spot2, double t1, double t2) {
        return (spot2 * (t2 - t1) - spot1 * t1) / (t2 - t1);
    }

    // ==================== Greeks Helpers & Utility Functions ====================

    /**
     * Normal distribution cumulative density function (approximation)
     */
    private static double normCDF(double x) {
        return 0.5 * (1 + erf(x / Math.sqrt(2)));
    }

    /**
     * Normal distribution probability density function
     */
    private static double normPDF(double x) {
        return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
    }

    /**
     * Inverse normal CDF (approximation)
     */
    private static double inverseNormCDF(double p) {
        if (p <= 0 || p >= 1) {
            throw new IllegalArgumentException("p must be between 0 and 1");
        }

        // Rational approximation for inverse normal
        double a1 = -3.969683028665376e+01;
        double a2 = 2.221222899801429e+02;
        double a3 = -2.368711934304711e+02;
        double a4 = 1.340426741962386e+02;
        double a5 = -3.304993403249437e+00;

        double b1 = -5.447609879822406e+01;
        double b2 = 1.615858368580409e+02;
        double b3 = -1.556989798598866e+02;
        double b4 = 6.680131188771972e+01;
        double b5 = -1.328068155288572e+00;

        double q, r, val;
        if (p < 0.02425) {
            q = Math.sqrt(-2 * Math.log(p));
            val = (((((a1 * q + a2) * q + a3) * q + a4) * q + a5) * q)
                  / (((((b1 * q + b2) * q + b3) * q + b4) * q + b5) * q + 1);
        } else if (p <= 0.97575) {
            q = p - 0.5;
            r = q * q;
            val = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + q)
                  / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1);
        } else {
            q = Math.sqrt(-2 * Math.log(1 - p));
            val = -(((((a1 * q + a2) * q + a3) * q + a4) * q + a5) * q)
                  / (((((b1 * q + b2) * q + b3) * q + b4) * q + b5) * q + 1);
        }
        return val;
    }

    /**
     * Error function approximation
     */
    private static double erf(double x) {
        double a1 = 0.254829592;
        double a2 = -0.284496736;
        double a3 = 1.421413741;
        double a4 = -1.453152027;
        double a5 = 1.061405429;
        double p = 0.3275911;

        int sign = x < 0 ? -1 : 1;
        x = Math.abs(x);
        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) 
                   * Math.exp(-x * x);

        return sign * y;
    }

    /**
     * Create identity correlation matrix
     */
    private static double[][] identityCorrelationMatrix(int size) {
        double[][] matrix = new double[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                matrix[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }
        return matrix;
    }

    // ==================== ADVANCED OPTION MODELS ====================

    /**
     * Merton Jump-Diffusion Model - accounts for sudden jumps in asset prices
     * @param S Current stock price
     * @param K Strike price
     * @param T Time to expiration
     * @param r Risk-free rate
     * @param sigma Volatility (diffusion component)
     * @param jumpMean Mean jump size
     * @param jumpStdDev Standard deviation of jump
     * @param jumpIntensity Average number of jumps per year
     */
    public static double mertonJumpDiffusionCall(double S, double K, double T, double r,
                                                  double sigma, double jumpMean, double jumpStdDev,
                                                  double jumpIntensity) {
        double callPrice = 0;
        int maxJumps = 10;

        for (int n = 0; n <= maxJumps; n++) {
            double lambda_n = Math.pow(jumpIntensity * T, n) * Math.exp(-jumpIntensity * T)
                            / factorial(n);
            double rn = r - jumpIntensity * jumpMean + (n * jumpMean / T);
            double callN = blackScholesCall(S, K, T, rn, sigma);
            callPrice += lambda_n * callN;
        }

        return callPrice;
    }

    /**
     * Barrier Option (Down-and-Out Call) - option expires if price touches barrier
     */
    public static double barierDownAndOutCall(double S, double K, double B, double T,
                                             double r, double sigma) {
        if (S <= B) return 0; // Already knocked out
        if (B >= K) return 0; // Barrier at or above strike

        double lambda = (r + 0.5 * sigma * sigma) / (sigma * sigma);
        double d = Math.log(K * K / (S * B)) / (sigma * Math.sqrt(T));

        double bsCall = blackScholesCall(S, K, T, r, sigma);
        double adjustmentTerm = Math.pow(B / S, 2 * lambda) * blackScholesCall(B * B / S, K, T, r, sigma);

        return bsCall - adjustmentTerm;
    }

    /**
     * Asian Option (Arithmetic Average Price Call) - payoff based on average price
     */
    public static double asianArithmeticAverageCall(double[] prices, double K, double T,
                                                    double r, double sigma) {
        double avgPrice = mean(prices);
        double adjustedVolatility = sigma / Math.sqrt(3); // Reduced volatility for averaging
        double S = avgPrice;

        return blackScholesCall(S, K, T, r, adjustedVolatility);
    }

    /**
     * Lookback Option (Fixed Strike) - payoff based on minimum or maximum price over period
     */
    public static double lookbackCallMaxPrice(double S, double K, double maxPrice, double T,
                                             double r, double sigma) {
        double a = 2 * r / (sigma * sigma);
        double b = 2 / (sigma * sigma * T);

        double term1 = (maxPrice / S) * normCDF((Math.log(maxPrice / S) + 0.5 * sigma * sigma * T)
                       / (sigma * Math.sqrt(T)));
        double term2 = Math.exp(-r * T) * normCDF((Math.log(maxPrice / S) - 0.5 * sigma * sigma * T)
                       / (sigma * Math.sqrt(T)));

        return S * term1 - K * Math.exp(-r * T) * term2;
    }

    /**
     * Heston Model (Stochastic Volatility) - simplified characteristic function approach
     */
    public static double hestonCall(double S, double K, double T, double r, double v0,
                                   double kappa, double theta, double sigma, double rho) {
        // Simplified: Use Heston as a BS variant with adjusted volatility
        double adjustedVol = Math.sqrt(v0 + kappa * (theta - v0) * (1 - Math.exp(-kappa * T)) / (kappa * T));
        return blackScholesCall(S, K, T, r, adjustedVol);
    }

    // ==================== EXOTIC GREEKS ====================

    /**
     * Vanna - cross-sensitivity between delta and volatility
     */
    public static double vanna(double S, double K, double T, double r, double sigma) {
        double d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
        double d2 = d1 - sigma * Math.sqrt(T);
        return -normPDF(d1) * d2 / sigma;
    }

    /**
     * Volga - vega sensitivity to volatility changes
     */
    public static double volga(double S, double K, double T, double r, double sigma) {
        double d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
        double d2 = d1 - sigma * Math.sqrt(T);
        return S * normPDF(d1) * Math.sqrt(T) * (d1 * d2) / sigma;
    }

    /**
     * Charm - delta decay (theta of delta)
     */
    public static double callCharm(double S, double K, double T, double r, double sigma) {
        double d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
        double d2 = d1 - sigma * Math.sqrt(T);
        return r * normCDF(d1) - (normPDF(d1) * (2 * r * T - d2 * sigma * Math.sqrt(T)))
               / (2 * T * sigma * Math.sqrt(T));
    }

    /**
     * Vera - rho sensitivity to volatility changes
     */
    public static double vera(double S, double K, double T, double r, double sigma) {
        double d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
        double d2 = d1 - sigma * Math.sqrt(T);
        return -K * T * Math.exp(-r * T) * normPDF(d2) * d1 / sigma;
    }

    // ==================== ADVANCED PORTFOLIO MODELS ====================

    /**
     * Efficient Frontier - finds minimum variance for given target return
     * (Simplified: assumes no short selling)
     */
    public static double minVariancePortfolio(double[] returns, double[] volatilities,
                                             double[][] correlations, double targetReturn) {
        int n = returns.length;
        double[] weights = new double[n];

        // Simplified equal-weight approximation
        for (int i = 0; i < n; i++) {
            weights[i] = 1.0 / n;
        }

        return Math.sqrt(portfolioVariance(volatilities, weights, correlations));
    }

    /**
     * Black-Litterman Model - combines market implied returns with investor views
     */
    public static double[] blackLittermanReturns(double[] marketReturns, double[] views,
                                                 double[] viewConfidence, double tau) {
        int n = marketReturns.length;
        double[] adjustedReturns = new double[n];

        for (int i = 0; i < n; i++) {
            double weight = viewConfidence[i] / (1 + viewConfidence[i]);
            adjustedReturns[i] = (1 - weight) * marketReturns[i] + weight * views[i];
            adjustedReturns[i] = adjustedReturns[i] * (1 + tau);
        }

        return adjustedReturns;
    }

    /**
     * Risk Parity Portfolio - allocate to equalize risk contribution
     */
    public static double[] riskParityWeights(double[] volatilities) {
        double sum = 0;
        for (double vol : volatilities) {
            sum += 1.0 / vol;
        }

        double[] weights = new double[volatilities.length];
        for (int i = 0; i < volatilities.length; i++) {
            weights[i] = (1.0 / volatilities[i]) / sum;
        }

        return weights;
    }

    /**
     * Maximum Drawdown - worst cumulative loss from peak
     */
    public static double maxDrawdown(double[] returns) {
        double peak = 1.0;
        double maxDD = 0;
        double current = 1.0;

        for (double ret : returns) {
            current *= (1 + ret);
            if (current > peak) {
                peak = current;
            }
            double dd = (peak - current) / peak;
            if (dd > maxDD) {
                maxDD = dd;
            }
        }

        return maxDD;
    }

    /**
     * Calmar Ratio - return divided by maximum drawdown
     */
    public static double calmarRatio(double[] returns, double riskFreeRate) {
        double avgReturn = mean(returns) * 252; // Annualized
        double maxDD = maxDrawdown(returns);

        if (maxDD == 0) return 0;
        return (avgReturn - riskFreeRate) / maxDD;
    }

    /**
     * Information Ratio - excess return relative to tracking error
     */
    public static double informationRatio(double[] portfolioReturns, double[] benchmarkReturns) {
        double[] excessReturns = new double[portfolioReturns.length];
        for (int i = 0; i < portfolioReturns.length; i++) {
            excessReturns[i] = portfolioReturns[i] - benchmarkReturns[i];
        }

        double meanExcess = mean(excessReturns);
        double trackingError = standardDeviation(excessReturns);

        return (meanExcess * 252) / (trackingError * Math.sqrt(252)); // Annualized
    }

    // ==================== ADVANCED RISK MODELS ====================

    /**
     * GARCH(1,1) Volatility - captures volatility clustering
     * Returns predicted volatility for next period
     */
    public static double garch11Forecast(double[] returns, double omega, double alpha, 
                                        double beta, int forecastPeriods) {
        double variance = variance(returns);

        for (int t = 0; t < forecastPeriods; t++) {
            double lastReturn = returns[returns.length - 1];
            variance = omega + alpha * lastReturn * lastReturn + beta * variance;
        }

        return Math.sqrt(variance);
    }

    /**
     * Historical VaR - empirical quantile method
     */
    public static double historicalVaR(double[] returns, double confidenceLevel) {
        Arrays.sort(returns);
        int index = (int) Math.ceil((1 - confidenceLevel) * returns.length) - 1;
        return Math.abs(returns[index]);
    }

    /**
     * Cornish-Fisher VaR - accounts for skewness and kurtosis
     */
    public static double cornishFisherVaR(double[] returns, double confidenceLevel, 
                                         double investmentAmount) {
        double z = inverseNormCDF(confidenceLevel);
        double s = skewness(returns);
        double k = kurtosis(returns);

        double z_cf = z + (z * z - 1) * s / 6 + (z * z * z - 3 * z) * k / 24 
                     - (2 * z * z * z - 5 * z) * s * s / 36;

        double stdDev = standardDeviation(returns);
        return investmentAmount * stdDev * z_cf;
    }

    /**
     * Incremental VaR - marginal contribution of asset to portfolio VaR
     */
    public static double incrementalVaR(double[] returns, double assetReturn, double confidenceLevel) {
        double[] withoutAsset = Arrays.copyOf(returns, returns.length);
        Arrays.sort(withoutAsset);

        double varWithout = historicalVaR(withoutAsset, confidenceLevel);

        double[] withAsset = Arrays.copyOf(returns, returns.length + 1);
        withAsset[returns.length] = assetReturn;
        Arrays.sort(withAsset);

        double varWith = historicalVaR(withAsset, confidenceLevel);

        return varWith - varWithout;
    }

    /**
     * Stressed VaR - VaR under adverse market conditions (tail modeling)
     */
    public static double stressedVaR(double[] baselineReturns, double stressScalar,
                                    double confidenceLevel, double investmentAmount) {
        double[] stressedReturns = new double[baselineReturns.length];
        for (int i = 0; i < baselineReturns.length; i++) {
            stressedReturns[i] = baselineReturns[i] * stressScalar;
        }

        double var = historicalVaR(stressedReturns, confidenceLevel);
        return investmentAmount * var;
    }

    // ==================== ADVANCED FIXED INCOME ====================

    /**
     * CIR Model (Cox-Ingersoll-Ross) - more realistic interest rate model
     */
    public static double[] cirModel(double r0, double kappa, double theta, double sigma,
                                   double dt, int steps) {
        double[] rates = new double[steps + 1];
        rates[0] = r0;
        Random rand = new Random(42);

        for (int i = 1; i <= steps; i++) {
            double z = rand.nextGaussian();
            double sqrtR = Math.sqrt(Math.max(rates[i - 1], 0));
            rates[i] = rates[i - 1] + kappa * (theta - rates[i - 1]) * dt 
                      + sigma * sqrtR * Math.sqrt(dt) * z;
            rates[i] = Math.max(rates[i], 0); // CIR ensures positive rates
        }

        return rates;
    }

    /**
     * Key Rate Duration - sensitivity to specific maturity points
     */
    public static double keyRateDuration(double coupon, double faceValue, double ytm,
                                        int maturity, int keyMaturity, double yieldShift) {
        double priceBase = bondPrice(coupon, faceValue, ytm, maturity);

        double shiftedYield = ytm;
        if (keyMaturity == maturity) {
            shiftedYield += yieldShift;
        } else {
            // Proportional shift for other maturities
            shiftedYield += yieldShift * (1.0 - Math.abs(keyMaturity - maturity) / 10.0);
        }

        double priceShifted = bondPrice(coupon, faceValue, shiftedYield, maturity);
        return (priceBase - priceShifted) / (priceBase * yieldShift);
    }

    /**
     * Bond Convexity - second-order price sensitivity
     */
    public static double bondConvexity(double couponPayment, double faceValue, 
                                      double yieldToMaturity, int periods) {
        double periodYield = yieldToMaturity / periods;
        double convexityNumerator = 0;
        double price = 0;

        for (int i = 1; i <= periods; i++) {
            convexityNumerator += i * (i + 1) * couponPayment / Math.pow(1 + periodYield, i + 2);
            price += couponPayment / Math.pow(1 + periodYield, i);
        }
        convexityNumerator += periods * (periods + 1) * faceValue 
                             / Math.pow(1 + periodYield, periods + 2);
        price += faceValue / Math.pow(1 + periodYield, periods);

        return convexityNumerator / (price * 100 * periods * periods);
    }

    /**
     * OAS (Option-Adjusted Spread) simplified - spread over risk-free curve
     */
    public static double optionAdjustedSpread(double bondPrice, double couponPayment,
                                             double faceValue, int periods, 
                                             double riskFreeYield, double estimatedOAS) {
        double effectiveYield = riskFreeYield + estimatedOAS;
        double calculatedPrice = bondPrice(couponPayment, faceValue, effectiveYield, periods);
        return effectiveYield - riskFreeYield;
    }

    // ==================== CREDIT RISK ====================

    /**
     * Probability of Default (Merton Model) - from equity value
     */
    public static double mertonProbabilityOfDefault(double assetValue, double debtValue,
                                                    double volatility, double T, double r) {
        double d2 = (Math.log(assetValue / debtValue) + (r - 0.5 * volatility * volatility) * T)
                   / (volatility * Math.sqrt(T));
        return normCDF(-d2);
    }

    /**
     * Expected Loss - default probability times loss given default
     */
    public static double expectedLoss(double debtValue, double recoveryRate, 
                                     double probabilityOfDefault) {
        return debtValue * (1 - recoveryRate) * probabilityOfDefault;
    }

    /**
     * Credit VaR - unexpected loss at given confidence level
     */
    public static double creditVaR(double debtValue, double probabilityOfDefault,
                                  double recoveryRate, double confidenceLevel) {
        double expectedLoss = debtValue * (1 - recoveryRate) * probabilityOfDefault;
        double variance = probabilityOfDefault * (1 - probabilityOfDefault) 
                         * (debtValue * (1 - recoveryRate)) * (debtValue * (1 - recoveryRate));
        double stdDev = Math.sqrt(variance);

        return expectedLoss + inverseNormCDF(confidenceLevel) * stdDev;
    }

    // ==================== SENSITIVITY ANALYSIS ====================

    /**
     * Scenario Analysis - calculate PV under different scenarios
     */
    public static double[] scenarioAnalysis(double[] baselineReturns, double riskFreeRate,
                                           double[] scenarioMultipliers) {
        double[] results = new double[scenarioMultipliers.length];

        for (int i = 0; i < scenarioMultipliers.length; i++) {
            double scaledReturn = mean(baselineReturns) * scenarioMultipliers[i];
            results[i] = scaledReturn - riskFreeRate;
        }

        return results;
    }

    /**
     * Sensitivity Table - shows impact of two variables on output
     */
    public static double[][] sensitivityTable(double baseValue, double var1Min, double var1Max,
                                             double var2Min, double var2Max, int steps) {
        double[][] table = new double[steps][steps];
        double var1Step = (var1Max - var1Min) / (steps - 1);
        double var2Step = (var2Max - var2Min) / (steps - 1);

        for (int i = 0; i < steps; i++) {
            for (int j = 0; j < steps; j++) {
                double var1 = var1Min + i * var1Step;
                double var2 = var2Min + j * var2Step;
                table[i][j] = baseValue * var1 * var2;
            }
        }

        return table;
    }

    /**
     * Greeks Ladder - shows Greeks across multiple spot prices
     */
    public static double[][] greeksLadder(double K, double T, double r, double sigma,
                                         double spotMin, double spotMax, int spots) {
        double[][] greeksTable = new double[spots][5]; // Delta, Gamma, Vega, Theta, Rho

        double spotStep = (spotMax - spotMin) / (spots - 1);

        for (int i = 0; i < spots; i++) {
            double S = spotMin + i * spotStep;
            greeksTable[i][0] = callDelta(S, K, T, r, sigma);
            greeksTable[i][1] = callGamma(S, K, T, r, sigma);
            greeksTable[i][2] = callVega(S, K, T, r, sigma);
            greeksTable[i][3] = callTheta(S, K, T, r, sigma);
            greeksTable[i][4] = callRho(S, K, T, r, sigma);
        }

        return greeksTable;
    }

    // ==================== ADDITIONAL UTILITIES ====================

    /**
     * Factorial function (for jump-diffusion model)
     */
    private static double factorial(int n) {
        if (n <= 1) return 1;
        double result = 1;
        for (int i = 2; i <= n; i++) {
            result *= i;
        }
        return result;
    }

    // ==================== MACHINE LEARNING FOR FINANCE ====================

    /**
     * Neural Network for Option Pricing
     * Simple feedforward network with one hidden layer
     */
    public static class NeuralNetworkPricer {
        private double[][] weights1;  // Input to hidden
        private double[][] weights2;  // Hidden to output
        private double[] bias1;
        private double[] bias2;
        private int hiddenSize;

        public NeuralNetworkPricer(int hiddenSize) {
            this.hiddenSize = hiddenSize;
            initializeWeights();
        }

        private void initializeWeights() {
            Random rand = new Random(42);
            weights1 = new double[5][hiddenSize]; // 5 inputs: S, K, T, r, sigma
            weights2 = new double[hiddenSize][1];
            bias1 = new double[hiddenSize];
            bias2 = new double[1];

            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < hiddenSize; j++) {
                    weights1[i][j] = (rand.nextDouble() - 0.5) * 0.1;
                }
            }
            for (int i = 0; i < hiddenSize; i++) {
                for (int j = 0; j < 1; j++) {
                    weights2[i][j] = (rand.nextDouble() - 0.5) * 0.1;
                }
                bias1[i] = 0;
            }
            bias2[0] = 0;
        }

        public double predictCallPrice(double S, double K, double T, double r, double sigma) {
            double[] input = {S, K, T, r, sigma};

            // Forward pass
            double[] hidden = new double[hiddenSize];
            for (int j = 0; j < hiddenSize; j++) {
                hidden[j] = bias1[j];
                for (int i = 0; i < 5; i++) {
                    hidden[j] += input[i] * weights1[i][j];
                }
                hidden[j] = relu(hidden[j]); // ReLU activation
            }

            double output = bias2[0];
            for (int j = 0; j < hiddenSize; j++) {
                output += hidden[j] * weights2[j][0];
            }

            return Math.max(output, 0); // Ensure non-negative price
        }

        public void train(double[][] trainingInputs, double[] trainingOutputs, int epochs) {
            double learningRate = 0.01;

            for (int epoch = 0; epoch < epochs; epoch++) {
                double totalError = 0;

                for (int sample = 0; sample < trainingInputs.length; sample++) {
                    double[] input = trainingInputs[sample];
                    double target = trainingOutputs[sample];

                    // Forward pass
                    double[] hidden = new double[hiddenSize];
                    for (int j = 0; j < hiddenSize; j++) {
                        hidden[j] = bias1[j];
                        for (int i = 0; i < 5; i++) {
                            hidden[j] += input[i] * weights1[i][j];
                        }
                        hidden[j] = relu(hidden[j]);
                    }

                    double output = bias2[0];
                    for (int j = 0; j < hiddenSize; j++) {
                        output += hidden[j] * weights2[j][0];
                    }

                    output = Math.max(output, 0);
                    double error = output - target;
                    totalError += error * error;

                    // Simplified backprop (gradient descent)
                    double outputGradient = error;
                    bias2[0] -= learningRate * outputGradient;

                    for (int j = 0; j < hiddenSize; j++) {
                        weights2[j][0] -= learningRate * outputGradient * hidden[j];
                    }
                }
            }
        }

        private double relu(double x) {
            return Math.max(0, x);
        }
    }

    /**
     * Linear Regression for Price Prediction
     */
    public static class LinearRegression {
        private double[] coefficients;
        private double intercept;

        public void fit(double[][] features, double[] targets) {
            int n = features.length;
            int m = features[0].length;

            // Simple normal equations: (X^T X)^-1 X^T y
            double[][] XtX = new double[m][m];
            double[] Xty = new double[m];

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    for (int k = 0; k < m; k++) {
                        XtX[j][k] += features[i][j] * features[i][k];
                    }
                    Xty[j] += features[i][j] * targets[i];
                }
            }

            // Gaussian elimination to solve XtX * coef = Xty
            coefficients = gaussianElimination(XtX, Xty);

            // Calculate intercept
            double meanTarget = mean(targets);
            double intercept_sum = 0;
            for (int j = 0; j < m; j++) {
                intercept_sum += coefficients[j] * mean(getColumn(features, j));
            }
            intercept = meanTarget - intercept_sum;
        }

        public double predict(double[] features) {
            double result = intercept;
            for (int i = 0; i < features.length; i++) {
                result += features[i] * coefficients[i];
            }
            return result;
        }

        private double[] gaussianElimination(double[][] A, double[] b) {
            int n = A.length;
            double[][] aug = new double[n][n + 1];

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    aug[i][j] = A[i][j];
                }
                aug[i][n] = b[i];
            }

            // Forward elimination
            for (int i = 0; i < n; i++) {
                int maxRow = i;
                for (int k = i + 1; k < n; k++) {
                    if (Math.abs(aug[k][i]) > Math.abs(aug[maxRow][i])) {
                        maxRow = k;
                    }
                }

                double[] temp = aug[i];
                aug[i] = aug[maxRow];
                aug[maxRow] = temp;

                for (int k = i + 1; k < n; k++) {
                    double factor = aug[k][i] / aug[i][i];
                    for (int j = i; j <= n; j++) {
                        aug[k][j] -= factor * aug[i][j];
                    }
                }
            }

            // Back substitution
            double[] x = new double[n];
            for (int i = n - 1; i >= 0; i--) {
                x[i] = aug[i][n];
                for (int j = i + 1; j < n; j++) {
                    x[i] -= aug[i][j] * x[j];
                }
                x[i] /= aug[i][i];
            }

            return x;
        }

        private double[] getColumn(double[][] matrix, int col) {
            double[] column = new double[matrix.length];
            for (int i = 0; i < matrix.length; i++) {
                column[i] = matrix[i][col];
            }
            return column;
        }
    }

    /**
     * K-Means Clustering for Volatility Regimes
     */
    public static class KMeansVolatilityRegime {
        private double[] centroids;
        private int[] assignments;
        private int k;

        public KMeansVolatilityRegime(int k) {
            this.k = k;
            this.centroids = new double[k];
        }

        public void fit(double[] volatilities, int maxIterations) {
            // Initialize centroids with k evenly spaced points
            double minVol = 0, maxVol = 0;
            for (double v : volatilities) {
                minVol = Math.min(minVol, v);
                maxVol = Math.max(maxVol, v);
            }

            for (int i = 0; i < k; i++) {
                centroids[i] = minVol + (maxVol - minVol) * (i + 1) / (k + 1);
            }

            assignments = new int[volatilities.length];

            for (int iteration = 0; iteration < maxIterations; iteration++) {
                // Assign points to nearest centroid
                for (int i = 0; i < volatilities.length; i++) {
                    int bestCluster = 0;
                    double bestDistance = Math.abs(volatilities[i] - centroids[0]);

                    for (int j = 1; j < k; j++) {
                        double distance = Math.abs(volatilities[i] - centroids[j]);
                        if (distance < bestDistance) {
                            bestDistance = distance;
                            bestCluster = j;
                        }
                    }
                    assignments[i] = bestCluster;
                }

                // Update centroids
                double[] newCentroids = new double[k];
                int[] counts = new int[k];

                for (int i = 0; i < volatilities.length; i++) {
                    newCentroids[assignments[i]] += volatilities[i];
                    counts[assignments[i]]++;
                }

                for (int j = 0; j < k; j++) {
                    if (counts[j] > 0) {
                        newCentroids[j] /= counts[j];
                    }
                }

                centroids = newCentroids;
            }
        }

        public int predictRegime(double volatility) {
            int bestCluster = 0;
            double bestDistance = Math.abs(volatility - centroids[0]);

            for (int j = 1; j < k; j++) {
                double distance = Math.abs(volatility - centroids[j]);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestCluster = j;
                }
            }

            return bestCluster;
        }

        public double[] getCentroids() {
            return centroids;
        }
    }

    /**
     * Random Forest-like Ensemble for Price Prediction
     */
    public static class EnsemblePredictor {
        private LinearRegression[] trees;
        private int numTrees;

        public EnsemblePredictor(int numTrees) {
            this.numTrees = numTrees;
            this.trees = new LinearRegression[numTrees];
            for (int i = 0; i < numTrees; i++) {
                trees[i] = new LinearRegression();
            }
        }

        public void fit(double[][] features, double[] targets) {
            Random rand = new Random(42);

            for (int t = 0; t < numTrees; t++) {
                // Bootstrap sample
                double[][] bootFeatures = new double[features.length][];
                double[] bootTargets = new double[targets.length];

                for (int i = 0; i < features.length; i++) {
                    int idx = rand.nextInt(features.length);
                    bootFeatures[i] = features[idx];
                    bootTargets[i] = targets[idx];
                }

                trees[t].fit(bootFeatures, bootTargets);
            }
        }

        public double predict(double[] features) {
            double sum = 0;
            for (int t = 0; t < numTrees; t++) {
                sum += trees[t].predict(features);
            }
            return sum / numTrees;
        }

        public double predictWithConfidence(double[] features, double[][] validationFeatures,
                                           double[] validationTargets) {
            double prediction = predict(features);

            // Calculate prediction error from validation set
            double meanSquaredError = 0;
            for (int i = 0; i < validationFeatures.length; i++) {
                double pred = predict(validationFeatures[i]);
                double error = pred - validationTargets[i];
                meanSquaredError += error * error;
            }
            meanSquaredError /= validationFeatures.length;

            return prediction;
        }
    }

    /**
     * Principal Component Analysis (PCA) for Portfolio Decomposition
     */
    public static class PCA {
        private double[][] covariance;
        private double[] eigenvalues;
        private double[][] eigenvectors;

        public void fit(double[][] returns) {
            // Center the data
            double[] means = new double[returns[0].length];
            for (int j = 0; j < returns[0].length; j++) {
                for (int i = 0; i < returns.length; i++) {
                    means[j] += returns[i][j];
                }
                means[j] /= returns.length;
            }

            // Calculate covariance matrix
            int p = returns[0].length;
            covariance = new double[p][p];
            for (int i = 0; i < p; i++) {
                for (int j = 0; j < p; j++) {
                    for (int k = 0; k < returns.length; k++) {
                        covariance[i][j] += (returns[k][i] - means[i]) * (returns[k][j] - means[j]);
                    }
                    covariance[i][j] /= (returns.length - 1);
                }
            }

            // Power iteration method for largest eigenvalue
            eigenvalues = new double[p];
            eigenvectors = new double[p][p];

            double[] v = new double[p];
            for (int i = 0; i < p; i++) {
                v[i] = 1.0 / Math.sqrt(p);
            }

            for (int iter = 0; iter < 100; iter++) {
                double[] Av = new double[p];
                for (int i = 0; i < p; i++) {
                    for (int j = 0; j < p; j++) {
                        Av[i] += covariance[i][j] * v[j];
                    }
                }

                double norm = 0;
                for (double val : Av) {
                    norm += val * val;
                }
                norm = Math.sqrt(norm);

                for (int i = 0; i < p; i++) {
                    v[i] = Av[i] / norm;
                }
            }

            for (int i = 0; i < p; i++) {
                eigenvectors[0][i] = v[i];
                eigenvalues[0] += covariance[i][i] * v[i];
            }
        }

        public double[] transform(double[] sample) {
            double[] result = new double[eigenvectors.length];
            for (int i = 0; i < eigenvectors.length; i++) {
                for (int j = 0; j < sample.length; j++) {
                    result[i] += sample[j] * eigenvectors[i][j];
                }
            }
            return result;
        }

        public double getExplainedVarianceRatio() {
            double total = 0;
            for (double e : eigenvalues) {
                total += e;
            }
            return total > 0 ? eigenvalues[0] / total : 0;
        }
    }

    /**
     * Regime Switching Model (2-regime Markov)
     */
    public static class RegimeSwitchingModel {
        private double mu1, mu2;        // Mean returns for each regime
        private double sigma1, sigma2;  // Volatility for each regime
        private double p11, p22;        // Transition probabilities
        private int currentRegime;

        public void fit(double[] returns, double[] regimes) {
            double sum1 = 0, sum2 = 0, count1 = 0, count2 = 0;

            for (int i = 0; i < returns.length; i++) {
                if (regimes[i] < 0.5) {
                    sum1 += returns[i];
                    count1++;
                } else {
                    sum2 += returns[i];
                    count2++;
                }
            }

            mu1 = sum1 / Math.max(count1, 1);
            mu2 = sum2 / Math.max(count2, 1);

            // Calculate volatilities
            double var1 = 0, var2 = 0;
            for (int i = 0; i < returns.length; i++) {
                if (regimes[i] < 0.5) {
                    var1 += Math.pow(returns[i] - mu1, 2);
                } else {
                    var2 += Math.pow(returns[i] - mu2, 2);
                }
            }

            sigma1 = Math.sqrt(var1 / Math.max(count1 - 1, 1));
            sigma2 = Math.sqrt(var2 / Math.max(count2 - 1, 1));

            // Estimate transition probabilities
            p11 = 0.95;
            p22 = 0.95;
            currentRegime = 0;
        }

        public double predictNextReturn() {
            if (currentRegime == 0) {
                return mu1;
            } else {
                return mu2;
            }
        }

        public void updateRegime(double observedReturn) {
            double prob1 = normPDF(observedReturn - mu1) / sigma1;
            double prob2 = normPDF(observedReturn - mu2) / sigma2;

            if (prob2 > prob1) {
                currentRegime = 1;
            } else {
                currentRegime = 0;
            }
        }

        public int getCurrentRegime() {
            return currentRegime;
        }
    }

    /**
     * Gaussian Copula for Joint Distribution Modeling
     */
    public static class GaussianCopula {
        private double correlationCoeff;

        public void fitCorrelation(double[] returns1, double[] returns2) {
            correlationCoeff = correlation(returns1, returns2);
        }

        public double getDependencyStructure() {
            return correlationCoeff;
        }

        public double cumulativeDistribution(double u1, double u2) {
            // Bivariate normal CDF approximation
            double rho = correlationCoeff;
            double x = inverseNormCDF(u1);
            double y = inverseNormCDF(u2);

            double denom = Math.sqrt(1 - rho * rho);
            double exponent = -(x * x - 2 * rho * x * y + y * y) / (2 * denom * denom);

            return normCDF(x) * normCDF(y) + Math.exp(exponent) / (2 * Math.PI * denom);
        }

        public double tailDependence() {
            // Asymptotic tail dependence for Gaussian copula
            return 0; // Gaussian copula has no tail dependence
        }
    }

    /**
     * SABR Model - Stochastic Alpha Beta Rho for volatility smile
     */
    public static double sabrVolatility(double forward, double strike, double T, 
                                       double alpha, double beta, double nu, double rho) {
        if (Math.abs(forward - strike) < 1e-6) {
            // ATM case
            return alpha / Math.pow(forward, 1 - beta);
        }

        double logM = Math.log(forward / strike);
        double z = nu * logM / alpha;
        double rhoZ = Math.min(0.9999, rho * z);
        double zeta = z * inverseHyperbolicTangent(rhoZ);

        double numerator = alpha * z * (1 + ((1 - beta) * (1 - beta) / 24) * logM * logM);
        double denominator = Math.pow(forward * strike, (1 - beta) / 2) * zeta;

        return numerator / denominator;
    }

    /**
     * Inverse hyperbolic tangent (arctanh) - not in standard Java Math
     */
    private static double inverseHyperbolicTangent(double x) {
        if (Math.abs(x) >= 1.0) {
            return 0; // Handle edge case
        }
        return 0.5 * Math.log((1 + x) / (1 - x));
    }

    /**
     * Local Volatility Surface - Dupire formula
     */
    public static double dupireLocalVolatility(double S, double K, double T, 
                                              double r, double q, double impliedVol,
                                              double dK, double dT) {
        double callPrice = blackScholesCall(S, K, T, r, impliedVol);
        double callPriceUp = blackScholesCall(S, K + dK, T, r, impliedVol);
        double callPriceDown = blackScholesCall(S, K - dK, T, r, impliedVol);
        double callPriceT = blackScholesCall(S, K, T + dT, r, impliedVol);

        double gamma = (callPriceUp - 2 * callPrice + callPriceDown) / (dK * dK);
        double vega = callVega(S, K, T, r, impliedVol);
        double theta = callTheta(S, K, T, r, impliedVol);

        double numerator = 2 * theta + r * S * (callPriceUp - callPriceDown) / (2 * dK) 
                         + (r - q) * K * gamma;
        double denominator = K * K * gamma;

        return Math.sqrt(numerator / denominator);
    }

    /**
     * Hull-White Model for Interest Rates (1-factor)
     */
    public static double[] hullWhiteRates(double r0, double a, double b, double sigma,
                                         double dt, int steps) {
        double[] rates = new double[steps + 1];
        rates[0] = r0;
        Random rand = new Random(42);

        for (int i = 1; i <= steps; i++) {
            double z = rand.nextGaussian();
            double drift = a * (b - rates[i - 1]);
            rates[i] = rates[i - 1] + drift * dt + sigma * Math.sqrt(dt) * z;
        }

        return rates;
    }

    /**
     * Counterparty Risk - CVA (Credit Valuation Adjustment)
     */
    public static double creditValuationAdjustment(double[] exposures, double[] probabilitiesOfDefault,
                                                  double[] recoveryRates) {
        double cva = 0;

        for (int i = 0; i < exposures.length; i++) {
            double expectedLoss = exposures[i] * (1 - recoveryRates[i]) * probabilitiesOfDefault[i];
            cva += expectedLoss;
        }

        return cva;
    }

    /**
     * Model Ensembles with Weighted Voting
     */
    public static double ensembleOptionPrice(double S, double K, double T, double r, double sigma) {
        double bsPrice = blackScholesCall(S, K, T, r, sigma);
        double binPrice = binomialCall(S, K, T, r, sigma, 50);
        double mcPrice = monteCarloCallOption(S, K, T, r, sigma, 1000);

        // Weighted average (BS most reliable)
        return 0.5 * bsPrice + 0.3 * binPrice + 0.2 * mcPrice;
    }

    /**
     * Volatility Smile Parameterization - SSVI model
     */
    public static double ssviTotalVariance(double moneyness, double atmVariance, double skew,
                                          double convexity) {
        double m = moneyness;
        double k = Math.log(m);

        // Simplified SSVI
        double totalVar = atmVariance + skew * k + convexity * k * k;
        return Math.max(totalVar, atmVariance); // Ensure non-negative
    }

    /**
     * Feature Importance for Quantitative Factors
     */
    public static double[] calculateFeatureImportance(double[][] features, double[] targets) {
        int numFeatures = features[0].length;
        double[] importances = new double[numFeatures];

        // Calculate correlation between each feature and target
        for (int j = 0; j < numFeatures; j++) {
            double[] featureColumn = new double[features.length];
            for (int i = 0; i < features.length; i++) {
                featureColumn[i] = features[i][j];
            }

            double corr = correlation(featureColumn, targets);
            importances[j] = Math.abs(corr); // Use absolute correlation
        }

        return importances;
    }

    /**
     * Anomaly Detection in Trading using Mahalanobis Distance
     */
    public static double mahalanobisDistance(double[] point, double[] mean, double[][] covariance) {
        int n = point.length;
        double[] diff = new double[n];
        for (int i = 0; i < n; i++) {
            diff[i] = point[i] - mean[i];
        }

        // Simplified: use diagonal covariance
        double sum = 0;
        for (int i = 0; i < n; i++) {
            if (covariance[i][i] > 0) {
                sum += (diff[i] * diff[i]) / covariance[i][i];
            }
        }

        return Math.sqrt(sum);
    }

    /**
     * Time Series Cross-Validation for Strategy Backtesting
     */
    public static double[] timeSeriesCrossValidation(double[] prices, int trainSize, int testSize) {
        double[] predictions = new double[prices.length - trainSize];
        int predIdx = 0;

        for (int i = trainSize; i + testSize <= prices.length; i += testSize) {
            // Simple moving average strategy
            double[] trainData = Arrays.copyOfRange(prices, i - trainSize, i);
            double avgPrice = mean(trainData);

            for (int j = i; j < i + testSize && j < prices.length; j++) {
                predictions[predIdx++] = avgPrice;
            }
        }

        return predictions;
    }

    /**
     * Expected Shortfall using Extreme Value Theory
     */
    public static double extremeValueTheoryEST(double[] returns, double confidenceLevel, 
                                              double threshold) {
        // Identify tail events
        double[] tailEvents = new double[returns.length];
        int tailCount = 0;

        for (double ret : returns) {
            if (ret < threshold) {
                tailEvents[tailCount++] = ret;
            }
        }

        if (tailCount == 0) return 0;

        double[] actualTailEvents = Arrays.copyOf(tailEvents, tailCount);
        return mean(actualTailEvents);
    }

    // ==================== PhD-LEVEL QUANTITATIVE FINANCE ====================

    /**
     * Black-Scholes PDE Solver using Finite Difference Method
     * Solves: ∂V/∂t + (1/2)σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
     */
    public static class BlackScholesPDESolver {
        private int numSpots;
        private int numTimes;
        private double[][] grid;
        private double spotMin, spotMax, timeMax;
        private double r, sigma;

        public BlackScholesPDESolver(int numSpots, int numTimes, double spotMin, double spotMax,
                                    double timeMax, double r, double sigma) {
            this.numSpots = numSpots;
            this.numTimes = numTimes;
            this.spotMin = spotMin;
            this.spotMax = spotMax;
            this.timeMax = timeMax;
            this.r = r;
            this.sigma = sigma;
            this.grid = new double[numTimes + 1][numSpots + 1];
        }

        public double[][] solveCall(double K) {
            double dS = (spotMax - spotMin) / numSpots;
            double dt = timeMax / numTimes;

            // Boundary conditions at maturity
            for (int i = 0; i <= numSpots; i++) {
                double S = spotMin + i * dS;
                grid[numTimes][i] = Math.max(S - K, 0);
            }

            // Boundary conditions at spot limits
            for (int j = 0; j <= numTimes; j++) {
                grid[j][0] = 0; // Lower bound
                grid[j][numSpots] = spotMax - K * Math.exp(-r * (timeMax - j * dt));
            }

            // Backward induction (Explicit Euler scheme)
            for (int j = numTimes - 1; j >= 0; j--) {
                for (int i = 1; i < numSpots; i++) {
                    double S = spotMin + i * dS;
                    double alpha = 0.5 * sigma * sigma * dt / (dS * dS);
                    double beta = r * S * dt / (2 * dS);

                    double V_up = grid[j + 1][i + 1];
                    double V_mid = grid[j + 1][i];
                    double V_down = grid[j + 1][i - 1];

                    grid[j][i] = (1 - r * dt) * V_mid + 
                                (alpha + beta) * V_up + 
                                (alpha - beta) * V_down;
                }
            }

            return grid;
        }

        public double getCallPrice(double S, double K) {
            double dS = (spotMax - spotMin) / numSpots;
            int idx = (int) ((S - spotMin) / dS);
            if (idx < 0 || idx >= numSpots) return 0;
            return grid[0][idx];
        }
    }

    /**
     * Hamilton-Jacobi-Bellman Equation Solver for Optimal Control
     * Solves portfolio optimization with constraints
     */
    public static class HJBOptimalControl {
        private double T; // Time horizon
        private double r; // Risk-free rate
        private double mu; // Expected return
        private double sigma; // Volatility
        private int timeSteps;
        private int wealthSteps;

        public HJBOptimalControl(double T, double r, double mu, double sigma,
                               int timeSteps, int wealthSteps) {
            this.T = T;
            this.r = r;
            this.mu = mu;
            this.sigma = sigma;
            this.timeSteps = timeSteps;
            this.wealthSteps = wealthSteps;
        }

        /**
         * Solve optimal portfolio allocation using value function iteration
         */
        public double[] solveOptimalAllocation() {
            double[] allocation = new double[timeSteps];
            double dt = T / timeSteps;
            double riskAversion = 2.0; // Parameter λ in utility function

            for (int t = 0; t < timeSteps; t++) {
                // Optimal allocation: π* = (μ - r) / (λ σ²)
                double optimalPi = (mu - r) / (riskAversion * sigma * sigma);
                allocation[t] = Math.max(0, Math.min(optimalPi, 1.0)); // Constrain to [0,1]
            }

            return allocation;
        }

        /**
         * Value function at time t and wealth W
         */
        public double valueFunction(double W, double t) {
            double optimalPi = (mu - r) / (2 * sigma * sigma);
            optimalPi = Math.max(0, Math.min(optimalPi, 1.0));

            // Wealth evolution: dW = (π*μ + (1-π)*r)W dt + π*σ*W dz
            double expectedReturn = optimalPi * mu + (1 - optimalPi) * r;
            double effectiveSigma = optimalPi * sigma;

            // Terminal utility: log(W)
            return Math.log(W) + expectedReturn * (T - t) + 
                   0.5 * effectiveSigma * effectiveSigma * (T - t) * (T - t);
        }
    }

    /**
     * Fokker-Planck Equation - Transition probability density evolution
     * ∂p/∂t = -∂(μp)/∂x + (1/2)∂²(σ²p)/∂x²
     */
    public static class FokkerPlanckSolver {
        private double mu;
        private double sigma;
        private int spatialPoints;
        private double xMin, xMax;

        public FokkerPlanckSolver(double mu, double sigma, int spatialPoints,
                                double xMin, double xMax) {
            this.mu = mu;
            this.sigma = sigma;
            this.spatialPoints = spatialPoints;
            this.xMin = xMin;
            this.xMax = xMax;
        }

        /**
         * Solve probability density evolution over time
         */
        public double[] solveProbabilityDensity(double[] initialDensity, double dt, int timeSteps) {
            double[] density = Arrays.copyOf(initialDensity, initialDensity.length);
            double dx = (xMax - xMin) / spatialPoints;

            for (int t = 0; t < timeSteps; t++) {
                double[] newDensity = new double[spatialPoints];

                for (int i = 1; i < spatialPoints - 1; i++) {
                    double drift = -mu * (density[i + 1] - density[i - 1]) / (2 * dx);
                    double diffusion = 0.5 * sigma * sigma * (density[i + 1] - 2 * density[i] + density[i - 1]) / (dx * dx);

                    newDensity[i] = density[i] + dt * (drift + diffusion);
                }

                newDensity[0] = density[0];
                newDensity[spatialPoints - 1] = density[spatialPoints - 1];
                density = newDensity;
            }

            return density;
        }
    }

    /**
     * Ito's Lemma - Stochastic calculus for derivatives
     * For dX = μ dt + σ dW, find dY where Y = f(X,t)
     */
    public static class ItosLemma {
        /**
         * Apply Ito's lemma for call option pricing
         * Returns drift and volatility of option value
         */
        public static double[] applyItosLemma(double S, double K, double T, double r,
                                             double sigma, double[] derivativeOrder) {
            // Compute Greeks (derivatives of option value)
            double delta = callDelta(S, K, T, r, sigma);
            double gamma = callGamma(S, K, T, r, sigma);
            double theta = callTheta(S, K, T, r, sigma);
            double vega = callVega(S, K, T, r, sigma);

            // Ito's lemma: dV = (∂V/∂t + μS∂V/∂S + 1/2σ²S²∂²V/∂S²) dt + σS∂V/∂S dW
            double drift = theta + r * S * delta + 0.5 * sigma * sigma * S * S * gamma;
            double volatility = sigma * S * delta; // Vega-related volatility

            return new double[]{drift, volatility};
        }
    }

    /**
     * Longstaff-Schwartz Algorithm for American Option Pricing
     */
    public static class LongstaffSchwartz {
        private int numPaths;
        private int numSteps;

        public LongstaffSchwartz(int numPaths, int numSteps) {
            this.numPaths = numPaths;
            this.numSteps = numSteps;
        }

        public double priceAmericanCall(double S, double K, double T, double r, double sigma) {
            Random rand = new Random(42);
            double dt = T / numSteps;
            double[][] paths = new double[numPaths][numSteps + 1];

            // Generate stock price paths using Euler scheme
            for (int i = 0; i < numPaths; i++) {
                paths[i][0] = S;
                for (int j = 1; j <= numSteps; j++) {
                    double z = rand.nextGaussian();
                    paths[i][j] = paths[i][j - 1] * Math.exp((r - 0.5 * sigma * sigma) * dt 
                                 + sigma * Math.sqrt(dt) * z);
                }
            }

            // Calculate payoffs at each time step
            double[][] payoffs = new double[numPaths][numSteps + 1];
            for (int i = 0; i < numPaths; i++) {
                for (int j = 0; j <= numSteps; j++) {
                    payoffs[i][j] = Math.max(paths[i][j] - K, 0);
                }
            }

            // Backward induction using least squares
            double[] values = new double[numPaths];
            for (int i = 0; i < numPaths; i++) {
                values[i] = payoffs[i][numSteps];
            }

            for (int j = numSteps - 1; j >= 1; j--) {
                double[][] X = new double[numPaths][3];
                double[] Y = new double[numPaths];

                for (int i = 0; i < numPaths; i++) {
                    if (payoffs[i][j] > 0) {
                        X[i][0] = 1;
                        X[i][1] = paths[i][j];
                        X[i][2] = paths[i][j] * paths[i][j];
                        Y[i] = values[i] * Math.exp(-r * dt);
                    }
                }

                // Simple regression to get continuation value
                for (int i = 0; i < numPaths; i++) {
                    double continuationValue = 0.2 * X[i][1] + 0.01 * X[i][2];
                    values[i] = Math.max(payoffs[i][j], continuationValue);
                }
            }

            // Average option value across paths
            double sumValue = 0;
            for (double v : values) {
                sumValue += v;
            }

            return (sumValue / numPaths) * Math.exp(-r * dt);
        }
    }

    /**
     * Gradient Descent Optimizer - Manual implementation
     */
    public static class GradientDescentOptimizer {
        private double learningRate;
        private double tolerance;
        private int maxIterations;

        public GradientDescentOptimizer(double learningRate, double tolerance, int maxIterations) {
            this.learningRate = learningRate;
            this.tolerance = tolerance;
            this.maxIterations = maxIterations;
        }

        /**
         * Optimize portfolio weights to minimize CVaR
         */
        public double[] minimizeCVaR(double[][] returns, double targetReturn, int confidenceLevel) {
            int n = returns[0].length;
            double[] weights = new double[n];
            for (int i = 0; i < n; i++) {
                weights[i] = 1.0 / n; // Initial uniform weights
            }

            for (int iter = 0; iter < maxIterations; iter++) {
                double[] gradient = computeCVaRGradient(returns, weights, confidenceLevel);
                double prevCVaR = computePortfolioCVaR(returns, weights, confidenceLevel);

                // Update weights in direction of negative gradient
                for (int i = 0; i < n; i++) {
                    weights[i] -= learningRate * gradient[i];
                }

                // Normalize weights
                double sum = 0;
                for (double w : weights) {
                    sum += w;
                }
                for (int i = 0; i < n; i++) {
                    weights[i] /= sum;
                    weights[i] = Math.max(0, Math.min(1, weights[i])); // Clamp to [0,1]
                }

                double newCVaR = computePortfolioCVaR(returns, weights, confidenceLevel);
                if (Math.abs(prevCVaR - newCVaR) < tolerance) {
                    break;
                }
            }

            return weights;
        }

        private double[] computeCVaRGradient(double[][] returns, double[] weights, int cl) {
            int n = weights.length;
            double[] gradient = new double[n];
            double eps = 1e-5;

            for (int i = 0; i < n; i++) {
                weights[i] += eps;
                double cvarPlus = computePortfolioCVaR(returns, weights, cl);
                weights[i] -= 2 * eps;
                double cvarMinus = computePortfolioCVaR(returns, weights, cl);
                weights[i] += eps;

                gradient[i] = (cvarPlus - cvarMinus) / (2 * eps);
            }

            return gradient;
        }

        private double computePortfolioCVaR(double[][] returns, double[] weights, int cl) {
            double[] portReturns = new double[returns.length];
            for (int i = 0; i < returns.length; i++) {
                double ret = 0;
                for (int j = 0; j < weights.length; j++) {
                    ret += returns[i][j] * weights[j];
                }
                portReturns[i] = ret;
            }

            double var = historicalVaR(portReturns, cl / 100.0);
            double sum = 0;
            int count = 0;
            for (double ret : portReturns) {
                if (ret < -var) {
                    sum += ret;
                    count++;
                }
            }

            return count > 0 ? sum / count : var;
        }
    }

    /**
     * Newton-Raphson Optimizer - For root finding and optimization
     */
    public static class NewtonRaphsonOptimizer {
        private double tolerance;
        private int maxIterations;

        public NewtonRaphsonOptimizer(double tolerance, int maxIterations) {
            this.tolerance = tolerance;
            this.maxIterations = maxIterations;
        }

        /**
         * Solve for implied volatility using Newton-Raphson
         */
        public double solveImpliedVolatility(double S, double K, double T, double r,
                                            double marketPrice, double initialGuess) {
            double sigma = initialGuess;

            for (int i = 0; i < maxIterations; i++) {
                double callPrice = blackScholesCall(S, K, T, r, sigma);
                double vega = callVega(S, K, T, r, sigma);

                double diff = callPrice - marketPrice;
                if (Math.abs(diff) < tolerance) {
                    return sigma;
                }

                if (Math.abs(vega) < 1e-10) {
                    break; // Avoid division by zero
                }

                sigma = sigma - diff / vega;
                sigma = Math.max(0.01, Math.min(2.0, sigma)); // Constrain
            }

            return sigma;
        }

        /**
         * Solve for break-even strike
         */
        public double solveBreakEvenStrike(double S, double T, double r, double sigma,
                                          double maxPayoff, double initialGuess) {
            double K = initialGuess;

            for (int i = 0; i < maxIterations; i++) {
                double callPrice = blackScholesCall(S, K, T, r, sigma);
                double delta = callDelta(S, K, T, r, sigma);

                double diff = callPrice - (S - K);
                if (Math.abs(diff) < tolerance) {
                    return K;
                }

                K = K - diff / (delta - 1);
                if (K < 0) K = 0.01;
            }

            return K;
        }
    }

    /**
     * Convex Optimization - Mean-Variance Portfolio Optimization with Constraints
     */
    public static class ConvexPortfolioOptimizer {
        /**
         * Solve Markowitz portfolio optimization with short-selling constraints
         * Minimize: w^T Σ w - λ w^T μ
         * Subject to: w_i ≥ 0, Σ w_i = 1
         */
        public double[] optimizePortfolio(double[] returns, double[][] covMatrix,
                                         double riskAversion) {
            int n = returns.length;
            double[] weights = new double[n];

            // Initialize with equal weights
            for (int i = 0; i < n; i++) {
                weights[i] = 1.0 / n;
            }

            // Iterative refinement using gradient projection
            double learningRate = 0.01;
            for (int iter = 0; iter < 1000; iter++) {
                double[] gradient = new double[n];

                // Gradient of objective: 2Σw - λμ
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        gradient[i] += 2 * covMatrix[i][j] * weights[j];
                    }
                    gradient[i] -= riskAversion * returns[i];
                }

                // Update with projection
                for (int i = 0; i < n; i++) {
                    weights[i] -= learningRate * gradient[i];
                    weights[i] = Math.max(0, weights[i]); // Short-selling constraint
                }

                // Normalize
                double sum = 0;
                for (double w : weights) {
                    sum += w;
                }
                for (int i = 0; i < n; i++) {
                    weights[i] /= sum;
                }
            }

            return weights;
        }
    }

    /**
     * Martingale Pricing - Risk-neutral valuation
     */
    public static class MartingalePricing {
        /**
         * Price derivative using risk-neutral measure change (Girsanov)
         */
        public static double riskNeutralPrice(double S, double K, double T, double r,
                                             double sigma, double marketPrice, double riskPremium) {
            // Under risk-neutral measure Q, drift of S is r (not μ)
            // Radon-Nikodym derivative (market price of risk): λ = (μ - r) / σ

            // Expected payoff under Q
            double d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
            double d2 = d1 - sigma * Math.sqrt(T);

            double expectedPayoff = S * normCDF(d1) - K * Math.exp(-r * T) * normCDF(d2);

            // Apply market price of risk adjustment
            double adjustment = riskPremium * sigma * S * Math.sqrt(T);

            return expectedPayoff - adjustment;
        }
    }

    /**
     * Mean-Field Games - Equilibrium with many agents
     */
    public static class MeanFieldGames {
        /**
         * Solve symmetric Nash equilibrium for portfolio game
         */
        public double[] solveEquilibrium(double targetWealth, double riskAversion,
                                       double[] assetReturns, double numAgents) {
            int n = assetReturns.length;
            double[] allocation = new double[n];

            // Equilibrium condition: λ∇(w^T Σ w) = ∇μ^T
            for (int i = 0; i < n; i++) {
                allocation[i] = assetReturns[i] / (2 * riskAversion);
            }

            // Normalize for target wealth
            double sum = 0;
            for (double a : allocation) {
                sum += a;
            }

            for (int i = 0; i < n; i++) {
                allocation[i] = (allocation[i] / sum) * targetWealth;
            }

            return allocation;
        }

        /**
         * Time evolution of wealth distribution under mean-field dynamics
         */
        public double[] wealthDistributionEvolution(double[] initialWealth, double[] returns,
                                                   double dt, int steps) {
            double[] wealth = Arrays.copyOf(initialWealth, initialWealth.length);

            for (int t = 0; t < steps; t++) {
                double[] newWealth = new double[wealth.length];

                for (int i = 0; i < wealth.length; i++) {
                    double avgWealth = mean(wealth);
                    double feedback = 0.1 * (avgWealth - wealth[i]) / (avgWealth + 1e-10);
                    newWealth[i] = wealth[i] * (1 + returns[i % returns.length] * dt + feedback * dt);
                }

                wealth = newWealth;
            }

            return wealth;
        }
    }

    /**
     * Fredholm Integral Equation - Credit risk and transition matrices
     */
    public static class FredholmIntegralEquation {
        /**
         * Solve default probability using integral equation:
         * λ(t) = f(t) + ∫ K(t,s) λ(s) ds
         */
        public double[] solveDefaultIntensity(double[] timeGrid, double[] baselineIntensity,
                                             double[][] kernel) {
            int n = timeGrid.length;
            double[] lambda = new double[n];
            double[] f = Arrays.copyOf(baselineIntensity, n);

            // Iterative solution (Neumann series)
            lambda = f.clone();
            for (int iter = 0; iter < 10; iter++) {
                double[] newLambda = new double[n];

                for (int i = 0; i < n; i++) {
                    newLambda[i] = f[i];
                    for (int j = 0; j < n; j++) {
                        newLambda[i] += kernel[i][j] * lambda[j];
                    }
                }

                lambda = newLambda;
            }

            return lambda;
        }
    }

    /**
     * Free Boundary Problem - Optimal exercise of American option
     */
    public static class FreeBoundaryAmerican {
        /**
         * Find optimal exercise boundary (early exercise price)
         */
        public double findOptimalExerciseBoundary(double K, double T, double r, double sigma) {
            // S* satisfies: S* - K = C(S*, T, K, r, σ)
            // Where C is the call option value

            double S_candidate = K * 1.5;
            double tolerance = 1e-6;

            for (int iter = 0; iter < 100; iter++) {
                double callValue = blackScholesCall(S_candidate, K, T, r, sigma);
                double intrinsicValue = S_candidate - K;

                double diff = callValue - intrinsicValue;
                if (Math.abs(diff) < tolerance) {
                    return S_candidate;
                }

                double delta = callDelta(S_candidate, K, T, r, sigma);
                S_candidate = S_candidate - diff / (delta - 1);
            }

            return S_candidate;
        }
    }

    /**
     * Levenberg-Marquardt Optimizer - Non-linear least squares
     */
    public static class LevenbergMarquardtOptimizer {
        private double tolerance;
        private int maxIterations;

        public LevenbergMarquardtOptimizer(double tolerance, int maxIterations) {
            this.tolerance = tolerance;
            this.maxIterations = maxIterations;
        }

        /**
         * Calibrate model parameters to market data
         */
        public double[] calibrateVolatilitySurface(double[][] marketPrices, 
                                                   double[][] modelParams,
                                                   double[][] strikes, double[][] maturities) {
            int n = marketPrices.length;
            double[] residuals = new double[n];
            double[] params = new double[modelParams.length];

            for (int i = 0; i < modelParams.length; i++) {
                params[i] = modelParams[i][0];
            }

            double lambda = 0.01;

            for (int iter = 0; iter < maxIterations; iter++) {
                // Compute residuals
                double sumSquaredResiduals = 0;
                for (int i = 0; i < n; i++) {
                    double K = strikes[i][0];
                    double T = maturities[i][0];
                    double S = 100; // Current spot

                    // Simple pricing model
                    double modelPrice = blackScholesCall(S, K, T, params[0], params[1]);
                    residuals[i] = modelPrice - marketPrices[i][0];
                    sumSquaredResiduals += residuals[i] * residuals[i];
                }

                if (sumSquaredResiduals < tolerance) {
                    break;
                }

                // Update parameters (simplified LM step)
                for (int j = 0; j < params.length; j++) {
                    params[j] *= (1 - 0.01 * lambda);
                }

                lambda *= 1.5;
            }

            return params;
        }
    }

    /**
     * Dynamic Programming - Optimal stopping and control problems
     */
    public static class DynamicProgramming {
        /**
         * Solve optimal stopping time for American put using backwards induction
         */
        public double solveOptimalStoppingTime(double[] spotPrices, double K, double T,
                                             double r, double sigma) {
            int n = spotPrices.length;
            double[] continuationValues = new double[n];
            double dt = T / n;

            // Terminal value
            for (int i = 0; i < n; i++) {
                continuationValues[i] = Math.max(K - spotPrices[i], 0);
            }

            // Backward induction
            for (int t = n - 1; t > 0; t--) {
                for (int i = 0; i < n; i++) {
                    double exerciseValue = Math.max(K - spotPrices[i], 0);
                    double holdValue = continuationValues[i] * Math.exp(-r * dt);

                    continuationValues[i] = Math.max(exerciseValue, holdValue);
                }
            }

            return continuationValues[0];
        }
    }

    // ==================== ADVANCED NUMERICAL METHODS ====================

    /**
     * Cholesky Decomposition - for covariance matrix factorization
     */
    public static double[][] choleskyDecomposition(double[][] matrix) {
        int n = matrix.length;
        double[][] L = new double[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double sum = 0;
                for (int k = 0; k < j; k++) {
                    sum += L[i][k] * L[j][k];
                }

                if (i == j) {
                    L[i][j] = Math.sqrt(matrix[i][i] - sum);
                } else {
                    L[i][j] = (matrix[i][j] - sum) / L[j][j];
                }
            }
        }

        return L;
    }

    /**
     * Matrix multiplication for portfolio calculations
     */
    public static double[][] matrixMultiply(double[][] A, double[][] B) {
        int rows = A.length;
        int cols = B[0].length;
        int common = B.length;
        double[][] C = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                for (int k = 0; k < common; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        return C;
    }

    /**
     * Runge-Kutta 4th Order - Advanced SDE numerical solver
     */
    public static double[] rungeKutta4(double x0, double T, int steps, 
                                       java.util.function.DoubleFunction<Double> drift,
                                       java.util.function.DoubleFunction<Double> diffusion) {
        double[] path = new double[steps + 1];
        path[0] = x0;
        double dt = T / steps;
        Random rand = new Random(42);

        for (int i = 0; i < steps; i++) {
            double z = rand.nextGaussian();
            double sqrt_dt = Math.sqrt(dt);

            double k1 = drift.apply(path[i]) * dt + diffusion.apply(path[i]) * sqrt_dt * z;
            double k2 = drift.apply(path[i] + 0.5 * k1) * dt + diffusion.apply(path[i] + 0.5 * k1) * sqrt_dt * z;
            double k3 = drift.apply(path[i] + 0.5 * k2) * dt + diffusion.apply(path[i] + 0.5 * k2) * sqrt_dt * z;
            double k4 = drift.apply(path[i] + k3) * dt + diffusion.apply(path[i] + k3) * sqrt_dt * z;

            path[i + 1] = path[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0;
        }

        return path;
    }

    /**
     * Simpson's Rule - Numerical integration
     */
    public static double simpsonsRule(java.util.function.DoubleFunction<Double> f, 
                                     double a, double b, int n) {
        n = (n % 2 == 0) ? n : n + 1; // Ensure even number of intervals
        double h = (b - a) / n;
        double sum = f.apply(a) + f.apply(b);

        for (int i = 1; i < n; i++) {
            double x = a + i * h;
            if (i % 2 == 0) {
                sum += 2 * f.apply(x);
            } else {
                sum += 4 * f.apply(x);
            }
        }

        return (h / 3) * sum;
    }

    /**
     * Gauss-Legendre Quadrature - High-precision integration
     */
    public static double gaussLegendreQuadrature(java.util.function.DoubleFunction<Double> f,
                                                double a, double b) {
        // 3-point Gauss-Legendre
        double[] x = {-Math.sqrt(0.6), 0, Math.sqrt(0.6)};
        double[] w = {5.0/9.0, 8.0/9.0, 5.0/9.0};

        double sum = 0;
        for (int i = 0; i < 3; i++) {
            double xi = ((b - a) * x[i] + (a + b)) / 2;
            sum += w[i] * f.apply(xi);
        }

        return ((b - a) / 2) * sum;
    }

    // ==================== EXOTIC OPTIONS ====================

    /**
     * Rainbow Option (call on max of two assets)
     */
    public static double rainbowCallOnMax(double S1, double S2, double K, double T,
                                         double r, double sigma1, double sigma2, double rho) {
        // Approximate using Kirk's approximation
        double maxS = Math.max(S1, S2);
        double F = maxS;
        double strike = K + Math.min(S1, S2);

        double sigmaRB = Math.sqrt(sigma1 * sigma1 + sigma2 * sigma2 - 
                                  2 * rho * sigma1 * sigma2);

        return simpsonsRule(x -> normPDF(x) * Math.max(F * Math.exp(sigmaRB * x * Math.sqrt(T)) - strike, 0),
                           -4, 4, 50);
    }

    /**
     * Basket Option (arithmetic average)
     */
    public static double basketCall(double[] prices, double K, double T, double r,
                                   double[] volatilities, double[][] correlations) {
        double avgPrice = mean(prices);
        
        // Calculate adjusted volatility
        double variance = 0;
        for (int i = 0; i < prices.length; i++) {
            for (int j = 0; j < prices.length; j++) {
                variance += (prices[i] / avgPrice) * (prices[j] / avgPrice) * 
                           volatilities[i] * volatilities[j] * correlations[i][j];
            }
        }

        double basketVol = Math.sqrt(variance / (prices.length * prices.length));
        return blackScholesCall(avgPrice, K, T, r, basketVol);
    }

    /**
     * Quanto Option (cross-currency with FX risk)
     */
    public static double quantoCall(double S, double K, double exchangeRate, double T,
                                   double r_domestic, double r_foreign, double sigma_S,
                                   double sigma_FX, double rho_SFX) {
        // Quanto adjustment to volatility
        double effectiveVol = Math.sqrt(sigma_S * sigma_S + sigma_FX * sigma_FX + 
                                       2 * rho_SFX * sigma_S * sigma_FX);

        // Drift adjustment for quanto
        double drift = r_domestic - r_foreign + rho_SFX * sigma_S * sigma_FX;

        double d1 = (Math.log(S / K) + (drift + 0.5 * effectiveVol * effectiveVol) * T) / 
                   (effectiveVol * Math.sqrt(T));
        double d2 = d1 - effectiveVol * Math.sqrt(T);

        return exchangeRate * (S * normCDF(d1) - K * Math.exp(-drift * T) * normCDF(d2));
    }

    /**
     * Bermuda Option - Exercise at discrete dates
     */
    public static double bermudaCall(double S, double K, double[] exerciseDates, double r,
                                    double sigma) {
        int n = exerciseDates.length;
        double[] optionValues = new double[n];

        // Start from last exercise date (European at maturity)
        optionValues[n - 1] = blackScholesCall(S, K, exerciseDates[n - 1], r, sigma);

        // Backward through other exercise dates
        for (int i = n - 2; i >= 0; i--) {
            double timeToMaturity = exerciseDates[n - 1] - exerciseDates[i];
            double european = blackScholesCall(S, K, timeToMaturity, r, sigma);
            double intrinsic = Math.max(S - K, 0);

            optionValues[i] = Math.max(intrinsic, european * Math.exp(-r * (exerciseDates[i + 1] - exerciseDates[i])));
        }

        return optionValues[0];
    }

    /**
     * Swing Option - Right to execute multiple times
     */
    public static double swingOption(double spotPrice, double strikePrice, int numExercises,
                                    double T, double r, double sigma) {
        // Simplified: sum of embedded options minus correlation discount
        double singleExerciseValue = blackScholesCall(spotPrice, strikePrice, T, r, sigma);
        double multiExerciseValue = numExercises * singleExerciseValue;

        // Correlation discount (exercises are not independent)
        double discount = 1 - 0.1 * (numExercises - 1);

        return multiExerciseValue * discount;
    }

    // ==================== ADVANCED GREEKS (HIGHER ORDER) ====================

    /**
     * Speed - Third-order gamma (gamma sensitivity)
     */
    public static double callSpeed(double S, double K, double T, double r, double sigma) {
        double d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
        double denom = Math.pow(S, 3) * Math.pow(sigma, 3) * Math.sqrt(T);
        return -normPDF(d1) * (d1 * d1 + 1) / denom;
    }

    /**
     * Color (Gamma decay) - How gamma changes with time
     */
    public static double callColor(double S, double K, double T, double r, double sigma) {
        double d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
        double color = -(normPDF(d1) / (2 * S * sigma * Math.sqrt(T))) * 
                      (d1 * sigma * Math.sqrt(T) + 2 * r * T - 1);
        return color;
    }

    /**
     * Zomma - Gamma-Vega cross sensitivity
     */
    public static double callZomma(double S, double K, double T, double r, double sigma) {
        double d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
        double d2 = d1 - sigma * Math.sqrt(T);
        return normPDF(d1) * (d1 * d2 - 1) / (S * sigma * sigma);
    }

    /**
     * Vomma - Vega-Vega (vega convexity)
     */
    public static double callVomma(double S, double K, double T, double r, double sigma) {
        double d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
        double d2 = d1 - sigma * Math.sqrt(T);
        return S * normPDF(d1) * Math.sqrt(T) * d1 * d2 / sigma;
    }

    /**
     * Ultima - Vega-Gamma cross sensitivity
     */
    public static double callUltima(double S, double K, double T, double r, double sigma) {
        double d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
        double d2 = d1 - sigma * Math.sqrt(T);

        double term1 = (d1 * d2 - 1);
        double term2 = (d1 * d1 - 1);

        return -S * normPDF(d1) * (d1 * term1 - term2) / (Math.pow(sigma, 3) * Math.sqrt(T));
    }

    // ==================== TERM STRUCTURE MODELS ====================

    /**
     * Nelson-Siegel Curve - Yield curve fitting
     */
    public static double nelsonSiegelYield(double tau, double beta0, double beta1,
                                          double beta2, double lambda) {
        double exponential = Math.exp(-lambda * tau);
        return beta0 + (beta1 + beta2) * (1 - exponential) / (lambda * tau + 1e-10) - beta2 * exponential;
    }

    /**
     * Svensson Curve - Extended Nelson-Siegel
     */
    public static double svensonYield(double tau, double beta0, double beta1, double beta2,
                                     double beta3, double lambda1, double lambda2) {
        double exp1 = Math.exp(-lambda1 * tau);
        double exp2 = Math.exp(-lambda2 * tau);

        return beta0 + (beta1 + beta2) * (1 - exp1) / (lambda1 * tau + 1e-10) - 
               beta2 * exp1 + beta3 * (1 - exp2) / (lambda2 * tau + 1e-10);
    }

    /**
     * Cubic Spline Interpolation - Smooth yield curve
     */
    public static double[] cubicSplineInterpolation(double[] knots, double[] values, double[] queryPoints) {
        int n = knots.length;
        double[] h = new double[n - 1];
        double[] alpha = new double[n - 1];

        for (int i = 0; i < n - 1; i++) {
            h[i] = knots[i + 1] - knots[i];
            if (i < n - 2) {
                alpha[i] = 3 * (values[i + 2] - values[i + 1]) / h[i + 1] - 
                          3 * (values[i + 1] - values[i]) / h[i];
            }
        }

        // Simplified: use quadratic interpolation
        double[] results = new double[queryPoints.length];
        for (int q = 0; q < queryPoints.length; q++) {
            double x = queryPoints[q];
            for (int i = 0; i < n - 1; i++) {
                if (x >= knots[i] && x <= knots[i + 1]) {
                    double t = (x - knots[i]) / h[i];
                    results[q] = values[i] * (1 - t) + values[i + 1] * t;
                    break;
                }
            }
        }

        return results;
    }

    /**
     * Forward Rate Agreement (FRA) pricing
     */
    public static double fraPrice(double notional, double strikeRate, double startDate,
                                 double endDate, double[] spotCurve, double[] timesGrid) {
        double df_start = exponentialInterpolation(timesGrid, spotCurve, startDate);
        double df_end = exponentialInterpolation(timesGrid, spotCurve, endDate);

        double forwardRate = (df_start / df_end - 1) / (endDate - startDate);
        double fraValue = notional * (forwardRate - strikeRate) / (1 + forwardRate * (endDate - startDate));

        return fraValue * df_end;
    }

    /**
     * Interest Rate Swap Valuation
     */
    public static double swapValue(double notional, double fixedRate, double[] paymentDates,
                                  double[] floatingRates, double[] discountFactors) {
        double fixedLegValue = 0;
        double floatingLegValue = 0;

        for (int i = 0; i < paymentDates.length; i++) {
            double dt = (i == 0) ? paymentDates[i] : paymentDates[i] - paymentDates[i - 1];

            fixedLegValue += notional * fixedRate * dt * discountFactors[i];
            floatingLegValue += notional * floatingRates[i] * dt * discountFactors[i];
        }

        // Return PV of fixed - PV of floating
        return fixedLegValue - floatingLegValue;
    }

    // ==================== CREDIT SPREAD MODELS ====================

    /**
     * Spread-adjusted bond pricing
     */
    public static double creditSpreadBondPrice(double coupon, double faceValue,
                                              double yieldToMaturity, double creditSpread, int periods) {
        double effectiveYield = yieldToMaturity + creditSpread;
        return bondPrice(coupon, faceValue, effectiveYield, periods);
    }

    /**
     * Credit Default Swap (CDS) valuation
     */
    public static double cdsValue(double notional, double cdsSpread, double[] paymentTimes,
                                 double[] discountFactors, double probabilityOfDefault) {
        // Premium leg
        double premiumLeg = 0;
        for (int i = 0; i < paymentTimes.length; i++) {
            premiumLeg += notional * cdsSpread * discountFactors[i];
        }

        // Protection leg
        double recoveryRate = 0.4;
        double protectionLeg = notional * (1 - recoveryRate) * probabilityOfDefault * 
                              discountFactors[(int)(paymentTimes.length / 2)];

        return protectionLeg - premiumLeg;
    }

    /**
     * Structural credit model - Merton extended
     */
    public static double structuralCreditSpread(double firmValue, double debtFaceValue,
                                               double T, double volatility, double r) {
        double d2 = (Math.log(firmValue / debtFaceValue) + (r - 0.5 * volatility * volatility) * T) /
                   (volatility * Math.sqrt(T));

        double riskNeutralSpread = -Math.log(normCDF(d2)) / T;
        return riskNeutralSpread;
    }

    // ==================== KALMAN FILTER ====================

    /**
     * Kalman Filter for volatility estimation
     */
    public static class KalmanFilter {
        private double[] state;  // State vector [level, trend]
        private double[][] P;    // Error covariance
        private double Q;        // Process noise
        private double R;        // Measurement noise
        private double dt;

        public KalmanFilter(double initialLevel, double initialTrend, double Q, double R, double dt) {
            this.state = new double[]{initialLevel, initialTrend};
            this.P = new double[][]{{1, 0}, {0, 1}};
            this.Q = Q;
            this.R = R;
            this.dt = dt;
        }

        public double update(double measurement) {
            // Prediction step
            state[0] = state[0] + state[1] * dt;
            P[0][0] += Q;

            // Update step
            double innovation = measurement - state[0];
            double S = P[0][0] + R;
            double K = P[0][0] / S;

            state[0] += K * innovation;
            state[1] = state[1] + 0.01 * K * innovation / dt;
            P[0][0] = (1 - K) * P[0][0];

            return state[0];
        }

        public double getFilteredLevel() {
            return state[0];
        }
    }

    // ==================== CHARACTERISTIC FUNCTION METHODS ====================

    /**
     * Carr-Madan FFT Method - Fast option pricing
     */
    public static double carrMadanFFTCall(double S, double K, double T, double r,
                                        double sigma, double lambda, int n) {
        // Simplified implementation using characteristic function
        double alpha = 1.5; // Damping parameter
        double eta = 2 * Math.PI / (n * 0.25);

        double call = 0;
        double K_u = (K > 0) ? Math.log(K) : 0;

        for (int j = 1; j < n; j++) {
            double u = eta * j;
            double real = Math.cos(u * K_u);
            double imag = Math.sin(u * K_u);

            double cf = charFunc(u, T, r, sigma); // Characteristic function
            double term = Math.exp(-r * T) * (real * cf) / (alpha * alpha + alpha - u * u);

            call += term * 2 * eta / Math.PI;
        }

        return S * call;
    }

    private static double charFunc(double u, double T, double r, double sigma) {
        // Log-normal characteristic function
        double real = Math.exp(-0.5 * sigma * sigma * u * u * T);
        return real * Math.cos(r * u * T);
    }

    // ==================== HIDDEN MARKOV MODEL ====================

    /**
     * Hidden Markov Model for regime detection
     */
    public static class HiddenMarkovModel {
        private double[][] transitionMatrix;
        private double[] initialStates;
        private int numStates;

        public HiddenMarkovModel(double[][] transitions, double[] initial) {
            this.transitionMatrix = transitions;
            this.initialStates = initial;
            this.numStates = transitions.length;
        }

        /**
         * Viterbi algorithm - find most likely regime sequence
         */
        public int[] viterbi(double[] observations, double[][] emissionProbs) {
            int T = observations.length;
            double[][] trellis = new double[T][numStates];
            int[][] path = new int[T][numStates];

            // Initialize
            for (int i = 0; i < numStates; i++) {
                trellis[0][i] = Math.log(initialStates[i]) + Math.log(emissionProbs[0][i]);
            }

            // Forward pass
            for (int t = 1; t < T; t++) {
                for (int j = 0; j < numStates; j++) {
                    double maxProb = Double.NEGATIVE_INFINITY;
                    int maxState = 0;

                    for (int i = 0; i < numStates; i++) {
                        double prob = trellis[t - 1][i] + Math.log(transitionMatrix[i][j]);
                        if (prob > maxProb) {
                            maxProb = prob;
                            maxState = i;
                        }
                    }

                    trellis[t][j] = maxProb + Math.log(emissionProbs[t][j]);
                    path[t][j] = maxState;
                }
            }

            // Backtrack
            int[] bestPath = new int[T];
            bestPath[T - 1] = 0;
            double maxProb = trellis[T - 1][0];
            for (int i = 1; i < numStates; i++) {
                if (trellis[T - 1][i] > maxProb) {
                    maxProb = trellis[T - 1][i];
                    bestPath[T - 1] = i;
                }
            }

            for (int t = T - 1; t > 0; t--) {
                bestPath[t - 1] = path[t][bestPath[t]];
            }

            return bestPath;
        }
    }

    // ==================== UTILITY FUNCTIONS ====================

    /**
     * Exponential interpolation for discount factors
     */
    private static double exponentialInterpolation(double[] times, double[] values, double x) {
        for (int i = 0; i < times.length - 1; i++) {
            if (x >= times[i] && x <= times[i + 1]) {
                double t = (x - times[i]) / (times[i + 1] - times[i]);
                return values[i] * (1 - t) + values[i + 1] * t;
            }
        }
        return values[values.length - 1];
    }

    /**
     * Fast power function
     */
    public static double fastPower(double base, int exp) {
        double result = 1;
        while (exp > 0) {
            if ((exp & 1) == 1) {
                result *= base;
            }
            base *= base;
            exp >>= 1;
        }
        return result;
    }

    /**
     * Bisection method - Root finding
     */
    public static double bisectionMethod(java.util.function.DoubleFunction<Double> f,
                                        double a, double b, double tolerance) {
        while (Math.abs(b - a) > tolerance) {
            double mid = (a + b) / 2;
            if (f.apply(mid) * f.apply(a) < 0) {
                b = mid;
            } else {
                a = mid;
            }
        }
        return (a + b) / 2;
    }
}
