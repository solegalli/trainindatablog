---
layout: post
title: "Multi-Seasonal Time Series Decomposition Using MSTL in Python"
author: kishan
description: "Masterclass on multi-seasonal time series decomposition using MSTL in Python. Discover how it works and see in action on real world data."
excerpt: "Masterclass on multi-seasonal time series decomposition using MSTL in Python. Discover how it works and see in action on real world data."
categories: [Time Series Forecasting]
image: assets/images/posts/multi-seasonal-time-series-decomposition-using-mstl-in-python/Blog-banners.png
---

In this article, we’ll decompose a time series with multiple seasonal components. We’ll explore a recently developed algorithm called Multiple Seasonal-Trend decomposition using Loess (MSTL) [[1](https://arxiv.org/abs/2107.13462)] and discuss its advantages over existing methods. Finally, we’ll try out MSTL in Python using a newly added module in statsmodels and apply it to real world data.

## Introduction

Time series decomposition is about breaking up a time series into components, most notably: a trend component, a seasonal component, and a residual component. There are many methods to decompose a time series with a single seasonal component implemented in Python, such as [STL](https://www.statsmodels.org/devel/generated/statsmodels.tsa.seasonal.STL.html) [2]and [X-13-ARIMA-SEATS](https://www.statsmodels.org/dev/generated/statsmodels.tsa.x13.x13_arima_analysis.html) [3]. But what about time series which have multiple seasonal components?

Consider electricity demand for example. Electricity demand has daily seasonality (more demand during the day than late at night), weekly seasonality (weekends vs weekdays), and yearly seasonality (demand in the summer and winter differs due to heating and cooling needs).

How common are multi-seasonal time series? Surprisingly common. Weather can drive multiple seasonalities. Think of how temperature and daylight hours repeat in daily and yearly cycles. Is it just about the weather? No. Human activities (e.g., the 9-to-5 workday, weekdays vs weekends, monthly payday) also repeat daily, weekly, and so on. Thus, time series ranging from air pollution to restaurant demand can have multiple seasonal components.

So now we know many time series can have multiple seasonalities, let’s think about how we can decompose them. In July 2021 [Bandara, Hyndman, and](https://arxiv.org/abs/2107.13462) [Bergmeir](https://arxiv.org/abs/2107.13462) proposed a new algorithm for multi-seasonal decomposition called Multiple Seasonal-Trend decomposition using Loess (MSTL) [1]. MSTL will be the focus of this article.

We’ll first revisit the concept of time series decomposition. Then, we’ll explore MSTL, how it works, and why to use it. Finally, we’ll use MSTL via a module in Statmodels, which [I recently added](https://github.com/statsmodels/statsmodels/pull/8160), and use it to decompose an electricity demand time series.

> To master time series decomposition and how to leverage the components to create features for forecasting, check out my course [Feature Engineering for Time Series Forecasting](https://www.trainindata.com/p/feature-engineering-for-forecasting).

## Time series decomposition

As you know by now, time series decomposition is about breaking up a time series into trend, seasonality, and residuals (Fig. 1).

![Fig. 1. Time series decomposition methods can be additive or multiplicative. The purpose of a decomposition method is to calculate the individual components. Image by author.]({{ site.baseurl }}/assets/images/posts/multi-seasonal-time-series-decomposition-using-mstl-in-python/fig01.webp)

Fig. 1. Time series decomposition methods can be additive or multiplicative. The purpose of a decomposition method is to calculate the individual components. Image by author.

But why would you decompose a time series? Time series decomposition is used in time series analysis for tasks such as:

- exploratory data analysis (e.g., has unemployment increased this quarter after adjusting for seasonality?);
- pre-processing time series to identify and impute outliers and missing data;
- extracting features from time series for later use in classification, regression, and forecasting tasks;
- building forecasts (e.g., the components can be forecasted separately and then aggregated to produce the final forecast).

Some commonly used methods for time series decomposition include:

- [A naive method](https://www.statsmodels.org/devel/generated/statsmodels.tsa.seasonal.seasonal_decompose.html) where we apply a moving average to extract a trend and take averages of a seasonal index (e.g., month) to extract seasonality;
- [Seasonal-Trend decomposition using Loess (STL)](https://www.statsmodels.org/devel/generated/statsmodels.tsa.seasonal.STL.html) [2];
- [X-13-ARIMA-SEATS](https://www.statsmodels.org/dev/generated/statsmodels.tsa.x13.x13_arima_analysis.html) [3].

These methods are designed to extract a single seasonal component from a time series.

Methods that can extract multiple seasonal components include Prophet [4], TBATS [5], and STR [6]. Prophet and TBATS infer the seasonal components as part of training a forecasting model. MSTL instead directly focuses on just decomposing a time series. Later we’ll see that MSTL outperforms these other methods on a set of benchmark time series [1]. With that, let’s dive into the details of MSTL.

> *Master time series decomposition methods and how to extract features for forecasting, with my course* [*Feature Engineering for Time Series Forecasting*](https://www.trainindata.com/p/feature-engineering-for-forecasting)*.*

[![Feature engineering for time series forecasting course]({{ site.baseurl }}/assets/images/posts/multi-seasonal-time-series-decomposition-using-mstl-in-python/feature-engineering-forecasting-course.png)](https://www.trainindata.com/p/feature-engineering-for-forecasting)

## MSTL: What is it?

MSTL stands for **M**ultiple **S**easonal-**T**rend decomposition using **L**oess [1]. It is a method to decompose a time series into a trend component, multiple seasonal components, and a residual component.

MSTL assumes that the time series can be expressed as an additive decomposition (Fig. 2):

![Fig. 2. An additive decomposition with multiple seasonal components. Image by author.]({{ site.baseurl }}/assets/images/posts/multi-seasonal-time-series-decomposition-using-mstl-in-python/fig02.webp)

Fig. 2. An additive decomposition with multiple seasonal components. Image by author.

where each seasonal component represents a different seasonality (e.g, daily, weekly, yearly, etc.). We’ll see later that MSTL builds on STL to iteratively extract each of the seasonal components.

## MSTL: Why use it?

In [1] the authors compared MSTL to other multi-seasonal decomposition methods: Prophet [4], TBATS [5], and STR [6]. They compared the accuracy and computational efficiency of each of these methods. They propose that MSTL is useful because it is:

- Accurate, MSTL typically produced the lowest RMSE on a range of benchmark time series;
- Computationally efficient, MSTL had the lowest execution time of all the methods compared;
- Robust to outliers, MSTL is able to use an outlier robust version of STL by passing a flag to the underlying STL fit;

I would also add:

- MSTL can model seasonality which changes with time (e.g., the daily pattern of electricity demand will be different in the winter compared to the summer).

Simplicity is another advantage that MSTL offers over forecasting models that extract the seasonal components (e.g., Prophet and TBATS). With forecasting models, there are many parameters to tune and things to consider to create a good forecast and therefore a good decomposition. Whereas MSTL focuses entirely on decomposing the time series and, in my experience, has fewer parameters that require tuning.

So now we know what MSTL is and why to use it. Let’s move on to how MSTL works.

## MSTL: How does it work?

MSTL builds on STL (short for **S**easonal-**T**rend decomposition using **L**oess), which is a decomposition method that can extract a single seasonal component. STL in turn builds on a smoothing method called Loess [7] (short for **Lo**cally **E**stimated **S**catterplot **S**moothing). Loess and STL deserve a blog post all to themselves, so I’ll only give a high-level summary of them so we can focus on MSTL.

### Loess overview

Loess is a method used to fit a smooth curve to scatterplots [7]. Loess computes the smooth curve at any given point by fitting a polynomial to a window of data around that point. This gives us two important arguments: the **window size**, which determines how smooth the fit is, and the **degree of the polynomia**l (typically this is 0 or 1).

Loess is easier to understand visually (Fig. 3). In the animation we can see the procedure to compute the Loess curve (grey points) from the real data (blue points): we look at a window of data (red points) where we want to compute the Loess value and fit a curve within that window (the line). In this example the polynomial degree is one.

The curve that Loess fits through a time series can be used to model the trend of a time series (Fig. 3). In Fig. 4, we see that Loess is able to capture the trend of a retail sales time series when an appropriate window size is selected.

![Fig. 3. An illustration of Loess fitting a curve to a scatter plot. At each point a polynomial (in this case a line) is fitted to a window of data points (red). The green curve represents weights used in the polynomial fit such that data at the edges of the window have less weight than the centre. Image by author.]({{ site.baseurl }}/assets/images/posts/multi-seasonal-time-series-decomposition-using-mstl-in-python/Untitled-design1.gif)

Fig. 3. An illustration of Loess fitting a curve to a scatter plot. At each point a polynomial (in this case a line) is fitted to a window of data points (red). The green curve represents weights used in the polynomial fit such that data at the edges of the window have less weight than the centre. Image by author.

Below we see the effect of LOESS with different window sizes:

![Fig. 4. Loess fits (orange) to a monthly retail sales dataset (blue) for different values of the window size parameter `frac`. When the window size is too small (frac=0.02) we see that Loess overfits to the seasonality, when the window size is too large (frac=0.8) it underfits the data, and when a good window size is chosen (frac=0.1) Loess captures the overall trend of the time series. The retail sales dataset is a monthly dataset of retail sales volumes and can be found here. Image by author.]({{ site.baseurl }}/assets/images/posts/multi-seasonal-time-series-decomposition-using-mstl-in-python/fig04.webp)

Fig. 4. Loess fits (orange) to a monthly retail sales dataset (blue) for different values of the window size parameter `frac`. When the window size is too small (frac=0.02) we see that Loess overfits to the seasonality, when the window size is too large (frac=0.8) it underfits the data, and when a good window size is chosen (frac=0.1) Loess captures the overall trend of the time series. The retail sales dataset is a monthly dataset of retail sales volumes and can be found here. Image by author.

### STL overview

The objective of STL is to extract a single seasonal component, a trend component, and a residual component from a time series (Fig. 5). It does this by applying Loess to multiple transformations of the original time series and recursively extracts the trend and seasonal component. There is a great explanation of STL in the original paper [2] and in this blog post [[8](http://www.gardner.fyi/blog/STL-Part-II/)].

![Fig. 5. STL decomposition of a retail sales dataset. The retail sales dataset is a monthly dataset of retail sales volumes and can be found here. Image by author.]({{ site.baseurl }}/assets/images/posts/multi-seasonal-time-series-decomposition-using-mstl-in-python/fig05.webp)

Fig. 5. STL decomposition of a retail sales dataset. The retail sales dataset is a monthly dataset of retail sales volumes and can be found here. Image by author.

So what are the main parameters of STL to know to understand MSTL? We will see that MSTL uses the seasonal component extracted by STL. STL computes the seasonal component using Loess. Hence, there is a parameter in STL for a window size and polynomial degree associated with the seasonal component. A summary of STL seasonal parameters are therefore:

- `period`, the period of the seasonal component we want STL to extract (e.g., `period=12` if we want yearly seasonality and the frequency of the data is monthly);
- `seasonal`, the window size of Loess used to extract the seasonal component in STL (this determines how smooth and regular the extracted seasonal component is);
- `seasonal_deg`, the polynomial degree used by Loess to extract the seasonal component in STL (typically set to 0 or 1).

### MSTL algorithm

Now we know what MSTL is (an algorithm that builds on STL to extract multiple seasonal components) and why to use it (it outperforms existing methods), let’s discuss the algorithm. The MSTL algorithm can be divided into: 1) pre-processing steps and 2) decomposition steps.

> For video tutorials on MSTL and other methods of time series decomposition, check out my course [Feature Engineering for Time Series Forecasting](https://www.trainindata.com/p/feature-engineering-for-forecasting).

### Pre-processing steps

There are two pre-processing steps which are commonly used for many time series analysis tasks, not just MSTL.

**Step 1: Impute missing data.** In the [R implementation](https://www.rdocumentation.org/packages/forecast/versions/8.16/topics/mstl) of MSTL this is done using `na.interp`. In the Python implementation you must impute missing data before using MSTL.

**Step 2: Apply a** [**Box Cox**](https://otexts.com/fpp2/transformations.html#mathematical-transformations) **transform if specified by the user.** This is used if we think the time series is **not** described by an additive decomposition. The Box Cox transform can convert the original time series into a new one which can be described by an additive decomposition.

These two steps can be skipped if the input time series does not have missing data and a Box Cox transform is not required.

### Decomposition

We break the decomposition part of the algorithm into multiple steps.

**Step 1: Extract each seasonal component using STL.** We pass the period of each seasonal component that we want to extract to MSTL (e.g., for hourly data with daily and weekly seasonality `periods = (24, 24*7)` — a period of 24 hours for daily seasonality, a period of 24*7=168 hours for weekly seasonality, etc.).

We will iterate through each seasonal component starting from the shortest period (e.g., daily) to the longest period (e.g., yearly). On each iteration, we extract the seasonal component via STL (Fig. 6a, 6b) and then subtract it from the time series (Fig. 6c). We then pass the resulting de-seasonalised time series to the next iteration where we extract the next seasonal component (Fig. 6d) and so on until all seasonal components have been extracted (Fig. 6e).

![Fig. 6. Step 1 of the decomposition part of the MSTL algorithm. This figure shows an example of a time series that has a trend, daily seasonality, weekly seasonality, and noise. Iteratively extract and subtract the seasonal components from the original time series until all seasonalities have been extracted. Image by author.]({{ site.baseurl }}/assets/images/posts/multi-seasonal-time-series-decomposition-using-mstl-in-python/fig06.webp)

Fig. 6. Step 1 of the decomposition part of the MSTL algorithm. This figure shows an example of a time series that has a trend, daily seasonality, weekly seasonality, and noise. Iteratively extract and subtract the seasonal components from the original time series until all seasonalities have been extracted. Image by author.

Why do we start iterating with the shortest period? Because if the longer seasonality were extracted first then the algorithm would erroneously include the shorter seasonality as part of the longer seasonal component.

**Step 2: Refine each of the extracted seasonal components.** So far we have an estimate for each seasonal component and a fully de-seasonalised time series (Fig. 7a). We now iterate through each seasonal component again. On each iteration we add this single seasonal component back to the fully de-seasonalised time series from the end of Step 1 (Fig. 7b). Then extract the same seasonal component back from this time series using STL (Fig. 7c). Subtract this new estimate of the seasonal component from the time series so that it is, once again, fully de-seasonalised (Fig. 7d).

This step is helpful because the time series that we now pass to STL only contains the single seasonal component of interest, the trend, and noise. This makes it easier for STL to re-capture any part of the seasonal component that it missed in Step 1.

Repeat this step N times, in [1] N = 2 is used.

![Fig. 7. Step 2 of the decomposition step of the MSTL algorithm. Refine each seasonal component by adding it back to the de-seasonalised time series and re-extracting the seasonal component using STL. This figure shows one iteration of this step with the daily seasonality. Image by author.]({{ site.baseurl }}/assets/images/posts/multi-seasonal-time-series-decomposition-using-mstl-in-python/fig07.webp)

Fig. 7. Step 2 of the decomposition step of the MSTL algorithm. Refine each seasonal component by adding it back to the de-seasonalised time series and re-extracting the seasonal component using STL. This figure shows one iteration of this step with the daily seasonality. Image by author.

**Step 3: Extract the trend.** Extract the trend component from the last STL fit that occurs in Step 2 (i.e., for the longest period seasonal component).

**Step 4: Extract the residual.** The residual component is calculated by subtracting the trend component from the fully de-seasonalised time series at the end of Step 2.

After these 4 steps we are left with a trend component, a set of refined seasonal components, and a residual component.

Note: in the original paper [1] there is another branch of the algorithm that deals with the case when the user specifies that there is no seasonal component. In this case, the time series is just smoothed to provide a trend component and no seasonal component is returned.

## MSTL parameters

In practice, there are only a few parameters that you need to know to use MSTL in Python. Let’s summarise the most important parameters:

- `periods`: the period of each seasonal component, `period`, passed to STL (e.g., for hourly data with daily and weekly seasonality we set `periods = (24, 24*7)`);
- `windows`: the seasonal window sizes, `seasonal`, passed to STL for each respective seasonal component (e.g., `windows = (11, 15)`). MSTL uses default values based on experiments which gave the best results in [1];
- `seasonal_deg`: the degree of the polynomial for the seasonal component used in [1] is `seasonal_deg = 0` ;
- `lmbda`: if a Box Cox transform is required then we need to pick a value for the λ parameter of a Box Cox transform. λ can be set manually, `lmbda=0.7` , or automatically, `lmbda="auto"` .

## MSTL applied to electricity demand data in Python

### Electricity demand in Victoria, Australia, dataset

We will now look at a real-world dataset where we can apply MSTL. Let’s take a look at the electricity demand in Victoria, Australia, dataset used in [1]. We can find the original data where the demand is recorded at a half-hourly granularity [here](https://github.com/tidyverts/tsibbledata/tree/master/data-raw/vic_elec/VIC2015) [9]. We follow the same steps as [1]: we resample the data to hourly and restrict our view to the first half of 2012 to focus on daily and weekly seasonality. All the code for the following sections can be found [here](https://github.com/KishManani/MSTL).

Let’s plot our time series (Fig. 8) and build intuition about any seasonality in the data.

![Fig. 8. Hourly electricity demand (megawatts) in Victoria, Australia between January and May 2012. Image by author.]({{ site.baseurl }}/assets/images/posts/multi-seasonal-time-series-decomposition-using-mstl-in-python/fig08.webp)

Fig. 8. Hourly electricity demand (megawatts) in Victoria, Australia between January and May 2012. Image by author.

We expect there to be a daily seasonality associated with electricity demand. Let’s confirm this by plotting the hourly demand for each day and segregating it by month.

![Fig. 9. Daily seasonality. The demand for each day in the respective month is shown by the light blue lines. The mean demand is shown by the dark blue line. There is a daily pattern where there is more demand in the day than late at night. Image by author.]({{ site.baseurl }}/assets/images/posts/multi-seasonal-time-series-decomposition-using-mstl-in-python/fig09.webp)

Fig. 9. Daily seasonality. The demand for each day in the respective month is shown by the light blue lines. The mean demand is shown by the dark blue line. There is a daily pattern where there is more demand in the day than late at night. Image by author.

In Fig. 9, we see that there is daily seasonality. We can also see the daily seasonality change in time. In the summer months (e.g., January) there is a daily peak around 4 PM whereas in the winter months (e.g., May) there are now two peaks, one around 8 AM and another around 6 PM. One cause of the shift in daily seasonality may be the transition from air conditioning use in the summer to the use of heating units in the winter.

Now let’s plot the data to look at weekly seasonality.

![Fig. 10. Weekly seasonality. The demand for each week of the year is shown by the light blue line. The mean demand is shown by the dark blue line. Weekends appear to have lower demand on average than weekdays. Image by author.]({{ site.baseurl }}/assets/images/posts/multi-seasonal-time-series-decomposition-using-mstl-in-python/fig10.webp)

Fig. 10. Weekly seasonality. The demand for each week of the year is shown by the light blue line. The mean demand is shown by the dark blue line. Weekends appear to have lower demand on average than weekdays. Image by author.

In Fig. 10, we see that there is weekly seasonality, that is, there is less demand on weekends than on weekdays.

These plots have confirmed the presence of daily and weekly seasonality and have given us information about how we expect them to vary. Let’s move on and use MSTL to extract these seasonalities.

### Using MSTL to decompose electricity demand data

We can import MSTL like so:

```
from statsmodels.tsa.seasonal import MSTL
```

The main parameter that we need to specify is `periods` which is the period of each seasonal component in the time series. We expect there to be daily and weekly seasonality, therefore, we set `periods = (24, 24*7)`. We can also set the parameters which are fed to the underlying STL model by passing a dictionary to `stl_kwargs`.

```
stl_kwargs = {"seasonal_deg": 0}
model = MSTL(data, periods=(24, 24 * 7), stl_kwargs=stl_kwargs)
res = model.fit()
```

The trend, seasonal, and residual components are all accessible from the results object `res` :

```
seasonal = res.seasonal # contains both seasonal components
trend = res.trend
residual = res.resid
```

Let’s plot the decomposition using `res.plot()` (Fig. 11).

![Fig. 11. The MSTL decomposition of the electricity demand time series into trend, daily seasonality (Seasonal_24), weekly seasonality (Seasonal_168), and a residual component. Image by author.]({{ site.baseurl }}/assets/images/posts/multi-seasonal-time-series-decomposition-using-mstl-in-python/fig11.webp)

Fig. 11. The MSTL decomposition of the electricity demand time series into trend, daily seasonality (Seasonal_24), weekly seasonality (Seasonal_168), and a residual component. Image by author.

Let’s inspect the daily seasonal component from MSTL and check if it captures the intuition that we got from the daily plots in Fig. 9 (i.e., in the summer there is a daily peak at 4PM and in the winter there are two peaks).

![Fig. 12. The daily seasonal component extracted by MSTL during a sample of days in January and May. Image by author.]({{ site.baseurl }}/assets/images/posts/multi-seasonal-time-series-decomposition-using-mstl-in-python/fig12.webp)

Fig. 12. The daily seasonal component extracted by MSTL during a sample of days in January and May. Image by author.

We see that MSTL has captured the daily seasonality correctly and even shows the additional daily peak in May (Fig. 12). This shows the usefulness of MSTL in being able to model seasonal components which change in time.

Let’s inspect the weekly seasonal component from MSTL and check if it captures the intuition that we got from the weekly plots in Fig. 10 (i.e., there is less demand during weekends than weekdays).

![Fig. 13. The weekly seasonal component extracted by MSTL over a 3 week period in January and May. The weekly seasonal component captured the dips on weekends and is stable across the two months. We can still see some daily seasonality that has leaked into the weekly component. Image by author.]({{ site.baseurl }}/assets/images/posts/multi-seasonal-time-series-decomposition-using-mstl-in-python/fig13.webp)

Fig. 13. The weekly seasonal component extracted by MSTL over a 3 week period in January and May. The weekly seasonal component captured the dips on weekends and is stable across the two months. We can still see some daily seasonality that has leaked into the weekly component. Image by author.

In Fig. 13, we see that the weekly seasonal component from MSTL has indeed been able to capture the dip in demand on weekends.

## Conclusion

There we have it! MSTL is a great tool to analyse time series with multiple seasonal components. In this article, we saw how we can use MSTL to decompose a multi-seasonal time series in the real world using Python.

If you made it this far, thank you for reading and I hope you found this helpful.

*If you want to know more about Loess, STL, and MSTL then they are covered in more detail in this course on* [*Feature Engineering for Time Series Forecasting*](https://www.trainindata.com/p/feature-engineering-for-forecasting)*.*

## Code

Notebook: <https://github.com/KishManani/MSTL>

## References

[1] [Bandara, K., Hyndman, R.J. and Bergmeir, C., 2021. MSTL: A Seasonal-Trend Decomposition Algorithm for Time Series with Multiple Seasonal Patterns. *arXiv preprint arXiv:2107.13462*.](https://arxiv.org/abs/2107.13462)

[2] Cleveland, R.B., Cleveland, W.S., McRae, J.E. and Terpenning, I., 1990. STL: A seasonal-trend decomposition. *J. Off. Stat*, *6*(1), pp.3–73.

[3] Bell, W.R. and Hillmer, S.C., 1984. Issues involved with the seasonal adjustment of economic time series. *Journal of Business & Economic Statistics*, *2*(4), pp.291–320.

[4] Taylor, S.J. and Letham, B., 2018. Forecasting at scale. *The American Statistician*, *72*(1), pp.37–45.

[5] De Livera, A.M., Hyndman, R.J. and Snyder, R.D., 2011. Forecasting time series with complex seasonal patterns using exponential smoothing. *Journal of the American statistical association*, *106*(496), pp.1513–1527.

[6] Dokumentov, A. and Hyndman, R.J., 2021. STR: Seasonal-Trend Decomposition Using Regression. *INFORMS Journal on Data Science*.

[7] Cleveland, W.S. and Devlin, S.J., 1988. Locally weighted regression: an approach to regression analysis by local fitting. *Journal of the American statistical association*, *83*(403), pp.596–610.

[8] <http://www.gardner.fyi/blog/STL-Part-II/>

[9] O’Hara-Wild, M., Hyndman, R.J., Wang, E., 2021. tsibbledata: Diverse Datasets for ’tsibble’. URL: <https://CRAN.R-project.org/package=tsibbledata>. R package version 0.3.0. ([Creative Commons License](https://zenodo.org/record/4659727#.Ymm92fPMJmp))
