# EDA Findings

Dataset: **250,000 listings, 42 cities, 20 states, 500 localities**. Prices are in lakhs of INR; price per sq ft is in INR.

> **Important context:** the supplied data behaves like a synthetic dataset - most raw features are statistically independent of price, so many relationships below are weak or absent. The analysis reports what the data shows rather than what a real market would be expected to show.

## 1-5. Price & Size Analysis

### Q1. What is the distribution of property prices?

![Q1. What is the distribution of property prices?](reports/eda/q01_price_distribution.png)

Prices span 10.0 to 500.0 lakhs (mean 254.6, median 253.9, skew +0.01). The distribution is almost flat, i.e. close to uniform, rather than the right-skewed shape typical of real markets.

### Q2. What is the distribution of property sizes?

![Q2. What is the distribution of property sizes?](reports/eda/q02_size_distribution.png)

Sizes range from 500 to 5,000 sq ft (mean 2,750, median 2,747, skew +0.00); again close to uniform.

### Q3. How does price per sq ft vary by property type?

![Q3. How does price per sq ft vary by property type?](reports/eda/q03_ppsf_by_type.png)

| Property type | Mean INR/sqft | Median INR/sqft |
|---|---|---|
| Apartment | 11,810 | 9,245 |
| Independent House | 11,857 | 9,305 |
| Villa | 11,776 | 9,190 |

The spread between types is 0.7%, so property type has little effect on price per sq ft in this dataset.

### Q4. Is there a relationship between property size and price?

![Q4. Is there a relationship between property size and price?](reports/eda/q04_size_vs_price.png)

Pearson r between size and price = **-0.003** (negligible). Spearman rho = -0.003. Larger properties are **not** priced higher here, which is unrealistic for a real market and means price is driven by something other than size in this synthetic data (so price per sq ft is dominated by size: larger area -> lower price per sq ft).

### Q5. Are there any outliers in price per sq ft or property size?

![Q5. Are there any outliers in price per sq ft or property size?](reports/eda/q05_outliers.png)

IQR rule: **19,723** price-per-sq-ft outliers (7.9%), **0** size outliers, **0** price outliers. The outliers in price per sq ft come from small, expensive listings. They are flagged (`PPSF_Outlier`) and the modelling feature is winsorised at the IQR fences.

## 6-10. Location-based Analysis

### Q6. What is the average price per sq ft by state?

![Q6. What is the average price per sq ft by state?](reports/eda/q06_ppsf_by_state.png)

Highest: **Karnataka** (INR 13,252/sqft); lowest: **Punjab** (INR 12,931/sqft). Spread between states is only 2.5%.

### Q7. What is the average property price by city?

![Q7. What is the average property price by city?](reports/eda/q07_price_by_city.png)

Most expensive city: **Bangalore** (258.5 lakhs); cheapest: **Cuttack** (250.8 lakhs). All 42 cities average between 251 and 258 lakhs (axis truncated for readability) - city has almost no influence on price in this data.

### Q8. What is the median age of properties by locality?

![Q8. What is the median age of properties by locality?](reports/eda/q08_median_age_by_locality.png)

Locality medians range from 16 to 21 years (overall median 18). Oldest: **Locality_72** (21 yrs); newest: **Locality_443** (16 yrs).

### Q9. How is BHK distributed across cities?

![Q9. How is BHK distributed across cities?](reports/eda/q09_bhk_by_city.png)

Overall share: 1 BHK 20.1%, 2 BHK 19.9%, 3 BHK 20.0%, 4 BHK 19.9%, 5 BHK 20.1%. Every city has a near-identical, roughly uniform mix of 1-5 BHK.

### Q10. What are the price trends for the top 5 most expensive localities?

![Q10. What are the price trends for the top 5 most expensive localities?](reports/eda/q10_top5_locality_trend.png)

Top 5 by mean price: **Locality_395** (275.4 lakhs), **Locality_366** (273.4 lakhs), **Locality_497** (270.0 lakhs), **Locality_360** (269.8 lakhs), **Locality_203** (269.0 lakhs). The dataset has no listing date, so the trend is shown against construction period; the lines fluctuate without a consistent direction.

## 11-15. Feature Relationship & Correlation

### Q11. How are numeric features correlated with each other?

![Q11. How are numeric features correlated with each other?](reports/eda/q11_correlation_matrix.png)

Strongest pairs: Size_in_SqFt / Price_per_SqFt_INR = -0.66; Price_in_Lakhs / Price_per_SqFt_INR = +0.64; Transport_Score / Infrastructure_Score = +0.61; Nearby_Schools / Infrastructure_Score = +0.51; Floor_No / Total_Floors = +0.50. Price per sq ft is strongly (negatively) tied to size and (positively) to price because it is derived from them; the Infrastructure_Score pairs and the Floor_No / Total_Floors pair are likewise by construction (the score is built from schools and transport; the floor fix puts the smaller of the two raw values in Floor_No). Apart from such derived features, raw features are essentially independent.

### Q12. How do nearby schools relate to price per sq ft?

![Q12. How do nearby schools relate to price per sq ft?](reports/eda/q12_schools_vs_ppsf.png)

Pearson r between nearby schools and price per sq ft = **-0.002** (negligible). Average price per sq ft moves between INR 11,707 and INR 11,920 across 1-10 schools, with no systematic trend.

### Q13. How do nearby hospitals relate to price per sq ft?

![Q13. How do nearby hospitals relate to price per sq ft?](reports/eda/q13_hospitals_vs_ppsf.png)

Pearson r between nearby hospitals and price per sq ft = **-0.000** (negligible). Average price per sq ft moves between INR 11,702 and INR 11,964 across 1-10 hospitals, with no systematic trend.

### Q14. How does price vary by furnished status?

![Q14. How does price vary by furnished status?](reports/eda/q14_price_by_furnished.png)

Average price: Furnished 254.4 lakhs, Semi-furnished 254.3 lakhs, Unfurnished 255.0 lakhs. Differences are within 0.3%.

### Q15. How does price per sq ft vary by property facing direction?

![Q15. How does price per sq ft vary by property facing direction?](reports/eda/q15_ppsf_by_facing.png)

Average INR/sqft: East 11,797, North 11,793, South 11,790, West 11,878. Best: **West**, but the gap to the lowest is only 0.7%.

## 16-20. Investment / Amenities / Ownership Analysis

### Q16. How many properties belong to each owner type?

![Q16. How many properties belong to each owner type?](reports/eda/q16_owner_type.png)

**Broker**: 83,479 (33.4%); **Owner**: 83,268 (33.3%); **Builder**: 83,253 (33.3%).

### Q17. How many properties are available under each availability status?

![Q17. How many properties are available under each availability status?](reports/eda/q17_availability.png)

**Under_Construction**: 125,035 (50.0%); **Ready_to_Move**: 124,965 (50.0%). (The data has only these two statuses; there is no 'Sold' category.)

### Q18. Does parking space affect property price?

![Q18. Does parking space affect property price?](reports/eda/q18_parking_vs_price.png)

With parking: 254.7 lakhs; without: 254.4 lakhs (difference +0.12%). Parking has no meaningful effect on price in this dataset.

### Q19. How do amenities affect price per sq ft?

![Q19. How do amenities affect price per sq ft?](reports/eda/q19_amenities_vs_ppsf.png)

Pearson r between amenity count and price per sq ft = **+0.002** (negligible). The largest with/without gap is for **Gym** (+0.6%) - negligible in practice.

### Q20. How does public transport accessibility relate to price per sq ft or investment potential?

![Q20. How does public transport accessibility relate to price per sq ft or investment potential?](reports/eda/q20_transport.png)

| Accessibility | Avg INR/sqft | % Good investment | Avg annual growth |
|---|---|---|---|
| Low | 11,842 | 27.6% | 8.02% |
| Medium | 11,816 | 40.6% | 8.40% |
| High | 11,785 | 52.1% | 8.77% |

Transport access has no effect on price per sq ft, but it clearly raises investment potential (by construction of the growth model and investment score).

## Additional Step-2 Analyses

### Price trends by city

![Price trends by city](reports/eda/x1_price_trends_by_city.png)

Highest modelled growth: **Pune** (9.54%/yr); lowest: **Bilaspur** (7.51%/yr). Current prices show no consistent trend over construction period.

### Correlation between area and investment return

![Correlation between area and investment return](reports/eda/x2_area_vs_return.png)

Pearson r between size and 5-year appreciation = **+0.001** (negligible). Pearson r between size and Good Investment = **+0.382** (moderate). Larger homes have a lower price per sq ft, so they score better on the 'value' component of the investment score, which is why size relates to the Good Investment label even though it does not relate to appreciation.

### Impact of crime rate on Good Investment classification (proxy)

![Impact of crime rate on Good Investment classification (proxy)](reports/eda/x3_safety_vs_investment.png)

**The dataset has no crime-rate column**, so security features and the infrastructure score are used as a safety proxy. Good-investment rate: 43.2% with security vs 37.0% without; it rises from 18.9% to 56.4% across infrastructure bands.

### Relationship between infrastructure score and resale value

![Relationship between infrastructure score and resale value](reports/eda/x4_infrastructure_vs_resale.png)

Pearson r between infrastructure score and current price = **+0.000** (negligible). Pearson r between infrastructure score and 5-year resale value = **+0.025** (negligible). Better infrastructure raises the modelled growth rate only slightly, so resale value is almost unchanged.
