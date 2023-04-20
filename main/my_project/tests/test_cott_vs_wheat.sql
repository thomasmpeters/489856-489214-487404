SELECT avg_correlation
FROM {{ref('AverageCottonvsWheatCorr')}}
WHERE avg_correlation > 1 AND avg_correlation < -1