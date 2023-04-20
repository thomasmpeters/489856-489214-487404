SELECT month, price
FROM {{ref('5yearmodel')}}
WHERE commodity = 'Corn' AND price < 0 AND price >1