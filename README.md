# Welcome to Commidity Correlation Tracker


## Commands

* `1. python main.py` - create the required API calls, CSV files, and formatted/melted files 
* `2. create_table.py` - Start the postgresSQL functions to upload formatted tables into SQL
* `3. dbt run` - Run dbt to upload data into pgAdmin4
* `4. dbt tests` - Run dbt tests

    main.py - start here and run "python main.py" in order to generate the proper CSV files, "melted_commodity_prices.csv"
    create_table.py  - run this after to initlialize the data for use in postgressql


## Project layout

Listed below are the model functions we wrote to test our implementation of our API, postgresql, and dbt

--------------------------
489856-489215-487404/

    main/my_project/target/

                /compiled/

                    models/

                        5YCorn.sql  #takes the most recent 5 year prices of corn against the the months the correspond with

                        5YCotton.sql #takes the most recent 5 year prices of cotton against the the months the correspond with

                        5YWheat.sql #takes the most recent 5 year prices of wheat against the the months the correspond with

                        5YWheatvsCorn.sql  #takes the most recent 5 year prices of wheat vs corn and finds the correlation of the two

                        5Ycorrelation.sql #finds the 5 year correlation of wheat vs. corn price over a 5y period

                        5yearmodel.sql #our reference - we use this to pull all commodities prices over the last 5 years

                        AverageCottonvsWheatCorr.sql #calculates the AVERAGE correlation on a 3 month timeframe of Cotton vs. Wheat

                        CornVsWheat.sql #calculates the correlation on a 3 month timeframe of Corn vs. Wheat

                        CottonvsWheat.sql #calculates the correlation on a 3 month timeframe of Cotton vs. Wheat

                        model1.sql #all data from table, should we need it      


## Brief Overview

Our ETL pipeline starts when we pull commodity data from alphaVantage API - here we are pulling ALL data we can on the following commodities:

-Brent Oil
-Crude Oil
-Corn
-Cotton
-Wheat

In this program, we are looking to extract the correlations between any combinations of the commodities, so that statistically better predictions of one 
commodity could be made with information about a correlating commodity. Once we have this data, we format, melt, and export the dataframe with pandas into 
postgreql, which is then inserted into a localhost sql database which allows for us to pull data and correlations quicker than we could otherwise in python.

One serious bottleneck we have is that not all of the commodities were listed in the same year (or decade), so we are starting at the year 1990, when all of the five commodities were officially listed on publically traded exchanges held in alphaVantage's database. 

We are more interested in the last five years of data, because though we are not macroeconomic experts, we think it would be fair to start from a time period where COVID affected many aspects of the stock market.

Another bottleneck in our ETL pipeline is the fact that we are only uploading it via .csv file with postgresql. If we were set on having this program be the fastest version of itself, we would likely use bulk insert statements, or directly taking data from the api into our server. We will likely get this working for the last part of our project. 

For our DAG, we think that there could be a bottleneck in the way we have to pull five years of data each time we want to run statistical comparisons on prices, which we attempted to solve by using the {{ ref ('xxx') }} command in DBT.
## Group Strengths and Weaknesses

Our group broke up the project very evenly - however was not able to meet in person synchronously as much as we would have liked. Michael implemented the postgresql, and Jon and I (Tom) implemented DBT implementations and modules. I think if we were all more flexible on time (as this is the reason it was split by postgres/dbt), I think we would have all worked on each part together more comprehensively. On top of this, we ran into some troubles installing pgAdmin4, which resulted in Jon and I (Tom) using liveshare to work and submit our work. For the future parts of the project, we will likely carve out time in all of our schedules to work on this so that we can all have greater understanding of postgresql, dbt, and mkdocs. 

As for strengths, I think that we all have a more thorough undrestanding of the individual topics we did cover (dbt, postgres) because of our need to implement them separately.

Thank you so much, and please feel free to contact us if you have any questions.
