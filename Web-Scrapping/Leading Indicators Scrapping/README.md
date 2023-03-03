# Economic Leading Indicators Scrapper

This script/scraper is helpful for students and professionals in the finance industry who require financial data for their research or investment decisions. The scraper can be used to extract data on various financial indicators, such as GDP growth rates, inflation rates, and interest rates,.. which can be used to analyze economic trends and make informed investment decisions. The extracted data can be analyzed using statistical and econometric models, such as time-series models, regression models, and machine learning algorithms, to build forecasts and make predictions. 


## Description

This code is a Python script that automate the process of retrieving, processing and storing financial data from the [Fred](https://fred.stlouisfed.org/) and [yale](http://www.econ.yale.edu/~shiller/) on [AWS] (https://aws.amazon.com/)RDS Database

The script starts by importing several Python libraries, such as requests, pandas, openpyxl, json, numpy, datetime, holidays, os, pyquery, math, sys, datetime and sqlalchemy. Then, it sets the current directory to a variable path.

The script defines two functions:

convert_yyyyq(dates): This function takes a Pandas series of date values (in the format 'yyyyq', where 'q' is the quarter number) and returns a corresponding Pandas series of datetime objects.

pull_data(data): This function takes a Pandas dataframe data containing information about financial data (including its frequency, source and the name of the data) and returns three Pandas dataframes daily, monthly, quarterly with the corresponding financial data. It also prints success messages to the console while retrieving the data from the sources.

After defining the functions, the script reads an Excel file with financial data using the read_excel method from the pandas library. It then loops through each sheet in the Excel file, retrieves the data using the pull_data function, and stores it into a SQL database using the create_engine function from the sqlalchemy library.

You can find a sample output called Data Summary in this Repo


## Getting Started

### Dependencies

* pandas, openpyxl, json, numpy, datetime, holidays, os, pyquery, math, sys, datetime and sqlalchemy
* You need to create a fred account to get your own API

### Installing

* Get your FRED API
* Create an AWS RDS Database. There is a free version but be careful as sometimes they will charge if you hit certain limits.




## Authors


 Mike Remezo
 [@Mike Remezo](https://www.linkedin.com/in/mike-remezo)



## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments


* Ascentris Real Estate Private Equity