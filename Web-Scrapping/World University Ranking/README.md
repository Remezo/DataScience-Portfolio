# Scraping and Analyzing thr Round University Ranking(RUR)

This analysis provides valuable insights for students who are looking for study abroad programs or graduate schools. By identifying top-performing universities outside the USA, students can expand their options and make informed decisions about their future academic endeavors. Additionally, the stacked barplot and word cloud analysis provide a deeper understanding of the universities' priorities and areas of focus, which can be valuable information for students who are looking to align their academic goals with those of the universities they are considering.


## Description

The Round University Ranking (RUR) is a widely recognized ranking of higher education institutions across the world. By evaluating the performance of over 867 universities, RUR provides a comprehensive analysis of various key areas of university activity including teaching, research, international diversity, and financial sustainability. Using web scraping techniques, we were able to extract valuable data fields such as university names, scores, country, and league rankings from the RUR website.

After filtering out universities located in the USA, we retained only those in the Diamond, Gold, Silver, and Bronze league rankings. This allowed us to identify the top-performing universities outside the USA, providing valuable information for students who are looking for study abroad programs or as a first step to look for graduate schools.

We then created a stacked barplot to visualize the proportion of universities in each league for different countries. This allowed us to identify countries with a high concentration of universities in the top leagues, providing additional information to students who are considering study abroad programs. We also eliminated countries that had fewer than 10 universities in the DataFrame and sorted the countries based on the number of universities they had in the dataset.

Finally, we analyzed mission statements of the top-ranking universities by creating a word cloud. This allowed us to identify the most commonly used words in their mission statements, which could be indicative of the universities' priorities and areas of focus. By omitting the word "university" from the data, we obtained a more focused word cloud that provided additional insights into the universities' goals and objectives.



## Getting Started

### Dependencies

* Python .3.5
* Selenium and Beautiful Soup
* Pandas and Selenium



## Help

It is way easier to run this in a conda environment.


## Authors



ex. Mike Remezo  
ex. [@MikeRemezo](https://www.linkedin.com/in/mike-remezo/)

