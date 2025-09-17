# perfume-recommender-ai
Recommender system using item-based collaborative filtering to recommend 5 perfumes you will like

Data was obtained from scraping >20k reviews and ~600 women fragrance products from ULTA website, using Requests and Selenium. (Scraping done in September 2025) 
After data cleaning and exploratory data analysis to uncover rating/review trends, predictive model was built using item-based collaborative filtering with SVD. 
Model was deployed allowing user to select a fragrance in a dropdown list, and generating a top-5 personalized recommendations for next product to try. 
Model performance was evaluated with ranking metrics (precision @5, recall @5, MAE @5) and show strong performance.

In parallel to deployment, I keep on working on refining model performance, and will push updated versions as I iterate.
