## Abstract

Our unique food routines shape who we are, whether you use carefully planned shopping lists or let creativity direct your grazing. With the help of the shopping purchasing and delivery app Instacart, you can easily keep your pantry and refrigerator stocked with your preferred foods and essentials. After you choose goods using the Instacart app, personal shoppers evaluate your order, complete the in-store shopping, and transport your items for you.
This project's main goal is to forecast the purchases that are regularly reordered. 

## Idea

- There is a sizable market for and possibility for internet commerce. In order to comprehend consumer behavior, market patterns, and machine learning for product suggestions, businesses have been using data analysis. 

- In order to learn more about consumer buying patterns and market trends, we would like to evaluate data from online food purchasing. 

- Using categorization methods, we want to use this anonymized data on customer purchases over time to forecast which previously bought items will be in a user's next transaction. 

- In addition, we'd like to use association rule mining to create a product recommendation system that suggests items for a specific user based on their order history and reviews from other users who have made comparable purchases. 

## About Data 

An online food purchasing dataset is obtained for the project from Kaggle.  The following link will take you there: https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset

* The dataset is 1.28 GB in size, and it contains approximately 3.4 million unique transactions.
* Each CSV file in the dataset describes a distinct element of the data, including aisles, departments, orders, goods, and transaction behavior. The dataset is made up of 5 CSV files. A distinct id is connected to each object (customer, item, transaction, aisle, etc.). 
* The majority of the variable and file titles should be self-explanatory.

  1. aisles.csv(aisle id, aisle description)
  2. departments.csv(dept id, dept name)
  3. order_products.csv(order_id, product_id, add_to_cart_order, reordered)
  4. orders.csv(order_id, user_id, eval_set, order_number, order_dow, order_hour_of_delay, days_since_prior)
  5. products.csv(product_id, product_name, aisle_id, dept_id)
  
## Exploratory Data Analysis.

### We have following DataFrames:

   - orders: This table includes all orders, namely prior, train, and test. It has single primary key (order_id).
   - order_products_train: This table includes training orders. It has a composite primary key (order_id and product_id) and indicates whether a product        in an order is a reorder or not (through the reordered variable).
   - order_products_prior : This table includes prior orders. It has a composite primary key (order_id and product_id) and indicates whether a product in      an order is a reorder or not (through the reordered variable).
   - products: This table includes all products. It has a single primary key (product_id)
   - aisles: This table includes all aisles. It has a single primary key (aisle_id)
   - departments: This table includes all departments. It has a single primary key (department_id)

### Ratio of reordered products

  1. 58.9697% of products in prior orders were re-ordered.
  2. 59.8594% of products in train orders were re-ordered.
  3. 11.868056% of orders having no reordered product.
  4. 21.493998% of orders having all product re-ordered.
  5. Most baskets contain from 5-8 products.
  6. Products placed first in cart are the products mostly reordered.

### Analyzing the count of reordered products in a basket

  1. Most baskets have from 0-5 reordered product
  2. Probability of a basket having 0 reordered products: 11.868056
  3. Probability of a basket having 1 reordered products: 10.262626
  4. Probability of a basket having 2 reordered products: 10.326791
  5. Probability of a basket having 3 reordered products: 9.735981
  6. Probability of a basket having 4 reordered products: 8.777965

### Time effects on purchasing behaviours

  - How Time affects the purchasing behaviour of customers?
  - Most orders are ordered on Day 0 and Day 1.
  - Orders are mostly ordered during day, from 9:00 AM to 4:00 PM.
  - Peak orders happens at Saturday afternoon (1:00PM), and Sunday morning (10:00AM)
  - By more than 65%, People usually buy previously ordered products from 6:00AM to 8:00AM
  - Most users make orders after a week from their last order. or from a month of their last order.
  - After a week from the last order, the probability of reordering within the same month is small. 
      - Send reminders of the most likely ordered products within a week, to catch the high prob of a customer to make their next order.
  - The Next order has higher probability to be during 10 days from the current order.

### Association between days since last order and the ratio of reorders

  - 74 % of products bought at the same day of prev order, are reorders.
  - 69% of products bought after 1 week of prev order, are reorders
    #### Conculsion:
      - If future order will be at the same day of prev order, percentage of reorders in the future product is high.
      - If future order will be after a week from the prev order, percentage of reorders in the future product is high.


### Analyzing products

   #### How often a product is purchased?
   
   - 5 Most Ordered Products
        - Banana
        - Bag of Organic Bananas
        - Organic Strawberries
        - Organic Baby Spinach
        - Organic Hass Avocado
        - 14% of all orders contains bananas.  
   - Organic products are frequently ordered.

   #### How often a product is the first item purchased?
  
   - 5 Most Add to Cart First Products
        - Banana
        - Bag of Organic Bananas
        - Organic Whole Milk
        - Organic Strawberries
        - Organic Hass Avocado
   - 3.4% of the orders, Banana is being the first product added to cart. Products contianing milk have very high probability to be reordered. Organic          products have very high probability to be reordered.

### Analyzing Organic Prodcuts

   - 10% of instacart's products are organic products
   - 31.5% of bought products are organic products
   - 67% probability of reordering an organic products.
   - 61% probability of reordering an non-organic products.
   - No significance pattern of when organic products are bought most, than when products in general are bought most.

### Purchasing behaviour on Departments and Aisles

   - Count of products in each department
