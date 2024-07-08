import numpy as np
import pickle
import pandas as pd
import streamlit as st 
from ProductRecommendor import prod_recommender, product

pickle_in = open("model.pkl","rb")
model = pickle.load(pickle_in)

# load mapping of product names to product IDs
products_df = pd.read_csv("data/all_orders.csv")
product_names_to_ids = dict(zip(products_df["product_name"], products_df["product_id"]))

def product_name_to_id(product_name):
    return product_names_to_ids[product_name]



def pred(user_id, product_name, lift, recc):
    # convert product name to product ID
    try:
        product_id = product_names_to_ids[product_name]
    except KeyError:
        return "Not associated in recommendation"
    
    prediction = model(user_id, product_id, lift, recc)
    return prediction


def main():
    st.title("Product Recommendation")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Product Recommendation
     </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    user_id = int(st.number_input("User ID",step=1,min_value=0))
    
    # create dropdown list for product names
    product_name = st.selectbox("Product Name", options=products_df["product_name"].unique())
    
    lift = float(st.number_input("Lift",step=0.1,format="%.2f", min_value=0.0))
    recc = int(st.number_input("No of Recommendations",step=1,min_value=0))

    result = ""
    if st.button("Predict"):
        try:
            # convert product name to product ID
            product_id = product_name_to_id(product_name)
            result = pred(user_id, product_name, lift, recc)
        except KeyError:
            result = "Not associated in recommendation"
    st.success("The output is {}".format(result))


if __name__=='__main__':
    main()
