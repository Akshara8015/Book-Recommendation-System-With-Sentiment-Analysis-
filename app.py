import streamlit as st

# from utils.functions import recommend, get_book_page, get_review_page, reviews_to_stars
from utils.storage import recommend, get_book_page, get_review_page, reviews_to_stars
# from utils.storage import load_data, books_reviews, book_img, recommend, get_book_page, get_review_page, reviews_to_stars

# Load saved data at start
# load_data()

st.set_page_config(page_title="Book Recommender", layout="wide")

st.title("📚 Book Recommendation & Review Intelligence System")

book_name = st.text_input("Enter a Book Name")

if st.button("Get Recommendations"):

    if not book_name.strip():
        st.warning("Please enter a valid book name")

    else:
        with st.spinner("Processing..."):

            try:
                recommended_books = recommend(book_name)
                book_rating = {}
                book_img_link = {}

                for book in recommended_books:
                    page_link = get_book_page(book)
                    print(page_link)
                    img_link, reviews = get_review_page(page_link)
                    book_img_link[book] = img_link

                for book in recommended_books:
                    rating = reviews_to_stars(reviews)
                    book_rating[book] = rating if rating else "Not Rated"

                st.success("Done!")

                # Display
                cols = st.columns(3)

                for idx, book in enumerate(recommended_books):
                    with cols[idx % 3]:

                        img = book_img_link[book]

                        st.image(img_link)
                        st.subheader(book)

                        rating = book_rating[book]
                        st.write(f"⭐ {rating}")

                        if isinstance(rating, float):
                            if rating >= 4:
                                st.write("😊 Positive")
                            elif rating >= 2.5:
                                st.write("😐 Mixed")
                            else:
                                st.write("😞 Negative")

            except Exception as e:
                st.error(str(e))

