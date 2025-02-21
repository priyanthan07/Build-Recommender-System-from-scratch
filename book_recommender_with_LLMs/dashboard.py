import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())

def retrieve_semantic_recommendation(
                        query: str,
                        category: str = None,
                        tone: str = None,
                        initial_top_k: int = 50,
                        final_top_k: int = 16,
                    ) -> pd.DataFrame:
    
    recommendations = db_books.similarity_search(query, k=initial_top_k)
    rec_books_isbn_list = [int(recommendation.page_content.strip('"').split()[0]) for recommendation in recommendations]
    books_rec_list = books[books["isbn13"].isin(rec_books_isbn_list)].head(final_top_k)
    
    if category != "All":
        books_rec_list = books_rec_list[books_rec_list["simple_categories"] == category].head(final_top_k)
    else:
        books_rec_list = books_rec_list.head(final_top_k)

    if tone == "Happy":
        books_rec_list.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        books_rec_list.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        books_rec_list.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        books_rec_list.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        books_rec_list.sort_values(by="sadness", ascending=False, inplace=True)

    return books_rec_list


def recommend_book(query: str, category:str, tone:str):
    
    recommendations = retrieve_semantic_recommendation(query, category, tone)
    results = []
    
    for _, row in recommendations.iterrows():
        desc = row["description"]
        desc_split = desc.split()
        truncated_desc = " ".join(desc_split[:30]) + "..."
        
        authers_split = row["authors"].split(";")
        if len(authers_split) == 2:
            authers_str = f"{authers_split[0]} and {authers_split[1] }"
        elif len(authers_split) >= 2:
            authers_str = f"{', '.join(authers_split[:-1])} and {authers_split[-1] }"
        else:
            authers_str=row["authors"]
            
        caption = f"{row['title']} by {authers_str}: {truncated_desc}"
        results.append((row["large_thumbnail"], caption))
    return results
        
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    submit_button.click(fn = recommend_book,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)


if __name__ == "__main__":
    dashboard.launch()
    